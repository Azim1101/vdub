package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.ensureActive
import java.io.File
import java.nio.LongBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.exp
import kotlin.random.Random

/**
 * Indri 0.1 — a 124M GPT-2 that emits Mimi codec tokens, decoded by
 * [MimiDecoder].
 *
 * The request is framed as a token sequence and the model continues it:
 *
 *     [text] <the text> [convert] [mimi] [spkr_NN]  ->  acoustic tokens… [stop]
 *
 * Those acoustic tokens are interleaved across 8 Mimi codebooks, one per step,
 * cycling. A mask enforces that: at step *k* only codebook `k % 8`'s 2048-wide
 * band is allowed (plus `[stop]`). Sampling without it produces ids from the
 * wrong codebook, which the decoder reads as a different sound entirely —
 * noise, not a wrong word.
 *
 * ### Preset voices only
 *
 * Indri conditions on a `[spkr_NN]` token, not on a reference clip, so it
 * cannot reproduce the original actor. [clonesVoice] is false and Step 5 warns
 * before a long run. Speakers are assigned round-robin so at least each
 * character keeps a distinct, consistent voice.
 *
 * ### Speed
 *
 * The export carries no KV cache — its only inputs are `input_ids`,
 * `attention_mask` and `position_ids` — so every token re-runs the entire
 * sequence, and cost grows quadratically with length. Measured at 6.6 tok/s on
 * two cores, i.e. ~15x slower than real time. [MAX_NEW_TOKENS] caps a runaway
 * line rather than letting one line consume an hour.
 */
class IndriTts private constructor(
    private val env: OrtEnvironment,
    private val languageModel: OrtSession,
    private val decoder: MimiDecoder,
    private val tokenizer: IndriTokenizer,
    private val textToken: Int,
    private val convertToken: Int,
    private val mimiToken: Int,
    private val stopToken: Int
) : TtsEngine {

    override val sampleRate = MimiDecoder.SAMPLE_RATE

    /** Preset speakers only — see the class comment. */
    override val clonesVoice = false

    companion object {
        /**
         * Hindi and Indian-English presets from the model card, in the order
         * they are handed out to speakers. Mixed genders so a two-hander does
         * not end up with two similar voices.
         */
        val PRESET_SPEAKERS = listOf(
            "[spkr_69]",   // IN male, book reader
            "[spkr_60]",   // IN female, book reader
            "[spkr_68]",   // IN male, book reader
            "[spkr_53]",   // IN female, recipe reciter
            "[spkr_70]",   // IN male, motivational speaker
            "[spkr_62]",   // IN male, book reader (heavy)
            "[spkr_75]",   // IN male, entrepreneur
            "[spkr_77]",   // IN male, influencer
            "[spkr_66]"    // IN male, politician
        )

        /**
         * About 24 s of audio at 8 tokens per 80 ms frame. Long enough for any
         * subtitle line; short enough that a model that fails to emit [stop]
         * gives up in minutes rather than hours.
         */
        const val MAX_NEW_TOKENS = 2400

        /** Upstream's generation config. */
        const val TEMPERATURE = 0.5f
        const val TOP_K = 15

        fun open(paths: VoiceEngine.Paths, threads: Int = 4): IndriTts {
            val missing = paths.missing
            check(missing.isEmpty()) {
                "Indri is incomplete — missing: " + missing.joinToString { it.name }
            }

            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            val lm = try {
                env.createSession(paths.languageModel.absolutePath, opts)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Could not load ${paths.languageModel.name}. Remove and " +
                        "re-download the Indri engine.\n\n" +
                        "(${e.message.orEmpty().take(300)})",
                    e
                )
            }

            val tok = IndriTokenizer.load(
                paths.vocabJson(), paths.mergesTxt(), paths.addedTokensJson()
            )
            fun need(name: String) = tok.id(name)
                ?: error("Indri tokenizer has no $name — the files are mismatched")

            return IndriTts(
                env = env,
                languageModel = lm,
                decoder = MimiDecoder.open(paths.mimiDecoder(), threads),
                tokenizer = tok,
                textToken = need("[text]"),
                convertToken = need("[convert]"),
                mimiToken = need("[mimi]"),
                stopToken = need("[stop]")
            )
        }

        /**
         * Upstream lowercases, collapses whitespace and de-duplicates trailing
         * punctuation. Reproduced exactly: the model saw only this form, and
         * uppercase text tokenizes to ids it was never trained on.
         */
        fun sanitize(text: String): String = text
            .lowercase()
            .replace(Regex("\\n+"), " ")
            .replace(Regex("[ \\t]+"), " ")
            .replace(Regex("([,.?])+"), "$1")
            .trim()
    }

    /** A preset speaker choice. There is nothing to enrol. */
    private class PresetVoice(val speakerToken: Int, val label: String) : TtsEngine.Voice

    private var assigned = 0

    /**
     * Hands out the next preset voice.
     *
     * [referenceWav] is ignored — this engine cannot clone. It is accepted so
     * the stage does not have to special-case the call.
     */
    override fun enrol(referenceWav: File?, transcript: String): TtsEngine.Voice {
        val label = PRESET_SPEAKERS[assigned % PRESET_SPEAKERS.size]
        assigned++
        val id = tokenizer.speakerId(label)
            ?: error("Indri has no speaker $label")
        return PresetVoice(id, label)
    }

    override suspend fun speak(
        text: String,
        voice: TtsEngine.Voice,
        language: String,
        exaggeration: Float,
        onToken: (Int) -> Unit
    ): FloatArray {
        val v = voice as? PresetVoice
            ?: error("enrol() this engine's own voice — got ${voice.javaClass.simpleName}")

        val clean = sanitize(text)
        require(clean.isNotEmpty()) { "nothing to speak" }

        val prompt = buildList {
            add(textToken)
            addAll(tokenizer.encode(clean).toList())
            add(convertToken)
            add(mimiToken)
            add(v.speakerToken)
        }.toIntArray()

        val promptLen = prompt.size
        val sequence = ArrayList<Long>(promptLen + 512)
        prompt.forEach { sequence.add(it.toLong()) }

        // Emotion nudges sampling temperature: a flatter distribution reads as
        // more measured, a sharper one as more animated. The model has no
        // explicit control, and this is the only honest lever available.
        val temperature = (TEMPERATURE * exaggeration).coerceIn(0.25f, 1.2f)
        val random = Random(SEED)
        val generated = ArrayList<Int>(512)

        for (step in 0 until MAX_NEW_TOKENS) {
            coroutineContext.ensureActive()

            val logits = runLm(sequence)
            val codebook = step % MimiDecoder.USED_CODEBOOKS
            val next = sampleInBand(logits, codebook, temperature, random)

            if (next == stopToken) break
            sequence.add(next.toLong())
            generated.add(next)
            if (generated.size % MimiDecoder.USED_CODEBOOKS == 0) {
                onToken(generated.size / MimiDecoder.USED_CODEBOOKS)
            }
        }

        // Only whole frames can be decoded; a partial one at the cap is noise.
        val whole = generated.size - generated.size % MimiDecoder.USED_CODEBOOKS
        check(whole >= MimiDecoder.USED_CODEBOOKS) {
            "Indri produced no audio for \"${text.take(40)}\""
        }

        val tokens = IntArray(whole) { generated[it] - IndriTokenizer.AUDIO_OFFSET }
        return decoder.decode(tokens)
    }

    /**
     * One forward pass over the whole sequence.
     *
     * There is no cache to reuse, so this is the entire cost of the stage:
     * `n` tokens means `n` passes over a sequence that keeps growing.
     */
    private fun runLm(sequence: List<Long>): FloatArray {
        val n = sequence.size
        val ids = LongArray(n) { sequence[it] }
        val mask = LongArray(n) { 1L }
        val positions = LongArray(n) { it.toLong() }
        val shape = longArrayOf(1, n.toLong())

        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { idsT ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape).use { maskT ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(positions), shape).use { posT ->
                    languageModel.run(
                        mapOf(
                            "input_ids" to idsT,
                            "attention_mask" to maskT,
                            "position_ids" to posT
                        )
                    ).use { r ->
                        @Suppress("UNCHECKED_CAST")
                        val logits = r[0].value as Array<Array<FloatArray>>
                        return logits[0].last()
                    }
                }
            }
        }
    }

    /**
     * Top-k sample restricted to one codebook's band.
     *
     * Everything outside `[offset + cb*2048, offset + (cb+1)*2048)` is
     * excluded, apart from `[stop]`. Without this the model wanders into
     * another codebook's ids and the decoder turns them into noise.
     */
    internal fun sampleInBand(
        logits: FloatArray,
        codebook: Int,
        temperature: Float,
        random: Random
    ): Int {
        val lo = IndriTokenizer.AUDIO_OFFSET + codebook * MimiDecoder.CODEBOOK_SIZE
        val hi = (lo + MimiDecoder.CODEBOOK_SIZE).coerceAtMost(logits.size)
        require(lo < logits.size) { "codebook $codebook is outside the logit range" }

        val k = TOP_K.coerceAtMost(hi - lo)
        val topIdx = IntArray(k)
        val topVal = FloatArray(k) { Float.NEGATIVE_INFINITY }
        var filled = 0

        fun offer(index: Int, value: Float) {
            if (filled < k) {
                var i = filled++
                while (i > 0 && topVal[i - 1] < value) {
                    topVal[i] = topVal[i - 1]; topIdx[i] = topIdx[i - 1]; i--
                }
                topVal[i] = value; topIdx[i] = index
            } else if (value > topVal[k - 1]) {
                var i = k - 1
                while (i > 0 && topVal[i - 1] < value) {
                    topVal[i] = topVal[i - 1]; topIdx[i] = topIdx[i - 1]; i--
                }
                topVal[i] = value; topIdx[i] = index
            }
        }

        for (i in lo until hi) offer(i, logits[i])
        // [stop] competes on its own merit rather than being forced in.
        if (stopToken < logits.size) offer(stopToken, logits[stopToken])

        val t = temperature.coerceAtLeast(1e-3f)
        var sum = 0f
        val probs = FloatArray(filled)
        val max = topVal[0]
        for (i in 0 until filled) {
            probs[i] = exp(((topVal[i] - max) / t).toDouble()).toFloat()
            sum += probs[i]
        }

        var r = random.nextFloat() * sum
        for (i in 0 until filled) {
            r -= probs[i]
            if (r <= 0f) return topIdx[i]
        }
        return topIdx[0]
    }

    override fun close() {
        runCatching { languageModel.close() }
        runCatching { decoder.close() }
    }
}

private const val SEED = 1234
