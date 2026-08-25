package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.ensureActive
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.coroutines.coroutineContext

/**
 * Chatterbox text-to-speech over ONNX Runtime.
 *
 * Four graphs run per line:
 *
 *   speech_encoder      reference wav  -> cond_emb, prompt_token,
 *                                         ref_x_vector, prompt_feat
 *   embed_tokens        token ids      -> embeddings (exaggeration applied here)
 *   language_model      embeddings     -> logits, with a KV cache across steps
 *   conditional_decoder speech tokens  -> 24 kHz waveform
 *
 * The language model is autoregressive: one token per call, feeding its own
 * cache back in. That loop is the whole cost of the stage — a few hundred
 * iterations per line.
 *
 * Sessions are opened once and reused for every line; opening them per line
 * would dominate the runtime.
 */
class ChatterboxTts private constructor(
    private val env: OrtEnvironment,
    private val speechEncoder: OrtSession,
    private val embedTokens: OrtSession,
    private val languageModel: OrtSession,
    private val conditionalDecoder: OrtSession,
    private val tokenizer: ChatterboxTokenizer
) : Closeable {

    companion object {
        const val SAMPLE_RATE = 24_000       // S3Gen output rate
        private const val NUM_HIDDEN_LAYERS = 30
        private const val NUM_KV_HEADS = 16
        private const val HEAD_DIM = 64
        private const val MAX_NEW_TOKENS = 1000

        fun open(paths: VoiceEngine.Paths, threads: Int = 4): ChatterboxTts {
            val missing = paths.missing
            check(missing.isEmpty()) {
                "Voice engine incomplete — missing: " + missing.joinToString { it.name }
            }

            val env = OrtEnvironment.getEnvironment()
            fun opts() = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            // Open in size order so a failure surfaces before the big ones load.
            val encoder = env.createSession(paths.speechEncoder.absolutePath, opts())
            val embed = env.createSession(paths.embedTokens.absolutePath, opts())
            val llm = env.createSession(paths.languageModel.absolutePath, opts())
            val decoder = env.createSession(paths.conditionalDecoder.absolutePath, opts())

            return ChatterboxTts(
                env, encoder, embed, llm, decoder,
                ChatterboxTokenizer.load(paths.tokenizer)
            )
        }
    }

    /** Speaker conditioning, computed once per speaker and reused per line. */
    class SpeakerVoice internal constructor(
        internal val condEmb: Array<Array<FloatArray>>,
        internal val promptToken: Array<LongArray>,
        internal val refXVector: Array<FloatArray>,
        internal val promptFeat: Array<Array<FloatArray>>
    )

    /**
     * Encode a reference clip into speaker conditioning.
     *
     * The clip must be 24 kHz mono float. Our clips are 16 kHz, so they are
     * resampled here rather than at cut time — the analysis models need 16 kHz
     * and re-cutting for TTS would double the disk.
     */
    fun enrol(referenceWav: File): SpeakerVoice {
        val audio = readWav24kMono(referenceWav)
        require(audio.isNotEmpty()) { "${referenceWav.name} is empty" }

        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(audio), longArrayOf(1, audio.size.toLong())
        ).use { tensor ->
            speechEncoder.run(mapOf(inputName(speechEncoder, 0) to tensor)).use { r ->
                @Suppress("UNCHECKED_CAST")
                return SpeakerVoice(
                    condEmb = r[0].value as Array<Array<FloatArray>>,
                    promptToken = toLong2D(r[1].value),
                    refXVector = r[2].value as Array<FloatArray>,
                    promptFeat = r[3].value as Array<Array<FloatArray>>
                )
            }
        }
    }

    /**
     * Speak [text] in [voice].
     *
     * @param exaggeration delivery strength; Step 3's emotion maps to this.
     * @return 24 kHz mono samples in [-1, 1].
     */
    suspend fun speak(
        text: String,
        voice: SpeakerVoice,
        language: String = "hi",
        exaggeration: Float = 0.5f,
        onToken: (Int) -> Unit = {}
    ): FloatArray {
        val prepared = tokenizer.withLanguage(text.trim(), language)
        val inputIds = tokenizer.encode(prepared)
        require(inputIds.isNotEmpty()) { "nothing to speak" }

        // Text positions count up; speech tokens are pinned to 0, matching the
        // reference implementation.
        val positionIds = LongArray(inputIds.size) { i ->
            if (inputIds[i] >= ChatterboxTokenizer.START_SPEECH_TOKEN) 0L else (i - 1).toLong()
        }

        var embeds = runEmbed(
            inputIds.map { it.toLong() }.toLongArray(),
            positionIds,
            exaggeration
        )
        // Prepend speaker conditioning so the LLM is primed with the voice.
        embeds = concatOnTime(voice.condEmb, embeds)

        var seqLen = embeds[0].size
        var attention = LongArray(seqLen) { 1L }
        var past = emptyCache()
        val generated = ArrayList<Long>(256)
        generated.add(ChatterboxTokenizer.START_SPEECH_TOKEN.toLong())

        for (step in 0 until MAX_NEW_TOKENS) {
            coroutineContext.ensureActive()

            val (logits, present) = runLanguageModel(embeds, attention, past)
            past = present

            val next = pickNextToken(logits, generated)
            if (next == ChatterboxTokenizer.STOP_SPEECH_TOKEN.toLong()) break
            generated.add(next)
            onToken(generated.size)

            // Next iteration feeds only the new token.
            embeds = runEmbed(longArrayOf(next), longArrayOf((step + 1).toLong()), exaggeration)
            seqLen += 1
            attention = LongArray(seqLen) { 1L }
        }

        // Drop the leading START and any trailing STOP, then prepend the
        // prompt tokens the encoder produced.
        val speech = generated.drop(1).toLongArray()
        val full = LongArray(voice.promptToken[0].size + speech.size)
        voice.promptToken[0].copyInto(full)
        speech.copyInto(full, voice.promptToken[0].size)

        return runDecoder(full, voice)
    }

    // ------------------------------------------------------------- graphs

    private fun runEmbed(
        ids: LongArray,
        positions: LongArray,
        exaggeration: Float
    ): Array<Array<FloatArray>> {
        val shape = longArrayOf(1, ids.size.toLong())
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { idT ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(positions), shape).use { posT ->
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(floatArrayOf(exaggeration)), longArrayOf(1)
                ).use { exT ->
                    embedTokens.run(
                        mapOf(
                            "input_ids" to idT,
                            "position_ids" to posT,
                            "exaggeration" to exT
                        )
                    ).use { r ->
                        @Suppress("UNCHECKED_CAST")
                        return r[0].value as Array<Array<FloatArray>>
                    }
                }
            }
        }
    }

    private fun runLanguageModel(
        embeds: Array<Array<FloatArray>>,
        attention: LongArray,
        past: Map<String, Array<Array<Array<FloatArray>>>>
    ): Pair<FloatArray, Map<String, Array<Array<Array<FloatArray>>>>> {
        val seq = embeds[0].size
        val hidden = embeds[0][0].size
        val flat = FloatArray(seq * hidden)
        var k = 0
        for (row in embeds[0]) {
            row.copyInto(flat, k)
            k += hidden
        }

        val inputs = HashMap<String, OnnxTensor>()
        val toClose = ArrayList<OnnxTensor>()
        try {
            OnnxTensor.createTensor(
                env, FloatBuffer.wrap(flat), longArrayOf(1, seq.toLong(), hidden.toLong())
            ).also { toClose.add(it); inputs["inputs_embeds"] = it }

            OnnxTensor.createTensor(
                env, LongBuffer.wrap(attention), longArrayOf(1, attention.size.toLong())
            ).also { toClose.add(it); inputs["attention_mask"] = it }

            past.forEach { (name, value) ->
                OnnxTensor.createTensor(env, value)
                    .also { toClose.add(it); inputs[name] = it }
            }

            languageModel.run(inputs).use { r ->
                @Suppress("UNCHECKED_CAST")
                val logits = r[0].value as Array<Array<FloatArray>>
                val last = logits[0].last()

                val present = HashMap<String, Array<Array<Array<FloatArray>>>>()
                val names = languageModel.outputNames.toList()
                for (i in 1 until names.size) {
                    @Suppress("UNCHECKED_CAST")
                    val v = r[i].value as Array<Array<Array<FloatArray>>>
                    present[names[i].replace("present", "past_key_values")] = v
                }
                return last to present
            }
        } finally {
            toClose.forEach { runCatching { it.close() } }
        }
    }

    private fun runDecoder(speechTokens: LongArray, voice: SpeakerVoice): FloatArray {
        OnnxTensor.createTensor(
            env, LongBuffer.wrap(speechTokens), longArrayOf(1, speechTokens.size.toLong())
        ).use { tokens ->
            OnnxTensor.createTensor(env, voice.refXVector).use { spk ->
                OnnxTensor.createTensor(env, voice.promptFeat).use { feat ->
                    conditionalDecoder.run(
                        mapOf(
                            "speech_tokens" to tokens,
                            "speaker_embeddings" to spk,
                            "speaker_features" to feat
                        )
                    ).use { r ->
                        return flattenWav(r[0].value)
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ helpers

    /**
     * Greedy pick with a repetition penalty.
     *
     * The penalty is not optional: at the upstream default of 2.0 this
     * quantized build never emits STOP and runs to the token cap on every
     * line. 1.2 is what the model card specifies.
     */
    private fun pickNextToken(logits: FloatArray, generated: List<Long>): Long {
        val penalty = VoiceEngine.REPETITION_PENALTY
        val seen = generated.toHashSet()
        var bestIdx = 0
        var bestVal = Float.NEGATIVE_INFINITY
        for (i in logits.indices) {
            var v = logits[i]
            if (seen.contains(i.toLong())) {
                v = if (v < 0) v * penalty else v / penalty
            }
            if (v > bestVal) {
                bestVal = v
                bestIdx = i
            }
        }
        return bestIdx.toLong()
    }

    private fun emptyCache(): Map<String, Array<Array<Array<FloatArray>>>> =
        buildMap {
            for (layer in 0 until NUM_HIDDEN_LAYERS) {
                for (kv in listOf("key", "value")) {
                    put(
                        "past_key_values.$layer.$kv",
                        Array(1) { Array(NUM_KV_HEADS) { Array(0) { FloatArray(HEAD_DIM) } } }
                    )
                }
            }
        }

    private fun concatOnTime(
        a: Array<Array<FloatArray>>,
        b: Array<Array<FloatArray>>
    ): Array<Array<FloatArray>> = arrayOf(a[0] + b[0])

    private fun inputName(session: OrtSession, index: Int): String =
        session.inputNames.toList()[index]

    private fun toLong2D(value: Any?): Array<LongArray> {
        @Suppress("UNCHECKED_CAST")
        return when (value) {
            is Array<*> -> when (value.firstOrNull()) {
                is LongArray -> value as Array<LongArray>
                is IntArray -> (value as Array<IntArray>)
                    .map { row -> LongArray(row.size) { row[it].toLong() } }.toTypedArray()
                else -> error("unexpected prompt_token type")
            }
            else -> error("unexpected prompt_token ${value?.javaClass}")
        }
    }

    private fun flattenWav(value: Any?): FloatArray {
        @Suppress("UNCHECKED_CAST")
        return when (value) {
            is Array<*> -> when (val first = value.firstOrNull()) {
                is FloatArray -> {
                    val rows = value as Array<FloatArray>
                    if (rows.size == 1) rows[0]
                    else FloatArray(rows.sumOf { it.size }).also { out ->
                        var k = 0
                        rows.forEach { it.copyInto(out, k); k += it.size }
                    }
                }
                is Array<*> -> flattenWav(first)
                else -> error("unexpected decoder output element ${first?.javaClass}")
            }
            is FloatArray -> value
            else -> error("unexpected decoder output ${value?.javaClass}")
        }
    }

    /** 16-bit PCM wav -> 24 kHz mono float, resampling if needed. */
    private fun readWav24kMono(wav: File): FloatArray {
        val fmt = WavIo.readFormat(wav)
        require(fmt.bitsPerSample == 16) { "${wav.name}: expected 16-bit PCM" }
        val bytesPerFrame = fmt.channels * 2
        val frames = (fmt.dataBytes / bytesPerFrame).toInt()
        if (frames <= 0) return FloatArray(0)

        val pcm = ByteArray(fmt.dataBytes.toInt())
        java.io.RandomAccessFile(wav, "r").use { raf ->
            raf.seek(fmt.dataOffset)
            raf.readFully(pcm)
        }

        val mono = FloatArray(frames)
        var p = 0
        for (i in 0 until frames) {
            var acc = 0
            for (c in 0 until fmt.channels) {
                val lo = pcm[p].toInt() and 0xFF
                val hi = pcm[p + 1].toInt()
                acc += (hi shl 8) or lo
                p += 2
            }
            mono[i] = (acc / fmt.channels) / 32768f
        }

        if (fmt.sampleRate == SAMPLE_RATE) return mono
        return resampleLinear(mono, fmt.sampleRate, SAMPLE_RATE)
    }

    override fun close() {
        listOf(speechEncoder, embedTokens, languageModel, conditionalDecoder)
            .forEach { runCatching { it.close() } }
    }
}

/** Linear resample. Shared with the mux stage. */
internal fun resampleLinear(input: FloatArray, inRate: Int, outRate: Int): FloatArray {
    if (input.isEmpty() || inRate == outRate) return input
    val ratio = outRate.toDouble() / inRate
    val outLen = (input.size * ratio).toInt().coerceAtLeast(1)
    val out = FloatArray(outLen)
    for (i in 0 until outLen) {
        val src = i / ratio
        val i0 = src.toInt().coerceIn(0, input.size - 1)
        val i1 = (i0 + 1).coerceAtMost(input.size - 1)
        val frac = (src - i0).toFloat()
        out[i] = input[i0] + (input[i1] - input[i0]) * frac
    }
    return out
}
