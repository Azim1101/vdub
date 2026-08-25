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
            val encoder = openGraph(env, paths.speechEncoder, opts())
            val embed = openGraph(env, paths.embedTokens, opts())
            val llm = openGraph(env, paths.languageModel, opts())
            val decoder = openGraph(env, paths.conditionalDecoder, opts())

            return ChatterboxTts(
                env, encoder, embed, llm, decoder,
                ChatterboxTokenizer.load(paths.tokenizer)
            )
        }

        /**
         * Open one graph, translating ONNX Runtime's failures into something
         * actionable. Its raw message is a full node signature dump, which
         * tells a user nothing about what to do next.
         */
        private fun openGraph(
            env: OrtEnvironment,
            file: File,
            opts: OrtSession.SessionOptions
        ): OrtSession = try {
            env.createSession(file.absolutePath, opts)
        } catch (e: Exception) {
            val raw = e.message.orEmpty()
            val hint = when {
                // Contrib ops gain inputs across ORT releases; a model exported
                // against a newer runtime cannot load on an older one.
                raw.contains("not in range", ignoreCase = true) ||
                    raw.contains("GroupQueryAttention") ->
                    "This model needs a newer ONNX Runtime than the app has. " +
                        "Update the app — the packaged runtime was raised for " +
                        "exactly this."
                raw.contains("No such file", ignoreCase = true) ->
                    "${file.name} is missing. Re-download the voice engine."
                raw.contains("Protobuf parsing failed", ignoreCase = true) ||
                    raw.contains("invalid model", ignoreCase = true) ->
                    "${file.name} looks corrupt or half-downloaded. " +
                        "Remove and re-download the voice engine."
                else -> "Could not load ${file.name}."
            }
            throw IllegalStateException("$hint\n\n(${raw.take(300)})", e)
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
        // Below about a second the encoder emits an empty prompt, which then
        // fails deep inside the decoder as a zero-dimension error rather than
        // as "this clip is too short".
        require(audio.size >= SAMPLE_RATE) {
            "${referenceWav.name} is only %.1f s — need at least 1 s of speech to clone a voice"
                .format(audio.size.toDouble() / SAMPLE_RATE)
        }

        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(audio), longArrayOf(1, audio.size.toLong())
        ).use { tensor ->
            speechEncoder.run(mapOf(inputName(speechEncoder, 0) to tensor)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val voice = SpeakerVoice(
                    condEmb = r[0].value as Array<Array<FloatArray>>,
                    promptToken = toLong2D(r[1].value),
                    refXVector = r[2].value as Array<FloatArray>,
                    promptFeat = r[3].value as Array<Array<FloatArray>>
                )
                check(voice.condEmb[0].isNotEmpty() && voice.promptFeat[0].isNotEmpty()) {
                    "${referenceWav.name} produced no speaker conditioning — " +
                        "the clip is probably silence"
                }
                return voice
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

        try {
            for (step in 0 until MAX_NEW_TOKENS) {
                coroutineContext.ensureActive()

                val (logits, present) = runLanguageModel(embeds, attention, past)
                // Release the previous step's cache as soon as the next one
                // exists; holding both is what blew the heap.
                past.close()
                past = present

                val next = pickNextToken(logits, generated)
                if (next == ChatterboxTokenizer.STOP_SPEECH_TOKEN.toLong()) break
                generated.add(next)
                onToken(generated.size)

                // Next iteration feeds only the new token.
                embeds = runEmbed(
                    longArrayOf(next), longArrayOf((step + 1).toLong()), exaggeration
                )
                seqLen += 1
                attention = LongArray(seqLen) { 1L }
            }
        } finally {
            past.close()
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

    /**
     * The KV cache, kept as live ONNX tensors.
     *
     * It must not be copied into Java arrays. By 700 tokens each of the 60
     * entries is ~2.8 MB, so one copy is ~165 MB of heap and holding old and
     * new at once doubles it — past Android's 512 MB cap, which is exactly the
     * OOM this hit.
     *
     * Tensors returned by a run are backed by native memory owned by that
     * result, so the result is kept alive alongside them and closed only once
     * the next step has replaced it.
     */
    private class KvCache(
        val tensors: Map<String, OnnxTensor>,
        private val owner: AutoCloseable?
    ) : AutoCloseable {
        override fun close() {
            // Closing the owning result frees the tensors with it; standalone
            // tensors (the initial empty cache) own themselves.
            val held = owner
            if (held != null) runCatching { held.close() }
            else tensors.values.forEach { runCatching { it.close() } }
        }
    }

    private fun runLanguageModel(
        embeds: Array<Array<FloatArray>>,
        attention: LongArray,
        past: KvCache
    ): Pair<FloatArray, KvCache> {
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

            // Feed the previous step's tensors straight back in — no copy.
            past.tensors.forEach { (name, t) -> inputs[name] = t }

            val result = languageModel.run(inputs)
            var keep = false
            try {
                @Suppress("UNCHECKED_CAST")
                val logits = result[0].value as Array<Array<FloatArray>>
                val last = logits[0].last()

                val present = HashMap<String, OnnxTensor>()
                val names = languageModel.outputNames.toList()
                for (i in 1 until names.size) {
                    present[names[i].replace("present", "past_key_values")] =
                        result[i] as OnnxTensor
                }
                keep = true
                // The result owns this memory, so it stays open until the
                // cache it backs is replaced.
                return last to KvCache(present, result)
            } finally {
                if (!keep) runCatching { result.close() }
            }
        } finally {
            // Only the tensors created here; the cache belongs to its owner.
            toClose.forEach { runCatching { it.close() } }
        }
    }

    private fun runDecoder(speechTokens: LongArray, voice: SpeakerVoice): FloatArray {
        require(speechTokens.isNotEmpty()) { "no speech tokens to decode" }

        // Flatten rather than passing nested arrays: ORT infers shapes from
        // those and rejects any zero-length axis, which a short prompt can
        // produce.
        val xv = voice.refXVector[0]
        val featRows = voice.promptFeat[0]
        val featDim = featRows.firstOrNull()?.size ?: 0
        val featFlat = FloatArray(featRows.size * featDim)
        var fk = 0
        for (row in featRows) {
            row.copyInto(featFlat, fk)
            fk += featDim
        }

        OnnxTensor.createTensor(
            env, LongBuffer.wrap(speechTokens), longArrayOf(1, speechTokens.size.toLong())
        ).use { tokens ->
            OnnxTensor.createTensor(
                env, FloatBuffer.wrap(xv), longArrayOf(1, xv.size.toLong())
            ).use { spk ->
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(featFlat),
                    longArrayOf(1, featRows.size.toLong(), featDim.toLong())
                ).use { feat ->
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

    /** Zero-length cache for the first step: [1, heads, 0, head_dim]. */
    private fun emptyCache(): KvCache {
        val shape = longArrayOf(1, NUM_KV_HEADS.toLong(), 0L, HEAD_DIM.toLong())
        val empty = FloatArray(0)
        val map = HashMap<String, OnnxTensor>(NUM_HIDDEN_LAYERS * 2)
        for (layer in 0 until NUM_HIDDEN_LAYERS) {
            for (kv in listOf("key", "value")) {
                map["past_key_values.$layer.$kv"] =
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(empty), shape.copyOf())
            }
        }
        return KvCache(map, owner = null)
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
