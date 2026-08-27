package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.ensureActive
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * DhVaani 0.5 — ZipVoice flow matching, zero-shot cloning for Indic languages.
 *
 * Structurally the opposite of Chatterbox. There is no autoregressive loop:
 * the text encoder plans the whole utterance in one pass, a fixed number of
 * Euler steps denoise it, and the vocoder turns mel into audio. That is why it
 * runs faster than real time on a phone CPU while Chatterbox takes about a
 * minute a line.
 *
 *   text_encoder   tokens, prompt tokens, prompt length, speed -> (1, T, 100)
 *   fm_decoder     t, x, text_condition, speech_condition, cfg -> velocity
 *   vocoder        (1, 100, T) mel -> (1, T, 512) hidden -> ISTFT -> waveform
 *
 * Cloning conditions on both the reference audio *and* its transcript, so
 * [enrol] takes one. Without it the model still speaks, but in a voice only
 * loosely related to the reference — the prompt text is what aligns the
 * reference features to phonetic content.
 *
 * Three details that are easy to get wrong and do not fail loudly:
 *
 *  - features are scaled by 0.1 going in and unscaled coming out. Skip it and
 *    the flow starts far outside its trained range, producing noise.
 *  - the reference is RMS-normalised to 0.1 before analysis, and the output is
 *    scaled back by the same ratio afterwards, so a quiet reference does not
 *    yield a clipped result.
 *  - the mel filterbank and ISTFT window are read from the shipped npz files
 *    rather than recomputed. See [NpzReader].
 */
class DhVaaniTts private constructor(
    private val env: OrtEnvironment,
    private val textEncoder: OrtSession,
    private val fmDecoder: OrtSession,
    private val vocoder: OrtSession,
    private val tokens: Map<String, Int>,
    private val melFb: Array<FloatArray>,      // (n_fft/2+1, 100)
    private val melWindow: FloatArray,         // (n_fft)
    private val head: VocosHead
) : TtsEngine {

    override val sampleRate = SAMPLE_RATE
    override val clonesVoice = true

    companion object {
        const val SAMPLE_RATE = 24_000
        const val N_FFT = 1024
        const val HOP = 256
        const val N_MELS = 100

        /** The model was trained on features at a tenth of their natural scale. */
        const val FEAT_SCALE = 0.1f

        /** References are levelled to this RMS before analysis. */
        const val TARGET_RMS = 0.1f

        /**
         * Euler steps. Quality rises with more and so does time, roughly
         * linearly: 4 steps is RTF 0.84 on two cores, 16 is 3.26. Eight is the
         * knee — clearly better than four, half the cost of sixteen.
         */
        const val DEFAULT_STEPS = 8

        /**
         * Warps the step schedule towards t=0, where the trajectory bends
         * most. ZipVoice's default.
         */
        const val T_SHIFT = 0.5f

        private val PUNCTUATION = ";:,.!?；：，。！？।".toSet()

        fun open(paths: VoiceEngine.Paths, threads: Int = 4): DhVaaniTts {
            val missing = paths.missing
            check(missing.isEmpty()) {
                "DhVaani is incomplete — missing: " + missing.joinToString { it.name }
            }

            val env = OrtEnvironment.getEnvironment()
            fun opts() = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            val te = openGraph(env, paths.textEncoder(), opts())
            val fm = openGraph(env, paths.fmDecoder(), opts())
            val vo = openGraph(env, paths.vocoderBackbone(), opts())

            val melz = NpzReader.read(paths.melFilterbank())
            val fb = (melz["fb"] ?: error("mel_fb.npz has no 'fb'")).as2D()
            val win = (melz["window"] ?: error("mel_fb.npz has no 'window'")).data
            require(fb.size == N_FFT / 2 + 1 && fb[0].size == N_MELS) {
                "mel filterbank is ${fb.size}x${fb[0].size}, expected " +
                    "${N_FFT / 2 + 1}x$N_MELS"
            }
            require(win.size == N_FFT) { "mel window is ${win.size}, expected $N_FFT" }

            return DhVaaniTts(
                env, te, fm, vo,
                loadTokens(paths.tokensTxt()),
                fb, win,
                VocosHead.from(NpzReader.read(paths.vocosHead()))
            )
        }

        /** `char \t id`, one per line — 1058 entries covering 13 scripts. */
        private fun loadTokens(file: File): Map<String, Int> {
            val map = HashMap<String, Int>(1200)
            file.forEachLine { line ->
                if (line.isEmpty()) return@forEachLine
                val tab = line.indexOf('\t')
                if (tab <= 0) return@forEachLine
                val id = line.substring(tab + 1).trim().toIntOrNull() ?: return@forEachLine
                map[line.substring(0, tab)] = id
            }
            check(map.size > 500) { "tokens.txt looks truncated (${map.size} entries)" }
            return map
        }

        private fun openGraph(
            env: OrtEnvironment,
            file: File,
            opts: OrtSession.SessionOptions
        ): OrtSession = try {
            env.createSession(file.absolutePath, opts)
        } catch (e: Exception) {
            val raw = e.message.orEmpty()
            val hint = when {
                raw.contains("No such file", true) ->
                    "${file.name} is missing. Re-download DhVaani."
                raw.contains("Protobuf parsing failed", true) ||
                    raw.contains("invalid model", true) ->
                    "${file.name} looks corrupt or half-downloaded. " +
                        "Remove and re-download DhVaani."
                else -> "Could not load ${file.name}."
            }
            throw IllegalStateException("$hint\n\n(${raw.take(300)})", e)
        }
    }

    /** The Vocos output layer: mel hidden state -> complex spectrum -> ISTFT. */
    internal class VocosHead(
        val weight: Array<FloatArray>,   // (1026, 512)
        val bias: FloatArray,            // (1026)
        val window: FloatArray,          // (1024)
        val nFft: Int,
        val hop: Int,
        val winLength: Int
    ) {
        companion object {
            fun from(npz: Map<String, NpzReader.NpArray>): VocosHead {
                fun need(k: String) = npz[k] ?: error("vocos_head.npz has no '$k'")
                val w = need("linear_weight").as2D()
                val b = need("linear_bias").data
                val win = need("window").data
                val nFft = need("n_fft").scalar.toInt()
                val hop = need("hop_length").scalar.toInt()
                val winLen = need("win_length").scalar.toInt()
                require(w.size == b.size) {
                    "vocos head weight ${w.size} vs bias ${b.size}"
                }
                require(win.size == winLen) {
                    "vocos window ${win.size} vs win_length $winLen"
                }
                return VocosHead(w, b, win, nFft, hop, winLen)
            }
        }
    }

    /** Reference features plus the transcript, both needed by the encoder. */
    private class DhVoice(
        val promptFeatures: FloatArray,   // flattened (1, frames, 100), pre-scaled
        val promptFrames: Int,
        val promptTokens: LongArray,
        val rmsScale: Float
    ) : TtsEngine.Voice

    override fun enrol(referenceWav: File?, transcript: String): TtsEngine.Voice {
        requireNotNull(referenceWav) { "DhVaani clones a voice — it needs a reference clip" }
        val audio = readWavMono(referenceWav, SAMPLE_RATE)
        require(audio.size >= SAMPLE_RATE / 2) {
            "${referenceWav.name} is only %.1f s — need at least 0.5 s to clone a voice"
                .format(audio.size.toDouble() / SAMPLE_RATE)
        }

        // Level the reference, remembering by how much so the output can be
        // put back at the speaker's own loudness.
        var rms = 0.0
        for (s in audio) rms += (s * s).toDouble()
        rms = sqrt(rms / audio.size)
        val scale = if (rms in 1e-6..TARGET_RMS.toDouble()) (TARGET_RMS / rms).toFloat() else 1f
        val levelled = if (scale == 1f) audio else FloatArray(audio.size) { audio[it] * scale }

        val feats = vocosFbank(levelled)
        val frames = feats.size
        check(frames > 0) { "${referenceWav.name} produced no features" }

        val flat = FloatArray(frames * N_MELS)
        for (i in 0 until frames) {
            for (j in 0 until N_MELS) flat[i * N_MELS + j] = feats[i][j] * FEAT_SCALE
        }

        // A transcript that is missing is better than one that is wrong: the
        // encoder aligns reference features to these tokens, so unrelated text
        // teaches it the wrong mapping. An empty prompt just weakens cloning.
        val promptText = addPunctuation(transcript.trim())
        return DhVoice(
            promptFeatures = flat,
            promptFrames = frames,
            promptTokens = encode(promptText),
            rmsScale = if (rms < TARGET_RMS && rms > 1e-6) (rms / TARGET_RMS).toFloat() else 1f
        )
    }

    override suspend fun speak(
        text: String,
        voice: TtsEngine.Voice,
        language: String,
        exaggeration: Float,
        onToken: (Int) -> Unit
    ): FloatArray {
        val v = voice as? DhVoice
            ?: error("enrol() this engine's own voice — got ${voice.javaClass.simpleName}")

        val prepared = addPunctuation(text.trim())
        val ids = encode(prepared)
        require(ids.isNotEmpty()) {
            "nothing speakable in \"${text.take(40)}\" — no character is in DhVaani's " +
                "vocabulary"
        }

        // Emotion maps to speaking rate here. The model has no exaggeration
        // input, and faking one by scaling the mel would change pitch as well,
        // so a small tempo shift is the honest interpretation: agitated
        // delivery is faster, subdued is slower.
        val speed = (1f + (exaggeration - 1f) * 0.35f).coerceIn(0.7f, 1.4f)

        val condition = runTextEncoder(ids, v, speed)
        val frames = condition.size / N_MELS
        check(frames > 0) { "text encoder produced no frames" }

        val x = flowSample(condition, v, frames, onToken)

        // Drop the prompt frames — the model regenerates them and they are the
        // reference speaking, not the new line.
        val keep = frames - v.promptFrames
        check(keep > 0) {
            "reference (${v.promptFrames} frames) is longer than the generated " +
                "utterance ($frames) — use a shorter reference clip"
        }

        val mel = FloatArray(N_MELS * keep)
        for (t in 0 until keep) {
            val src = (v.promptFrames + t) * N_MELS
            for (m in 0 until N_MELS) mel[m * keep + t] = x[src + m] / FEAT_SCALE
        }

        val audio = vocode(mel, keep)
        if (v.rmsScale != 1f) for (i in audio.indices) audio[i] *= v.rmsScale
        for (i in audio.indices) audio[i] = audio[i].coerceIn(-1f, 1f)
        return audio
    }

    // ------------------------------------------------------------- graphs

    private fun runTextEncoder(ids: LongArray, v: DhVoice, speed: Float): FloatArray {
        OnnxTensor.createTensor(
            env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())
        ).use { tokensT ->
            OnnxTensor.createTensor(
                env, LongBuffer.wrap(v.promptTokens),
                longArrayOf(1, v.promptTokens.size.toLong())
            ).use { promptT ->
                OnnxTensor.createTensor(
                    env, LongBuffer.wrap(longArrayOf(v.promptFrames.toLong())),
                    longArrayOf()
                ).use { lenT ->
                    OnnxTensor.createTensor(
                        env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf()
                    ).use { speedT ->
                        textEncoder.run(
                            mapOf(
                                "tokens" to tokensT,
                                "prompt_tokens" to promptT,
                                "prompt_features_len" to lenT,
                                "speed" to speedT
                            )
                        ).use { r -> return flatten3(r[0].value) }
                    }
                }
            }
        }
    }

    /**
     * Integrate the flow from noise to speech.
     *
     * x starts as Gaussian noise and is pushed along the velocity field the
     * decoder predicts, in [DEFAULT_STEPS] Euler steps. `speech_condition`
     * holds the reference features in the leading frames and zeros after, so
     * the model continues the reference speaker into new content.
     */
    private suspend fun flowSample(
        condition: FloatArray,
        v: DhVoice,
        frames: Int,
        onToken: (Int) -> Unit
    ): FloatArray {
        val n = frames * N_MELS
        val rng = java.util.Random(SEED)
        val x = FloatArray(n) { rng.nextGaussian().toFloat() }

        val speech = FloatArray(n)
        val copy = min(v.promptFrames, frames) * N_MELS
        v.promptFeatures.copyInto(speech, 0, 0, min(copy, v.promptFeatures.size))

        val shape = longArrayOf(1, frames.toLong(), N_MELS.toLong())
        val schedule = timeSteps(DEFAULT_STEPS, T_SHIFT)

        OnnxTensor.createTensor(env, FloatBuffer.wrap(condition), shape).use { condT ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(speech), shape).use { speechT ->
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(floatArrayOf(GUIDANCE_SCALE)), longArrayOf()
                ).use { cfgT ->
                    for (step in 0 until DEFAULT_STEPS) {
                        coroutineContext.ensureActive()
                        val dt = schedule[step + 1] - schedule[step]

                        OnnxTensor.createTensor(
                            env, FloatBuffer.wrap(floatArrayOf(schedule[step])), longArrayOf()
                        ).use { tT ->
                            OnnxTensor.createTensor(
                                env, FloatBuffer.wrap(x), shape
                            ).use { xT ->
                                fmDecoder.run(
                                    mapOf(
                                        "t" to tT,
                                        "x" to xT,
                                        "text_condition" to condT,
                                        "speech_condition" to speechT,
                                        "guidance_scale" to cfgT
                                    )
                                ).use { r ->
                                    val vel = flatten3(r[0].value)
                                    for (i in x.indices) x[i] += vel[i] * dt
                                }
                            }
                        }
                        onToken(step + 1)
                    }
                }
            }
        }
        return x
    }

    /** Mel (100, T) -> waveform, through the ConvNeXt backbone and ISTFT head. */
    private fun vocode(mel: FloatArray, frames: Int): FloatArray {
        val hidden: Array<FloatArray>
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel), longArrayOf(1, N_MELS.toLong(), frames.toLong())
        ).use { melT ->
            vocoder.run(mapOf("mels" to melT)).use { r ->
                @Suppress("UNCHECKED_CAST")
                hidden = (r[0].value as Array<Array<FloatArray>>)[0]
            }
        }

        // hidden (T, 512) -> (T, 1026): magnitude and phase, interleaved as
        // two halves.
        val bins = head.nFft / 2 + 1
        val t = hidden.size
        val magnitude = Array(t) { FloatArray(bins) }
        val phase = Array(t) { FloatArray(bins) }

        for (frame in 0 until t) {
            val h = hidden[frame]
            for (k in 0 until bins) {
                var accMag = head.bias[k]
                var accPhase = head.bias[k + bins]
                val wm = head.weight[k]
                val wp = head.weight[k + bins]
                for (c in h.indices) {
                    accMag += wm[c] * h[c]
                    accPhase += wp[c] * h[c]
                }
                // Clamped before exp: the model occasionally predicts a large
                // magnitude on the first frame and it becomes an infinity that
                // poisons the whole overlap-add.
                magnitude[frame][k] = exp(min(accMag, MAX_LOG_MAGNITUDE))
                phase[frame][k] = accPhase
            }
        }
        return istft(magnitude, phase)
    }

    /**
     * Inverse STFT with `padding="same"`, matching Vocos.
     *
     * Overlap-add divided by the summed squared window, so the analysis window
     * cancels exactly; without that division the output has a periodic tremor
     * at the hop rate.
     */
    private fun istft(magnitude: Array<FloatArray>, phase: Array<FloatArray>): FloatArray {
        val frames = magnitude.size
        if (frames == 0) return FloatArray(0)

        val n = head.nFft
        val hop = head.hop
        val winLen = head.winLength
        val pad = (winLen - hop) / 2
        val outSize = (frames - 1) * hop + winLen

        val acc = FloatArray(outSize)
        val envelope = FloatArray(outSize)
        val re = FloatArray(n)
        val im = FloatArray(n)

        for (f in 0 until frames) {
            java.util.Arrays.fill(re, 0f)
            java.util.Arrays.fill(im, 0f)
            val mag = magnitude[f]
            val ph = phase[f]
            // Rebuild the full hermitian spectrum from the half we predicted.
            for (k in mag.indices) {
                val r = mag[k] * cos(ph[k])
                val i = mag[k] * sin(ph[k])
                re[k] = r
                im[k] = i
                if (k in 1 until n / 2) {
                    re[n - k] = r
                    im[n - k] = -i
                }
            }
            im[0] = 0f
            if (n % 2 == 0) im[n / 2] = 0f

            inverseFft(re, im)

            val base = f * hop
            for (i in 0 until winLen) {
                val w = head.window[i]
                acc[base + i] += re[i] * w
                envelope[base + i] += w * w
            }
        }

        val outLen = outSize - 2 * pad
        if (outLen <= 0) return FloatArray(0)
        return FloatArray(outLen) { i ->
            val e = envelope[i + pad]
            if (e > 1e-11f) acc[i + pad] / e else 0f
        }
    }

    // ------------------------------------------------------------ features

    /**
     * Log-mel features matching ZipVoice's VocosFbank.
     *
     * Centre-padded, magnitude (not power) spectrum, the shipped filterbank,
     * then a natural log with a 1e-7 floor. Frame count is derived from the
     * duration the way the reference does, so the encoder's `prompt_features_len`
     * agrees with what it is given.
     */
    private fun vocosFbank(wav: FloatArray): Array<FloatArray> {
        val pad = N_FFT / 2
        val padded = FloatArray(wav.size + 2 * pad)
        wav.copyInto(padded, pad)

        val usable = if (padded.size < N_FFT) FloatArray(N_FFT) else padded
        if (padded.size < N_FFT) padded.copyInto(usable, 0, 0, padded.size)

        val frames = 1 + (usable.size - N_FFT) / HOP
        if (frames <= 0) return emptyArray()

        val re = FloatArray(N_FFT)
        val im = FloatArray(N_FFT)
        val bins = N_FFT / 2 + 1
        val out = ArrayList<FloatArray>(frames)

        for (f in 0 until frames) {
            val start = f * HOP
            for (i in 0 until N_FFT) re[i] = usable[start + i] * melWindow[i]
            java.util.Arrays.fill(im, 0f)
            forwardFft(re, im)

            val row = FloatArray(N_MELS)
            for (k in 0 until bins) {
                val magnitude = sqrt(re[k] * re[k] + im[k] * im[k])
                val weights = melFb[k]
                for (m in 0 until N_MELS) {
                    val w = weights[m]
                    if (w != 0f) row[m] += w * magnitude
                }
            }
            for (m in 0 until N_MELS) {
                row[m] = ln(maxOf(row[m], 1e-7f).toDouble()).toFloat()
            }
            out.add(row)
        }

        // Match the reference's frame count exactly, padding by repetition.
        val want = (wav.size + HOP / 2) / HOP
        return when {
            out.size > want -> Array(want) { out[it] }
            out.size < want && out.isNotEmpty() ->
                Array(want) { out[min(it, out.size - 1)] }
            else -> out.toTypedArray()
        }
    }

    private fun encode(text: String): LongArray {
        val ids = ArrayList<Long>(text.length)
        var i = 0
        while (i < text.length) {
            // Surrogate pairs must be looked up whole, or an emoji becomes two
            // unknown halves.
            val cp = text.codePointAt(i)
            val s = String(Character.toChars(cp))
            tokens[s]?.let { ids.add(it.toLong()) }
            i += Character.charCount(cp)
        }
        return ids.toLongArray()
    }

    /** The model expects a terminated sentence; an unterminated one runs on. */
    private fun addPunctuation(text: String): String = when {
        text.isEmpty() -> "."
        text.last() in PUNCTUATION -> text
        else -> "$text."
    }

    private fun timeSteps(steps: Int, shift: Float): FloatArray =
        FloatArray(steps + 1) { i ->
            val t = i.toFloat() / steps
            shift * t / (1f + (shift - 1f) * t)
        }

    @Suppress("UNCHECKED_CAST")
    private fun flatten3(value: Any?): FloatArray {
        val batch = value as? Array<*> ?: error("expected a 3-D tensor")
        val rows = batch[0] as Array<FloatArray>
        if (rows.isEmpty()) return FloatArray(0)
        val cols = rows[0].size
        val out = FloatArray(rows.size * cols)
        var k = 0
        for (row in rows) {
            row.copyInto(out, k)
            k += cols
        }
        return out
    }

    private fun readWavMono(wav: File, targetRate: Int): FloatArray {
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
        return if (fmt.sampleRate == targetRate) mono
        else resampleLinear(mono, fmt.sampleRate, targetRate)
    }

    override fun close() {
        listOf(textEncoder, fmDecoder, vocoder).forEach { runCatching { it.close() } }
    }
}

private const val SEED = 666L
private const val GUIDANCE_SCALE = 1.0f
private const val MAX_LOG_MAGNITUDE = 4.6051702f    // ln(100), Vocos' clamp

/** In-place radix-2 forward FFT. Length must be a power of two. */
internal fun forwardFft(re: FloatArray, im: FloatArray) = fftInPlace(re, im, false)

/** In-place radix-2 inverse FFT, scaled by 1/n. */
internal fun inverseFft(re: FloatArray, im: FloatArray) = fftInPlace(re, im, true)

private fun fftInPlace(re: FloatArray, im: FloatArray, inverse: Boolean) {
    val n = re.size
    require(n and (n - 1) == 0) { "FFT length $n is not a power of two" }

    var j = 0
    for (i in 1 until n) {
        var bit = n shr 1
        while (j and bit != 0) {
            j = j xor bit
            bit = bit shr 1
        }
        j = j or bit
        if (i < j) {
            var t = re[i]; re[i] = re[j]; re[j] = t
            t = im[i]; im[i] = im[j]; im[j] = t
        }
    }

    var len = 2
    while (len <= n) {
        val ang = (if (inverse) 2.0 else -2.0) * Math.PI / len
        val wRe = cos(ang).toFloat()
        val wIm = sin(ang).toFloat()
        var i = 0
        while (i < n) {
            var curRe = 1f
            var curIm = 0f
            val half = len / 2
            for (k in 0 until half) {
                val uRe = re[i + k]
                val uIm = im[i + k]
                val vRe = re[i + k + half] * curRe - im[i + k + half] * curIm
                val vIm = re[i + k + half] * curIm + im[i + k + half] * curRe
                re[i + k] = uRe + vRe
                im[i + k] = uIm + vIm
                re[i + k + half] = uRe - vRe
                im[i + k + half] = uIm - vIm
                val nextRe = curRe * wRe - curIm * wIm
                curIm = curRe * wIm + curIm * wRe
                curRe = nextRe
            }
            i += len
        }
        len = len shl 1
    }

    if (inverse) {
        val scale = 1f / n
        for (i in 0 until n) {
            re[i] *= scale
            im[i] *= scale
        }
    }
}
