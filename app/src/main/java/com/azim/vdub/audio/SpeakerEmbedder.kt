package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.azim.vdub.core.VdubPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.sqrt

/**
 * campplus speaker embeddings via ONNX Runtime Mobile.
 *
 * Contract (3D-Speaker CAM++):
 *   input  : float32 (batch, frames, 80)   80-dim log-mel fbank, CMN applied
 *   output : float32 (batch, 192)          speaker embedding
 *
 * Note the model does NOT accept a waveform — the ONNX graph begins after
 * feature extraction, so [Fbank] must produce kaldi-compatible features first.
 */
class SpeakerEmbedder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    val embeddingDim: Int
) : Closeable {

    companion object {
        const val EMBED_DIM = 192

        const val MODEL_NAME = "campplus.onnx"   // ModelCatalog.CAMPPLUS

        /** Looks in both the shared /AI/models and the app-private fallback. */
        fun modelFile(): File =
            VdubPaths.findModel(MODEL_NAME) ?: File(VdubPaths.modelsDir, MODEL_NAME)

        fun isModelPresent(): Boolean =
            modelFile().let { it.exists() && it.length() > 1_000_000 }

        /**
         * @throws IllegalStateException with an actionable message when the
         *         model is missing — this is the most likely user error.
         */
        fun open(threads: Int = 4): SpeakerEmbedder {
            val file = modelFile()
            check(file.exists()) {
                "campplus.onnx not found.\n" +
                    "Open Settings and download it — no PC needed."
            }
            check(file.length() > 1_000_000) {
                "campplus.onnx looks truncated (${file.length()} bytes) — re-copy it."
            }

            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = try {
                env.createSession(file.absolutePath, opts)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Could not load campplus.onnx — is it a real ONNX export? " +
                        "A FunASR .bin/.pt checkpoint will not work; it must be " +
                        "converted to ONNX first. (${e.message})",
                    e
                )
            }

            val inputName = session.inputNames.firstOrNull()
                ?: error("campplus.onnx exposes no inputs")

            return SpeakerEmbedder(env, session, inputName, EMBED_DIM)
        }
    }

    /** Embed one clip. Returns an L2-normalised 192-dim vector. */
    fun embed(wav: File): FloatArray {
        val samples = readWavAsFloat(wav)
        require(samples.isNotEmpty()) { "${wav.name} is empty" }

        val feats = Fbank.compute(samples, applyCmn = true)
        check(feats.isNotEmpty()) {
            "${wav.name} is shorter than one 25 ms frame — cannot embed"
        }

        val frames = feats.size
        val flat = FloatArray(frames * Fbank.NUM_BINS)
        var k = 0
        for (row in feats) {
            System.arraycopy(row, 0, flat, k, Fbank.NUM_BINS)
            k += Fbank.NUM_BINS
        }

        val shape = longArrayOf(1, frames.toLong(), Fbank.NUM_BINS.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val raw = result[0].value as Array<FloatArray>
                return l2Normalise(raw[0])
            }
        }
    }

    /** Embed every clip, reporting progress. Missing clips are skipped. */
    suspend fun embedAll(
        clips: List<File>,
        onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }
    ): LinkedHashMap<String, FloatArray> = withContext(Dispatchers.Default) {
        val out = LinkedHashMap<String, FloatArray>(clips.size)
        clips.forEachIndexed { i, clip ->
            coroutineContext.ensureActive()
            if (clip.exists() && clip.length() > WavIo.HEADER_BYTES) {
                runCatching { embed(clip) }
                    .onSuccess { out[clip.nameWithoutExtension] = it }
            }
            onProgress(i + 1, clips.size)
        }
        out
    }

    override fun close() {
        runCatching { session.close() }
    }

    /** 16-bit PCM wav -> float samples on kaldi's ±32768 scale. */
    private fun readWavAsFloat(wav: File): FloatArray {
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

        val out = FloatArray(frames)
        var p = 0
        for (i in 0 until frames) {
            var acc = 0
            for (c in 0 until fmt.channels) {
                val lo = pcm[p].toInt() and 0xFF
                val hi = pcm[p + 1].toInt()
                acc += (hi shl 8) or lo
                p += 2
            }
            // kaldi operates on the raw int16 scale, not [-1, 1]
            out[i] = (acc / fmt.channels).toFloat()
        }
        return out
    }

    private fun l2Normalise(v: FloatArray): FloatArray {
        var sum = 0.0
        for (x in v) sum += x.toDouble() * x
        val norm = sqrt(sum).toFloat()
        if (norm < 1e-10f) return v
        return FloatArray(v.size) { v[it] / norm }
    }
}
