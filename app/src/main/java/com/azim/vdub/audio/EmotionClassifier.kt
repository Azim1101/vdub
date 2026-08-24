package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.azim.vdub.core.VdubPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.exp

/**
 * emotion2vec+ base — per-clip emotion.
 *
 * Contract (from the model card):
 *   input  : float32 (1, samples)   raw 16 kHz mono waveform in [-1, 1]
 *   output : float32 (1, T, D)      frame features
 *   then   : pooled = mean over T ; probs = softmax(W @ pooled + B)
 *
 * The classifier head is NOT in the graph — it ships as emotion2vec_head.json
 * with `weight`, `bias` and `labels`. Skipping it would leave raw features
 * that look fine but classify nothing.
 *
 * Note the scale difference from [SpeakerEmbedder]: campplus consumes kaldi
 * fbank computed on the ±32768 integer scale, while emotion2vec takes the
 * normalised waveform. Feeding either one the other's scale produces
 * confident nonsense rather than an error.
 */
class EmotionClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    private val weight: Array<FloatArray>,   // [labels][dim]
    private val bias: FloatArray,            // [labels]
    val labels: List<String>
) : Closeable {

    data class Result(
        val label: String,
        val confidence: Float,
        /** All labels with probabilities, best first. */
        val ranked: List<Pair<String, Float>>
    )

    companion object {
        const val MODEL_NAME = "emotion2vec.onnx"
        const val HEAD_NAME = "emotion2vec_head.json"

        /** Minimum clip length: shorter than this the model has nothing to chew. */
        private const val MIN_SAMPLES = 16_000 / 10   // 100 ms

        fun modelFile(): File =
            VdubPaths.findModel(MODEL_NAME) ?: File(VdubPaths.modelsDir, MODEL_NAME)

        fun headFile(): File =
            VdubPaths.findModel(HEAD_NAME) ?: File(VdubPaths.modelsDir, HEAD_NAME)

        fun isModelPresent(): Boolean =
            modelFile().let { it.exists() && it.length() > 1_000_000 } && headFile().exists()

        fun open(threads: Int = 4): EmotionClassifier {
            val model = modelFile()
            val head = headFile()
            check(model.exists()) {
                "emotion2vec.onnx not found — download it in Settings."
            }
            check(head.exists()) {
                "emotion2vec_head.json not found. The classifier head is a " +
                    "separate file; re-download the model in Settings."
            }

            val (w, b, labels) = parseHead(head)

            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = try {
                env.createSession(model.absolutePath, opts)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Could not load emotion2vec.onnx (${e.message})", e
                )
            }
            val inputName = session.inputNames.firstOrNull()
                ?: error("emotion2vec.onnx exposes no inputs")

            return EmotionClassifier(env, session, inputName, w, b, labels)
        }

        /** head json: {"weight": [[...]], "bias": [...], "labels": [...]} */
        internal fun parseHead(file: File): Triple<Array<FloatArray>, FloatArray, List<String>> {
            val root = Json.parseToJsonElement(file.readText()).jsonObject

            val weight = root["weight"]?.jsonArray
                ?: error("emotion2vec_head.json has no 'weight'")
            val bias = root["bias"]?.jsonArray
                ?: error("emotion2vec_head.json has no 'bias'")
            val labelsJson = root["labels"]?.jsonArray
                ?: error("emotion2vec_head.json has no 'labels'")

            val w = Array(weight.size) { i ->
                val row = weight[i].jsonArray
                FloatArray(row.size) { j -> row[j].jsonPrimitive.content.toFloat() }
            }
            val b = FloatArray(bias.size) { i -> bias[i].jsonPrimitive.content.toFloat() }
            val labels = labelsJson.map { cleanLabel(it.jsonPrimitive.content) }

            check(w.size == b.size && w.size == labels.size) {
                "head mismatch: ${w.size} weights, ${b.size} biases, ${labels.size} labels"
            }
            return Triple(w, b, labels)
        }

        /**
         * Upstream labels look like "生气/angry" or "<unk>". Keep the English
         * side so downstream mapping and the UI stay readable.
         */
        internal fun cleanLabel(raw: String): String {
            val slash = raw.substringAfterLast('/', raw)
            return slash.trim().trim('<', '>').ifBlank { raw.trim() }
        }
    }

    fun classify(wav: File): Result {
        val samples = readWavNormalised(wav)
        check(samples.size >= MIN_SAMPLES) {
            "${wav.name} is too short to classify (${samples.size} samples)"
        }

        val shape = longArrayOf(1, samples.size.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(samples), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { out ->
                val pooled = meanPool(out[0].value)
                return score(pooled)
            }
        }
    }

    suspend fun classifyAll(
        clips: List<File>,
        onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }
    ): LinkedHashMap<String, Result> = withContext(Dispatchers.Default) {
        val out = LinkedHashMap<String, Result>(clips.size)
        clips.forEachIndexed { i, clip ->
            coroutineContext.ensureActive()
            if (clip.exists() && clip.length() > WavIo.HEADER_BYTES) {
                runCatching { classify(clip) }
                    .onSuccess { out[clip.nameWithoutExtension] = it }
            }
            onProgress(i + 1, clips.size)
        }
        out
    }

    /** Accepts (1, T, D) or (T, D) and averages over time. */
    private fun meanPool(value: Any?): FloatArray {
        @Suppress("UNCHECKED_CAST")
        val frames: Array<FloatArray> = when (value) {
            is Array<*> -> when (val first = value.firstOrNull()) {
                is Array<*> -> (value[0] as Array<FloatArray>)      // (1, T, D)
                is FloatArray -> value as Array<FloatArray>          // (T, D)
                else -> error("unexpected output element ${first?.javaClass}")
            }
            else -> error("unexpected emotion2vec output ${value?.javaClass}")
        }
        check(frames.isNotEmpty()) { "emotion2vec returned no frames" }
        val dim = frames[0].size
        val pooled = FloatArray(dim)
        for (row in frames) for (d in 0 until dim) pooled[d] += row[d]
        for (d in 0 until dim) pooled[d] /= frames.size
        return pooled
    }

    private fun score(pooled: FloatArray): Result {
        check(weight[0].size == pooled.size) {
            "head expects ${weight[0].size} dims but the model produced ${pooled.size}"
        }
        val logits = FloatArray(weight.size) { i ->
            var acc = 0.0
            val row = weight[i]
            for (d in pooled.indices) acc += row[d].toDouble() * pooled[d]
            (acc + bias[i]).toFloat()
        }
        val probs = softmax(logits)
        val ranked = labels.indices
            .map { labels[it] to probs[it] }
            .sortedByDescending { it.second }
        return Result(ranked[0].first, ranked[0].second, ranked)
    }

    private fun softmax(x: FloatArray): FloatArray {
        val max = x.max()
        var sum = 0.0
        val e = DoubleArray(x.size) { exp((x[it] - max).toDouble()).also { v -> sum += v } }
        return FloatArray(x.size) { (e[it] / sum).toFloat() }
    }

    /** 16-bit PCM wav -> float in [-1, 1], which is what this model expects. */
    private fun readWavNormalised(wav: File): FloatArray {
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
            out[i] = (acc / fmt.channels) / 32768f
        }
        return out
    }

    override fun close() {
        runCatching { session.close() }
    }
}
