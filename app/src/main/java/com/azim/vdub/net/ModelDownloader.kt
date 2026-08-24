package com.azim.vdub.net

import com.azim.vdub.core.VdubPaths
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext

/**
 * Downloads the ONNX models into /AI/models straight from the app.
 *
 * `adb push` assumes a PC and a cable; the phone can fetch these itself.
 * Downloads are Range-resumable and validated, because a truncated or
 * HTML-error-page "model" fails much later inside ONNX Runtime with a
 * meaningless error.
 */
@Singleton
class ModelDownloader @Inject constructor() {

    data class ModelSpec(
        val fileName: String,
        val label: String,
        val approxBytes: Long,
        /** Tried in order; first reachable one wins. */
        val urls: List<String>,
        val note: String = ""
    )

    companion object {
        /**
         * CAM++ speaker embedding, 3D-Speaker, 192-dim, 80-dim fbank input.
         * Exactly the 28 MB checkpoint the pipeline expects.
         */
        val CAMPPLUS = ModelSpec(
            fileName = "campplus.onnx",
            label = "campplus (speaker embedding)",
            approxBytes = 28_283_928L,
            urls = listOf(
                "https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx/resolve/main/campplus_cn_en_common_200k.onnx?download=true",
                // hf-mirror is the usual fallback when huggingface.co is slow/blocked
                "https://hf-mirror.com/welcomyou/campplus-3dspeaker-200k-onnx/resolve/main/campplus_cn_en_common_200k.onnx?download=true",
                "https://huggingface.co/Luigi/campplus-zh-en-onnx/resolve/main/campplus_zh_en_fp32.onnx?download=true",
                "https://hf-mirror.com/Luigi/campplus-zh-en-onnx/resolve/main/campplus_zh_en_fp32.onnx?download=true"
            ),
            note = "CAM++ zh+en, 192-dim embeddings"
        )

        val ALL = listOf(CAMPPLUS)
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .build()

    sealed interface Progress {
        data class Downloading(
            val bytes: Long,
            val total: Long,
            val mirror: Int,
            val mirrorCount: Int
        ) : Progress
        data class Verifying(val bytes: Long) : Progress
    }

    /**
     * Fetch [spec] into /AI/models, trying each mirror in turn.
     * @return the installed file.
     */
    suspend fun download(
        spec: ModelSpec,
        onProgress: (Progress) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {
        val dir = VdubPaths.modelsDir
        dir.mkdirs()
        check(dir.isDirectory) {
            "Cannot create ${dir.absolutePath} — grant All-files access or the " +
                "app will use its private folder."
        }

        val target = File(dir, spec.fileName)
        val partial = File(dir, spec.fileName + ".part")
        val failures = StringBuilder()

        spec.urls.forEachIndexed { index, url ->
            coroutineContext.ensureActive()
            try {
                fetch(url, partial, spec, index, spec.urls.size, onProgress)
                onProgress(Progress.Verifying(partial.length()))
                validate(partial, spec)
                if (target.exists()) target.delete()
                check(partial.renameTo(target)) { "could not move into place" }
                return@withContext target
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                failures.append("\n• mirror ${index + 1}: ${e.message}")
                // A corrupt partial must not poison the next mirror.
                partial.delete()
            }
        }

        error("Could not download ${spec.label}.$failures")
    }

    private suspend fun fetch(
        url: String,
        partial: File,
        spec: ModelSpec,
        mirror: Int,
        mirrorCount: Int,
        onProgress: (Progress) -> Unit
    ) {
        val already = if (partial.exists()) partial.length() else 0L
        val builder = Request.Builder()
            .url(url)
            .header("User-Agent", "vdub-android")
        if (already > 0) builder.header("Range", "bytes=$already-")

        client.newCall(builder.build()).execute().use { resp ->
            if (!resp.isSuccessful) error("HTTP ${resp.code}")

            val resuming = resp.code == 206 && already > 0
            val bodyLen = resp.body?.contentLength() ?: -1L
            val total = when {
                bodyLen > 0 -> bodyLen + if (resuming) already else 0L
                spec.approxBytes > 0 -> spec.approxBytes
                else -> -1L
            }
            if (!resuming) partial.delete()

            RandomAccessFile(partial, "rw").use { out ->
                if (resuming) out.seek(already) else out.setLength(0)
                var written = if (resuming) already else 0L
                val buf = ByteArray(256 * 1024)
                val src = resp.body!!.byteStream()
                var lastTick = 0L
                while (true) {
                    coroutineContext.ensureActive()
                    val n = src.read(buf)
                    if (n <= 0) break
                    out.write(buf, 0, n)
                    written += n
                    val now = System.currentTimeMillis()
                    if (now - lastTick > 150) {
                        lastTick = now
                        onProgress(Progress.Downloading(written, total, mirror, mirrorCount))
                    }
                }
                onProgress(Progress.Downloading(written, total, mirror, mirrorCount))
            }
        }
    }

    /**
     * Reject the two failure modes that otherwise surface as an unreadable
     * ONNX Runtime error much later: an HTML error page saved as .onnx, and a
     * truncated transfer.
     */
    private fun validate(file: File, spec: ModelSpec) {
        val size = file.length()
        if (size < 1_000_000) {
            val head = file.inputStream().use { String(it.readNBytes(200)) }.trim()
            if (head.startsWith("<") || head.contains("html", ignoreCase = true)) {
                error("server returned a web page, not the model")
            }
            error("file is only $size bytes — download was cut short")
        }
        if (spec.approxBytes > 0) {
            val ratio = size.toDouble() / spec.approxBytes
            if (ratio < 0.5) error("only ${pct(ratio)} of the expected size arrived")
        }
        // ONNX is protobuf: field 1 (ir_version) varint -> first byte 0x08.
        val magic = file.inputStream().use { it.readNBytes(1) }
        if (magic.isEmpty() || magic[0] != 0x08.toByte()) {
            error("not a valid ONNX file (bad header)")
        }
    }

    private fun pct(r: Double) = "%.0f%%".format(r * 100)
}
