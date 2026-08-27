package com.azim.vdub.net

import com.azim.vdub.core.ModelCatalog
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
 * Fetches models from the catalog into /AI/models.
 *
 * Downloads are Range-resumable and validated before install: an HTML error
 * page saved as .onnx and a truncated transfer both otherwise fail much later
 * inside ONNX Runtime with an error that says nothing useful.
 */
@Singleton
class ModelDownloader @Inject constructor() {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .build()

    data class Progress(
        val fileIndex: Int,
        val fileCount: Int,
        val fileName: String,
        val bytes: Long,
        val total: Long,
        val mirror: Int,
        val verifying: Boolean = false
    ) {
        /** Overall fraction across all files of the model. */
        val fraction: Float
            get() {
                if (fileCount <= 0) return -1f
                val within = if (total > 0) bytes.toFloat() / total else 0f
                return ((fileIndex + within) / fileCount).coerceIn(0f, 1f)
            }
    }

    /** True when every file of [model] is present and plausible. */
    fun isInstalled(model: ModelCatalog.Model): Boolean =
        model.files.all { f ->
            val file = File(VdubPaths.modelsDir, f.localName)
            file.exists() && file.length() >= (f.approxBytes * 0.5).toLong().coerceAtLeast(64L)
        }

    fun installedBytes(model: ModelCatalog.Model): Long =
        model.files.sumOf { File(VdubPaths.modelsDir, it.localName).length() }

    fun modelFile(name: String): File = File(VdubPaths.modelsDir, name)

    /** Remove a model's files to reclaim space. */
    fun delete(model: ModelCatalog.Model) {
        model.files.forEach { File(VdubPaths.modelsDir, it.localName).delete() }
    }

    suspend fun download(
        model: ModelCatalog.Model,
        onProgress: (Progress) -> Unit = {}
    ) = withContext(Dispatchers.IO) {
        val dir = VdubPaths.modelsDir
        dir.mkdirs()
        check(dir.isDirectory) {
            "Cannot create ${dir.absolutePath} — grant All-files access, or the " +
                "app will keep using its private folder."
        }

        model.files.forEachIndexed { index, spec ->
            coroutineContext.ensureActive()
            val target = File(dir, spec.localName)
            target.parentFile?.mkdirs()

            // Skip files already installed, so a retry resumes at the model level.
            if (target.exists() &&
                target.length() >= (spec.approxBytes * 0.5).toLong().coerceAtLeast(64L)
            ) {
                onProgress(
                    Progress(index, model.files.size, spec.localName,
                        target.length(), target.length(), 0)
                )
                return@forEachIndexed
            }

            val partial = File(dir, spec.localName + ".part")
            val failures = StringBuilder()
            var installed = false

            for ((mirror, url) in spec.urls.withIndex()) {
                coroutineContext.ensureActive()
                try {
                    fetch(url, partial, spec, index, model.files.size, mirror, onProgress)
                    onProgress(
                        Progress(index, model.files.size, spec.localName,
                            partial.length(), partial.length(), mirror, verifying = true)
                    )
                    validate(partial, spec)
                    if (target.exists()) target.delete()
                    check(partial.renameTo(target)) { "could not move into place" }
                    installed = true
                    break
                } catch (e: kotlinx.coroutines.CancellationException) {
                    throw e
                } catch (e: Exception) {
                    failures.append("\n• mirror ${mirror + 1}: ${e.message}")
                    partial.delete()   // never resume into a corrupt partial
                }
            }

            check(installed) { "Could not download ${spec.localName}.$failures" }
        }
    }

    private suspend fun fetch(
        url: String,
        partial: File,
        spec: ModelCatalog.ModelFile,
        fileIndex: Int,
        fileCount: Int,
        mirror: Int,
        onProgress: (Progress) -> Unit
    ) {
        val already = if (partial.exists()) partial.length() else 0L
        val builder = Request.Builder().url(url).header("User-Agent", "vdub-android")
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
                        onProgress(
                            Progress(fileIndex, fileCount, spec.localName,
                                written, total, mirror)
                        )
                    }
                }
                onProgress(
                    Progress(fileIndex, fileCount, spec.localName, written, total, mirror)
                )
            }
        }
    }

    private fun validate(file: File, spec: ModelCatalog.ModelFile) {
        val size = file.length()
        if (size == 0L) error("empty file")

        val head = file.inputStream().use { it.readNBytes(256) }
        val asText = String(head).trim()

        // Detect an HTML error page saved under the model's name. Match real
        // markup only — SenseVoice's tokens.txt legitimately begins "<unk> 0",
        // and a naive startsWith("<") rejected the whole download.
        val looksLikeHtml = Regex(
            """^\s*(<!DOCTYPE\s+html|<html\b|<head\b|<body\b|<\?xml)""",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(asText)
        if (looksLikeHtml) {
            error("server returned a web page, not the file")
        }
        if (spec.approxBytes > 0 && size < spec.approxBytes * 0.5) {
            error("only ${size / 1024} KB of ~${spec.approxBytes / 1024} KB arrived")
        }
        when (spec.kind) {
            // ONNX is protobuf: field 1 (ir_version) varint -> first byte 0x08.
            ModelCatalog.Kind.ONNX ->
                if (head.isEmpty() || head[0] != 0x08.toByte()) {
                    error("not a valid ONNX file (bad header)")
                }
            ModelCatalog.Kind.JSON ->
                if (!asText.startsWith("{") && !asText.startsWith("[")) {
                    error("not valid JSON")
                }
            // .npz is a zip archive: "PK\003\004", or "PK\005\006" when empty.
            // Worth checking — DhVaani's mel filterbank and vocoder head are
            // npz, and a truncated one fails much later as a shape mismatch
            // deep inside the vocoder.
            ModelCatalog.Kind.NPZ ->
                if (head.size < 4 || head[0] != 'P'.code.toByte() ||
                    head[1] != 'K'.code.toByte()
                ) {
                    error("not a valid .npz archive (bad header)")
                }
            ModelCatalog.Kind.TEXT, ModelCatalog.Kind.BIN,
            ModelCatalog.Kind.ONNX_DATA -> Unit
        }
    }
}
