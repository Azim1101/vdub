package com.azim.vdub.net

import com.azim.vdub.BuildConfig
import com.azim.vdub.data.model.DownloadRequest
import com.azim.vdub.data.model.DownloadResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext

/**
 * Video acquisition over the network.
 *
 * URL downloads are delegated to a helper server (Kaggle/Colab box) rather than
 * being done on-device: sites like iq.com need yt-dlp + a PhantomJS binary and
 * OPENSSL_CONF=/dev/null, none of which can run inside an APK.
 *
 * Server contract:
 *   POST {server}/download  {"url":..., "format":"500", "project":...}
 *     -> {"ok":true, "file_url":"/files/xxx.mp4", "size_bytes":149000000}
 *   GET  {file_url}         -> the mp4 bytes (Range-resumable)
 */
@Singleton
class DownloadClient @Inject constructor() {

    private val json = Json { ignoreUnknownKeys = true; explicitNulls = false }

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.MINUTES)   // yt-dlp on a 142 MB file takes a while
        .writeTimeout(60, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    var serverBase: String = BuildConfig.DOWNLOAD_SERVER

    /** Ask the server to resolve+fetch the URL, then stream the mp4 down. */
    suspend fun downloadFromUrl(
        pageUrl: String,
        project: String,
        target: File,
        format: String = "500",
        onProgress: (bytes: Long, total: Long) -> Unit = { _, _ -> }
    ): File = withContext(Dispatchers.IO) {
        val base = serverBase.trimEnd('/')
        val payload = json.encodeToString(
            DownloadRequest.serializer(),
            DownloadRequest(url = pageUrl, format = format, project = project)
        )
        val req = Request.Builder()
            .url("$base/download")
            .post(payload.toRequestBody("application/json".toMediaType()))
            .build()

        val meta: DownloadResponse = client.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) {
                error("Server ${resp.code}: ${body.take(300)}")
            }
            json.decodeFromString(DownloadResponse.serializer(), body)
        }
        if (!meta.ok || meta.fileUrl.isNullOrBlank()) {
            error(meta.error ?: "Server returned no file")
        }

        val fileUrl = if (meta.fileUrl.startsWith("http")) meta.fileUrl
        else "$base/${meta.fileUrl.trimStart('/')}"

        streamTo(fileUrl, target, meta.sizeBytes ?: -1L, onProgress)
        target
    }

    /** Straight HTTP(S) GET — direct mp4 links and Drive `uc?export=download`. */
    suspend fun downloadDirect(
        fileUrl: String,
        target: File,
        onProgress: (bytes: Long, total: Long) -> Unit = { _, _ -> }
    ): File = withContext(Dispatchers.IO) {
        streamTo(fileUrl, target, -1L, onProgress)
        target
    }

    private suspend fun streamTo(
        url: String,
        target: File,
        knownSize: Long,
        onProgress: (Long, Long) -> Unit
    ) {
        target.parentFile?.mkdirs()
        val partial = File(target.parentFile, target.name + ".part")
        val already = if (partial.exists()) partial.length() else 0L

        val builder = Request.Builder().url(url)
        if (already > 0) builder.header("Range", "bytes=$already-")

        client.newCall(builder.build()).execute().use { resp ->
            if (!resp.isSuccessful) error("HTTP ${resp.code} fetching $url")
            val resuming = resp.code == 206 && already > 0
            val bodyLen = resp.body?.contentLength() ?: -1L
            val total = when {
                knownSize > 0 -> knownSize
                bodyLen > 0 -> bodyLen + if (resuming) already else 0L
                else -> -1L
            }
            if (!resuming) partial.delete()

            val sink = java.io.RandomAccessFile(partial, "rw")
            sink.use { out ->
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
                        onProgress(written, total)
                    }
                }
                onProgress(written, total)
            }
        }
        if (target.exists()) target.delete()
        check(partial.renameTo(target)) { "Could not finalize ${target.name}" }
    }

    /** Quick reachability probe so the UI can show server status. */
    suspend fun ping(): Boolean = withContext(Dispatchers.IO) {
        runCatching {
            val req = Request.Builder().url("${serverBase.trimEnd('/')}/health").get().build()
            client.newCall(req).execute().use { it.isSuccessful }
        }.getOrDefault(false)
    }
}
