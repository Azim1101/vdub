package com.azim.vdub.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext

/**
 * Streams a media URL to disk.
 *
 * Resolving a *page* into a media URL is [VideoResolver]'s job; this class
 * only moves bytes. Transfers are Range-resumable via a .part file, so a
 * dropped connection continues instead of restarting a 142 MB download.
 */
@Singleton
class DownloadClient @Inject constructor() {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.MINUTES)   // large files
        .writeTimeout(60, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    /** Stream a resolved media URL to [target]. */
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

}
