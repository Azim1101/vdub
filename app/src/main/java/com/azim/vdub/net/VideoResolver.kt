package com.azim.vdub.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Turns a page URL into a downloadable media URL, entirely on-device.
 *
 * ## What this can and cannot do
 *
 * There is no yt-dlp for Android. yt-dlp is ~200k lines of Python with
 * site-specific extractors, and sites like iq.com additionally require a
 * JS engine to solve their signing challenge — that is why the original plan
 * used a PhantomJS server. None of that fits in an APK.
 *
 * What *is* achievable in-app, and what this does:
 *
 *  - direct media links (.mp4/.m4v/.webm/.mkv) — download straight through
 *  - HLS playlists (.m3u8) — ExoPlayer plays them, and we can pick a variant
 *  - pages that expose their media in `<video src>`, `<source src>`,
 *    Open Graph `og:video`, JSON-LD `contentUrl`, or a bare .mp4/.m3u8 in
 *    the HTML — a surprising number of small sites do
 *
 * What it deliberately does NOT pretend to do: DRM, signed/expiring CDN
 * tokens, or JS-generated URLs (iq.com, YouTube). For those the honest answer
 * is to download on a computer and pick the file from Gallery — which the app
 * says plainly instead of failing with a mystery error.
 */
@Singleton
class VideoResolver @Inject constructor() {

    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    sealed interface Resolution {
        /** A URL we can stream to disk. */
        data class Direct(val url: String, val kind: String) : Resolution
        /** We understand the site but cannot legally/technically extract it. */
        data class Unsupported(val site: String, val reason: String) : Resolution
    }

    private val mediaExt = listOf(".mp4", ".m4v", ".webm", ".mkv", ".mov", ".ts")

    /** Sites that need a JS engine or hold DRM — refuse clearly, up front. */
    private val knownBlocked = mapOf(
        "iq.com" to "iQIYI signs its streams with JavaScript and serves DRM",
        "iqiyi.com" to "iQIYI signs its streams with JavaScript and serves DRM",
        "youtube.com" to "YouTube generates URLs in JavaScript and rate-limits apps",
        "youtu.be" to "YouTube generates URLs in JavaScript and rate-limits apps",
        "netflix.com" to "DRM protected",
        "primevideo.com" to "DRM protected",
        "disneyplus.com" to "DRM protected",
        "hotstar.com" to "DRM protected"
    )

    suspend fun resolve(pageUrl: String): Resolution = withContext(Dispatchers.IO) {
        val url = pageUrl.trim()
        require(url.startsWith("http")) { "URL must start with http:// or https://" }

        val host = runCatching { java.net.URI(url).host.orEmpty().removePrefix("www.") }
            .getOrDefault("")

        knownBlocked.entries.firstOrNull { host.endsWith(it.key) }?.let {
            return@withContext Resolution.Unsupported(it.key, it.value)
        }

        val bare = url.substringBefore('?').lowercase()
        if (mediaExt.any { bare.endsWith(it) }) {
            return@withContext Resolution.Direct(url, "direct file")
        }
        if (bare.endsWith(".m3u8")) {
            return@withContext Resolution.Direct(url, "HLS playlist")
        }

        // Fetch the page and look for embedded media.
        val html = runCatching {
            val req = Request.Builder()
                .url(url)
                .header(
                    "User-Agent",
                    "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 " +
                        "(KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"
                )
                .build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                val type = resp.header("Content-Type").orEmpty()
                if (type.startsWith("video/") || type == "application/octet-stream") {
                    return@withContext Resolution.Direct(url, "direct stream")
                }
                resp.body?.string().orEmpty()
            }
        }.getOrElse {
            return@withContext Resolution.Unsupported(
                host.ifBlank { "site" },
                "could not fetch the page (${it.message})"
            )
        }

        extractMediaUrl(html, url)?.let {
            return@withContext Resolution.Direct(it, "found in page")
        }

        Resolution.Unsupported(
            host.ifBlank { "site" },
            "no direct media link in the page — it is probably built by JavaScript"
        )
    }

    /** Scan HTML for the usual places a media URL hides. */
    internal fun extractMediaUrl(html: String, baseUrl: String): String? {
        val patterns = listOf(
            // <video src="..."> / <source src="...">
            Regex("""<(?:video|source)[^>]+src\s*=\s*["']([^"']+)["']""", RegexOption.IGNORE_CASE),
            // <meta property="og:video" content="...">
            Regex(
                """<meta[^>]+(?:property|name)\s*=\s*["']og:video(?::url|:secure_url)?["'][^>]+content\s*=\s*["']([^"']+)["']""",
                RegexOption.IGNORE_CASE
            ),
            // JSON-LD / inline JSON  "contentUrl": "..."
            Regex("""["'](?:contentUrl|playUrl|videoUrl|file|hlsUrl)["']\s*:\s*["']([^"']+)["']""", RegexOption.IGNORE_CASE),
            // bare URL anywhere in the markup
            Regex("""https?://[^\s"'<>\\]+?\.(?:mp4|m3u8|webm)(?:\?[^\s"'<>\\]*)?""", RegexOption.IGNORE_CASE)
        )

        for (re in patterns) {
            val m = re.find(html) ?: continue
            val raw = (m.groupValues.getOrNull(1)?.takeIf { it.isNotBlank() } ?: m.value)
                .replace("\\/", "/")
                .replace("&amp;", "&")
                .trim()
            if (raw.isBlank()) continue
            val abs = absolutise(raw, baseUrl) ?: continue
            val bare = abs.substringBefore('?').lowercase()
            if (mediaExt.any { bare.endsWith(it) } || bare.endsWith(".m3u8")) return abs
        }
        return null
    }

    private fun absolutise(link: String, baseUrl: String): String? = runCatching {
        when {
            link.startsWith("http") -> link
            link.startsWith("//") -> "https:$link"
            else -> java.net.URI(baseUrl).resolve(link).toString()
        }
    }.getOrNull()
}
