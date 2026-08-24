package com.azim.vdub.data.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** One SRT cue as parsed from file. */
data class SrtCue(
    val index: Int,
    val startMs: Long,
    val endMs: Long,
    val text: String
)

/**
 * One merged dialogue line -> becomes one clip -> one TTS utterance.
 * Serialized into out/script_raw.json (the 190-line file from the spec).
 */
@Serializable
data class ScriptLine(
    @SerialName("id") val id: Int,
    @SerialName("start") val start: Double,          // seconds
    @SerialName("end") val end: Double,              // seconds
    @SerialName("text") val text: String,
    @SerialName("translated") val translated: String? = null,
    @SerialName("clip") val clip: String? = null,    // clips/line_0000.wav
    @SerialName("speaker") val speaker: String? = null,  // Step 2
    @SerialName("emotion") val emotion: String? = null   // Step 3
) {
    val durationSec: Double get() = (end - start).coerceAtLeast(0.0)
}

@Serializable
data class ScriptRaw(
    @SerialName("project") val project: String,
    @SerialName("source_video") val sourceVideo: String? = null,
    @SerialName("sample_rate") val sampleRate: Int = 16_000,
    @SerialName("pad_sec") val padSec: Double = 0.2,
    @SerialName("cue_count") val cueCount: Int = 0,
    @SerialName("lines") val lines: List<ScriptLine> = emptyList()
)

/** Where the source video came from. */
enum class VideoSource { NONE, GALLERY, URL, DRIVE }

/** Progress envelope used by every long-running op in Step 1. */
sealed interface JobState {
    data object Idle : JobState
    data class Running(
        val label: String,
        val progress: Float = -1f,   // -1 = indeterminate
        val detail: String = ""
    ) : JobState
    data class Done(val label: String, val detail: String = "") : JobState
    data class Failed(val label: String, val error: String) : JobState
}

/** Server contract for the yt-dlp / PhantomJS download box. */
@Serializable
data class DownloadRequest(
    val url: String,
    val format: String = "500",     // 720p per spec: yt-dlp -f 500
    val project: String
)

@Serializable
data class DownloadResponse(
    val ok: Boolean,
    @SerialName("file_url") val fileUrl: String? = null,
    @SerialName("size_bytes") val sizeBytes: Long? = null,
    val title: String? = null,
    val error: String? = null
)
