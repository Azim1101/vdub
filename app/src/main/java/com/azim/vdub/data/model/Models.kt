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

/**
 * How strongly a line should be delivered, per emotion.
 * Feeds the TTS stage: ANGRY 1.4 / HAPPY 1.1 etc. from the pipeline notes.
 */
object EmotionStyle {
    private val EXAGGERATION = mapOf(
        "angry" to 1.4f,
        "happy" to 1.1f,
        "sad" to 0.9f,
        "surprised" to 1.25f,
        "fearful" to 1.15f,
        "disgusted" to 1.2f,
        "neutral" to 1.0f,
        "other" to 1.0f,
        "unk" to 1.0f
    )

    const val DEFAULT = "neutral"

    val KNOWN: List<String> = EXAGGERATION.keys.toList()

    fun exaggeration(emotion: String?): Float =
        EXAGGERATION[emotion?.lowercase()?.trim()] ?: 1.0f

    fun normalise(raw: String?): String {
        val e = raw?.lowercase()?.trim().orEmpty()
        return if (e.isBlank() || e !in EXAGGERATION) DEFAULT else e
    }
}

/** Step 2 output: out/script_speakers.json */
@Serializable
data class SpeakerLine(
    @SerialName("utt") val utt: String,              // line_0000
    @SerialName("start") val start: Double,
    @SerialName("end") val end: Double,
    @SerialName("text") val text: String,
    @SerialName("spk") val spk: String,              // "Speaker 1"
    @SerialName("emotion") val emotion: String = "neutral",
    @SerialName("emotion_score") val emotionScore: Float = 0f,
    @SerialName("hi") val hi: String = ""            // translated text, Step 4
)

/** Step 3 output: out/script_emotion.json */
@Serializable
data class EmotionScript(
    @SerialName("project") val project: String,
    @SerialName("counts") val counts: Map<String, Int> = emptyMap(),
    @SerialName("lines") val lines: List<SpeakerLine> = emptyList()
)

@Serializable
data class SpeakerScript(
    @SerialName("project") val project: String,
    @SerialName("speaker_count") val speakerCount: Int,
    @SerialName("threshold") val threshold: Float,
    @SerialName("names") val names: Map<String, String> = emptyMap(),
    @SerialName("lines") val lines: List<SpeakerLine> = emptyList()
)

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
