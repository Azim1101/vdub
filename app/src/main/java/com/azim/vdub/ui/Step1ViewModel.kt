package com.azim.vdub.ui

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.ScriptLine
import com.azim.vdub.data.model.VideoSource
import com.azim.vdub.data.repo.ProjectRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject

data class Step1UiState(
    val projectName: String = "vdub_step",
    val knownProjects: List<String> = emptyList(),
    val videoPath: String? = null,
    val videoSource: VideoSource = VideoSource.NONE,
    val durationMs: Long = 0L,
    val videoSizeBytes: Long = 0L,
    val sourceUrl: String = "",
    val storageShared: Boolean = true,
    val storagePath: String = "",
    val srtPath: String? = null,
    val translatedSrtPath: String? = null,
    val cueCount: Int = 0,
    val lineCount: Int = 0,
    val translatedCount: Int = 0,
    val clipCount: Int = 0,
    val clipsSizeBytes: Long = 0L,
    val mergeGapMs: Long = 400,
    val lines: List<ScriptLine> = emptyList(),
    val step1Done: Boolean = false,
    val job: JobState = JobState.Idle,
    val message: String? = null
) {
    val hasVideo: Boolean get() = !videoPath.isNullOrBlank()
    val hasSubtitles: Boolean get() = lineCount > 0
    val hasTranslation: Boolean get() = translatedCount > 0
    val canTrim: Boolean get() = hasVideo && hasSubtitles && job !is JobState.Running
    val busy: Boolean get() = job is JobState.Running
    val durationMinutes: Double get() = durationMs / 60_000.0
}

@HiltViewModel
class Step1ViewModel @Inject constructor(
    private val repo: ProjectRepository
) : ViewModel() {

    private val _state = MutableStateFlow(Step1UiState())
    val state: StateFlow<Step1UiState> = _state.asStateFlow()

    private var runningJob: Job? = null

    init {
        viewModelScope.launch {
            VdubPaths.ensureRoots()
            // Deliberately do NOT open a project here. Auto-loading "vdub_step"
            // meant a previous run's video, SRT and clips reappeared on launch
            // with no way to tell they were stale. The user picks or creates
            // one via Open / Resume.
            _state.update {
                it.copy(
                    knownProjects = VdubPaths.listProjects(),
                    storageShared = VdubPaths.usingSharedStorage,
                    storagePath = VdubPaths.projectsRoot.absolutePath
                )
            }
        }
    }

    // ------------------------------------------------------------- project

    /**
     * Editing the name detaches from whatever was loaded — otherwise the old
     * project's video and clips stay on screen under a new name.
     */
    fun setProjectName(name: String) {
        _state.update {
            if (name == it.projectName) it
            else Step1UiState(
                projectName = name,
                knownProjects = it.knownProjects,
                storageShared = it.storageShared,
                storagePath = it.storagePath,
                mergeGapMs = it.mergeGapMs
            )
        }
    }

    /** Wipe this project's files and start over. */
    fun resetProject() = launchJob("Clearing project") {
        val name = _state.value.projectName
        repo.resetProject(name)
        _state.update {
            Step1UiState(
                projectName = name,
                knownProjects = VdubPaths.listProjects(),
                storageShared = VdubPaths.usingSharedStorage,
                storagePath = VdubPaths.projectDir(name).absolutePath,
                mergeGapMs = it.mergeGapMs,
                job = JobState.Done("Project cleared", name)
            )
        }
    }

    /** Drop just the video, keeping subtitles. */
    fun clearVideo() = launchJob("Removing video") {
        val name = _state.value.projectName
        repo.clearVideo(name)
        _state.update {
            it.copy(
                videoPath = null,
                videoSource = VideoSource.NONE,
                durationMs = 0L,
                videoSizeBytes = 0L,
                sourceUrl = "",
                clipCount = 0,
                clipsSizeBytes = 0L,
                step1Done = false,
                job = JobState.Done("Video removed", "pick another")
            )
        }
    }

    /** Resume-safe: pull whatever already exists on disk + in Room. */
    fun loadProject(name: String = _state.value.projectName) = viewModelScope.launch {
        runCatching {
            repo.ensureProject(name)
            val entity = repo.rehydrate(name)
            val script = repo.readScriptRaw(name)
            val videoFile = entity.videoPath?.let(::File)
            _state.update {
                it.copy(
                    projectName = name,
                    storageShared = VdubPaths.usingSharedStorage,
                    storagePath = VdubPaths.projectDir(name).absolutePath,
                    knownProjects = VdubPaths.listProjects(),
                    videoPath = entity.videoPath,
                    videoSource = runCatching { VideoSource.valueOf(entity.videoSource) }
                        .getOrDefault(VideoSource.NONE),
                    durationMs = entity.durationMs,
                    videoSizeBytes = videoFile?.length() ?: 0L,
                    sourceUrl = entity.sourceUrl ?: it.sourceUrl,
                    srtPath = entity.srtPath,
                    translatedSrtPath = entity.translatedSrtPath,
                    cueCount = entity.cueCount,
                    lineCount = entity.lineCount,
                    translatedCount = script?.lines
                        ?.count { l -> !l.translated.isNullOrBlank() } ?: 0,
                    clipCount = entity.clipCount,
                    clipsSizeBytes = VdubPaths.sizeOf(VdubPaths.clipsDir(name)),
                    lines = script?.lines.orEmpty(),
                    step1Done = entity.step1Done,
                    job = JobState.Idle,
                    message = if (entity.step1Done)
                        "Step 1 already complete for \"$name\" — you can jump to Step 2."
                    else null
                )
            }
        }.onFailure { e -> fail("Load project", e) }
    }

    fun setSourceUrl(url: String) = _state.update { it.copy(sourceUrl = url) }
    fun dismissMessage() = _state.update { it.copy(message = null) }


    fun cancel() {
        runningJob?.cancel()
        runningJob = null
        _state.update { it.copy(job = JobState.Idle, message = "Cancelled") }
    }

    // --------------------------------------------------------------- video

    fun importVideo(uri: Uri) = launchJob("Importing video") {
        val project = _state.value.projectName
        val file = repo.importVideoFromUri(project, uri) { copied, total ->
            progress(
                "Importing video",
                if (total > 0) copied.toFloat() / total else -1f,
                "${mb(copied)} / ${if (total > 0) mb(total) else "?"}"
            )
        }
        afterVideo(file, VideoSource.GALLERY)
    }

    fun downloadVideo() = launchJob("Downloading video") {
        val s = _state.value
        require(s.sourceUrl.isNotBlank()) { "Paste a video URL first" }
        val file = repo.downloadVideoFromUrl(
            project = s.projectName,
            url = s.sourceUrl.trim(),
        ) { got, total ->
            progress(
                "Downloading video",
                if (total > 0) got.toFloat() / total else -1f,
                "${mb(got)} / ${if (total > 0) mb(total) else "?"}"
            )
        }
        afterVideo(file, VideoSource.URL)
    }

    private suspend fun afterVideo(file: File, source: VideoSource) {
        val duration = repo.probeDurationMs(file)
        _state.update {
            it.copy(
                videoPath = file.absolutePath,
                videoSource = source,
                durationMs = duration,
                videoSizeBytes = file.length(),
                step1Done = false,
                job = JobState.Done("Video ready", "${mb(file.length())} · ${fmtMin(duration)}")
            )
        }
    }

    // ----------------------------------------------------------- subtitles

    fun importSrt(uri: Uri) = launchJob("Parsing subtitles") {
        val project = _state.value.projectName
        progress("Parsing subtitles", -1f, "reading SRT")
        val (cues, lines) = repo.importSrt(project, uri, _state.value.mergeGapMs)
        _state.update {
            it.copy(
                srtPath = VdubPaths.originalSrt(project).absolutePath,
                cueCount = cues.size,
                lineCount = lines.size,
                lines = lines,
                clipCount = 0,
                step1Done = false,
                job = JobState.Done(
                    "Subtitles ready",
                    "${cues.size} cues → ${lines.size} lines"
                )
            )
        }
    }

    /**
     * Re-merge cues with a new gap threshold to hit a target line count
     * (the spec's 473 cues -> 190 lines is subtitle-specific).
     */
    fun setMergeGap(gapMs: Long) = launchJob("Re-merging lines") {
        _state.update { it.copy(mergeGapMs = gapMs) }
        val project = _state.value.projectName
        if (VdubPaths.originalSrt(project).exists()) {
            val lines = repo.remergeLines(project, gapMs)
            _state.update {
                it.copy(
                    lineCount = lines.size,
                    lines = lines,
                    clipCount = 0,
                    translatedCount = lines.count { l -> !l.translated.isNullOrBlank() },
                    step1Done = false,
                    job = JobState.Done(
                        "Re-merged",
                        "${it.cueCount} cues -> ${lines.size} lines @ ${gapMs}ms"
                    )
                )
            }
        } else {
            _state.update { it.copy(job = JobState.Idle) }
        }
    }

    fun importTranslatedSrt(uri: Uri) = launchJob("Importing translation") {
        val project = _state.value.projectName
        progress("Importing translation", -1f, "matching lines")
        val matched = repo.importTranslatedSrt(project, uri)
        val script = repo.readScriptRaw(project)
        _state.update {
            it.copy(
                translatedSrtPath = VdubPaths.translatedSrt(project).absolutePath,
                translatedCount = matched,
                lines = script?.lines.orEmpty(),
                job = JobState.Done(
                    "Translation attached",
                    "$matched / ${it.lineCount} lines"
                )
            )
        }
    }

    fun exportScriptForTranslation() = launchJob("Exporting SRT") {
        val out = repo.exportScriptAsSrt(_state.value.projectName)
        _state.update {
            it.copy(job = JobState.Done("SRT exported", out.absolutePath))
        }
    }

    fun autoAsrPlaceholder() {
        _state.update {
            it.copy(
                message = "Auto ASR (SenseVoice) runs in Step 2 — " +
                    "place sensevoice.onnx in /AI/models/ and re-open."
            )
        }
    }

    fun autoTranslatePlaceholder() {
        _state.update {
            it.copy(
                message = "Auto NLLB translation is Step 4 — " +
                    "it needs the 0.9 GB q8 model on the server."
            )
        }
    }

    // -------------------------------------------------------------- trim

    fun trim() = launchJob("Trimming clips") {
        val project = _state.value.projectName
        val result = repo.trimIntoClips(
            project = project,
            onExtractProgress = { p ->
                progress("Extracting audio", p * 0.35f, "16 kHz mono WAV")
            },
            onClipProgress = { done, total ->
                progress(
                    "Cutting clips",
                    0.35f + 0.65f * (done.toFloat() / total.coerceAtLeast(1)),
                    "$done / $total"
                )
            }
        )
        _state.update {
            it.copy(
                clipCount = result.clipCount,
                clipsSizeBytes = VdubPaths.sizeOf(result.outputDir),
                step1Done = true,
                job = JobState.Done(
                    "Step 1 complete",
                    "${result.clipCount} clips · ${"%.1f".format(result.totalSeconds / 60)} min"
                )
            )
        }
    }

    // ------------------------------------------------------------- helpers

    private fun launchJob(label: String, block: suspend () -> Unit) {
        if (_state.value.busy) return
        runningJob?.cancel()
        runningJob = viewModelScope.launch {
            _state.update { it.copy(job = JobState.Running(label), message = null) }
            runCatching { block() }
                .onFailure { e ->
                    if (e is kotlinx.coroutines.CancellationException) throw e
                    fail(label, e)
                }
        }
    }

    private fun progress(label: String, value: Float, detail: String) {
        _state.update { it.copy(job = JobState.Running(label, value, detail)) }
    }

    private fun fail(label: String, e: Throwable) {
        _state.update {
            it.copy(
                job = JobState.Failed(label, e.message ?: e::class.simpleName.orEmpty()),
                message = "$label failed: ${e.message}"
            )
        }
    }

    private fun mb(bytes: Long): String =
        if (bytes < 1024 * 1024) "%.0f KB".format(bytes / 1024.0)
        else "%.1f MB".format(bytes / 1024.0 / 1024.0)

    private fun fmtMin(ms: Long): String = "%.1f min".format(ms / 60_000.0)
}
