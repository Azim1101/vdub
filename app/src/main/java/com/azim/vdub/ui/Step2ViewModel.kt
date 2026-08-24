package com.azim.vdub.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.audio.SpeakerCluster
import com.azim.vdub.audio.SpeakerEmbedder
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.repo.ProjectRepository
import com.azim.vdub.net.ModelDownloader
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SpeakerInfo(
    val id: String,          // "Speaker 1"
    val displayName: String, // user-typed, or id
    val clipCount: Int,
    val colorIndex: Int
)

data class Step2UiState(
    val projectName: String = "vdub_step",
    val modelPresent: Boolean = false,
    val modelPath: String = "",
    val modelSizeBytes: Long = 0L,
    val clipCount: Int = 0,
    val embedCount: Int = 0,
    val threshold: Float = SpeakerCluster.DEFAULT_THRESHOLD,
    val useTargetK: Boolean = false,
    val targetK: Int = 3,
    val speakers: List<SpeakerInfo> = emptyList(),
    val lines: List<SpeakerLine> = emptyList(),
    val names: Map<String, String> = emptyMap(),
    val step2Done: Boolean = false,
    val job: JobState = JobState.Idle,
    val message: String? = null
) {
    val busy: Boolean get() = job is JobState.Running
    val hasEmbeds: Boolean get() = embedCount > 0
    val speakerCount: Int get() = speakers.size
    fun nameFor(id: String): String = names[id] ?: id
}

@HiltViewModel
class Step2ViewModel @Inject constructor(
    private val repo: ProjectRepository,
    private val modelDownloader: ModelDownloader
) : ViewModel() {

    private val _state = MutableStateFlow(Step2UiState())
    val state: StateFlow<Step2UiState> = _state.asStateFlow()

    private var runningJob: Job? = null
    private var cachedEmbeds: LinkedHashMap<String, FloatArray>? = null

    fun load(project: String) = viewModelScope.launch {
        val embeds = repo.readEmbeds(project)
        cachedEmbeds = embeds
        val speakerScript = repo.readSpeakerScript(project)
        val clips = VdubPaths.clipCount(project)

        _state.update {
            it.copy(
                projectName = project,
                modelPresent = SpeakerEmbedder.isModelPresent(),
                modelPath = SpeakerEmbedder.modelFile().absolutePath,
                modelSizeBytes = SpeakerEmbedder.modelFile().length(),
                clipCount = clips,
                embedCount = embeds?.size ?: 0,
                threshold = speakerScript?.threshold ?: it.threshold,
                names = speakerScript?.names.orEmpty(),
                lines = speakerScript?.lines.orEmpty(),
                speakers = buildSpeakers(
                    speakerScript?.lines.orEmpty(),
                    speakerScript?.names.orEmpty()
                ),
                step2Done = VdubPaths.isStepDone(project, 2),
                job = JobState.Idle
            )
        }
    }

    fun setThreshold(value: Float) {
        _state.update { it.copy(threshold = value) }
    }

    fun setUseTargetK(enabled: Boolean) = _state.update { it.copy(useTargetK = enabled) }
    fun setTargetK(k: Int) = _state.update { it.copy(targetK = k.coerceIn(1, 20)) }
    fun dismissMessage() = _state.update { it.copy(message = null) }

    fun cancel() {
        runningJob?.cancel()
        runningJob = null
        _state.update { it.copy(job = JobState.Idle, message = "Cancelled") }
    }

    /** Fetch campplus.onnx straight to the phone — no PC or adb needed. */
    fun downloadModel() = launchJob("Downloading campplus") {
        val spec = ModelDownloader.CAMPPLUS
        val file = modelDownloader.download(spec) { p ->
            when (p) {
                is ModelDownloader.Progress.Downloading -> progress(
                    "Downloading campplus",
                    if (p.total > 0) p.bytes.toFloat() / p.total else -1f,
                    buildString {
                        append(mb(p.bytes))
                        if (p.total > 0) append(" / ").append(mb(p.total))
                        if (p.mirror > 0) append("  (mirror ${p.mirror + 1}/${p.mirrorCount})")
                    }
                )
                is ModelDownloader.Progress.Verifying ->
                    progress("Verifying model", -1f, mb(p.bytes))
            }
        }
        _state.update {
            it.copy(
                modelPresent = true,
                modelPath = file.absolutePath,
                modelSizeBytes = file.length(),
                job = JobState.Done("Model ready", mb(file.length()))
            )
        }
    }

    private fun mb(b: Long) =
        if (b < 1024 * 1024) "%.0f KB".format(b / 1024.0)
        else "%.1f MB".format(b / 1024.0 / 1024.0)

    /** Full run: campplus over every clip, then cluster. */
    fun extractAndCluster() = launchJob("Extracting embeddings") {
        val project = _state.value.projectName
        val embeds = repo.embedSpeakers(project) { done, total ->
            progress(
                "Extracting embeddings",
                done.toFloat() / total.coerceAtLeast(1),
                "$done / $total clips"
            )
        }
        cachedEmbeds = embeds
        _state.update { it.copy(embedCount = embeds.size) }
        applyClustering(reuseCache = true)
    }

    /** Re-cluster from the cache — instant, no model needed. */
    fun recluster() = launchJob("Clustering") {
        applyClustering(reuseCache = true)
    }

    private suspend fun applyClustering(reuseCache: Boolean) {
        val s = _state.value
        progress("Clustering", -1f, "average linkage")
        val script = repo.clusterSpeakers(
            project = s.projectName,
            threshold = s.threshold,
            targetK = if (s.useTargetK) s.targetK else null,
            embeds = if (reuseCache) cachedEmbeds else null
        )
        _state.update {
            it.copy(
                lines = script.lines,
                names = script.names,
                speakers = buildSpeakers(script.lines, script.names),
                step2Done = true,
                job = JobState.Done(
                    "Diarization complete",
                    "${script.speakerCount} speakers · ${script.lines.size} lines"
                )
            )
        }
    }

    fun renameSpeaker(id: String, name: String) = viewModelScope.launch {
        val updated = repo.renameSpeaker(_state.value.projectName, id, name) ?: return@launch
        _state.update {
            it.copy(
                names = updated.names,
                speakers = buildSpeakers(updated.lines, updated.names)
            )
        }
    }

    private fun buildSpeakers(
        lines: List<SpeakerLine>,
        names: Map<String, String>
    ): List<SpeakerInfo> {
        if (lines.isEmpty()) return emptyList()
        val counts = LinkedHashMap<String, Int>()
        lines.forEach { counts[it.spk] = (counts[it.spk] ?: 0) + 1 }
        return counts.entries
            .sortedByDescending { it.value }
            .mapIndexed { i, (id, count) ->
                SpeakerInfo(
                    id = id,
                    displayName = names[id] ?: id,
                    clipCount = count,
                    colorIndex = i
                )
            }
    }

    private fun launchJob(label: String, block: suspend () -> Unit) {
        if (_state.value.busy) return
        runningJob?.cancel()
        runningJob = viewModelScope.launch {
            _state.update { it.copy(job = JobState.Running(label), message = null) }
            runCatching { block() }.onFailure { e ->
                if (e is kotlinx.coroutines.CancellationException) throw e
                _state.update {
                    it.copy(
                        job = JobState.Failed(label, e.message ?: "failed"),
                        message = e.message
                    )
                }
            }
        }
    }

    private fun progress(label: String, value: Float, detail: String) {
        _state.update { it.copy(job = JobState.Running(label, value, detail)) }
    }
}
