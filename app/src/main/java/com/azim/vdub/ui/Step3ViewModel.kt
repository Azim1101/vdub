package com.azim.vdub.ui

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.audio.EmotionClassifier
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.DubbingService
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.model.EmotionStyle
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.repo.ProjectRepository
import com.azim.vdub.net.ModelDownloader
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class Step3UiState(
    val projectName: String = "vdub_step",
    val modelPresent: Boolean = false,
    val modelPath: String = "",
    val modelSizeBytes: Long = 0L,
    val clipCount: Int = 0,
    val lines: List<SpeakerLine> = emptyList(),
    val counts: Map<String, Int> = emptyMap(),
    val speakerNames: Map<String, String> = emptyMap(),
    val step3Done: Boolean = false,
    val job: JobState = JobState.Idle,
    val message: String? = null
) {
    val busy: Boolean get() = job is JobState.Running
    val hasResults: Boolean get() = lines.any { it.emotionScore > 0f } || counts.isNotEmpty()
    fun speakerName(id: String) = speakerNames[id] ?: id
}

@HiltViewModel
class Step3ViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val repo: ProjectRepository,
    private val downloader: ModelDownloader
) : ViewModel() {

    private val _state = MutableStateFlow(Step3UiState())
    val state: StateFlow<Step3UiState> = _state.asStateFlow()

    private var job: Job? = null

    fun load(project: String) = viewModelScope.launch {
        val emotion = repo.readEmotionScript(project)
        val speakers = repo.readSpeakerScript(project)
        val model = EmotionClassifier.modelFile()

        _state.update {
            it.copy(
                projectName = project,
                modelPresent = EmotionClassifier.isModelPresent(),
                modelPath = model.absolutePath,
                modelSizeBytes = model.length(),
                clipCount = VdubPaths.clipCount(project),
                lines = emotion?.lines ?: speakers?.lines.orEmpty(),
                counts = emotion?.counts.orEmpty(),
                speakerNames = speakers?.names.orEmpty(),
                step3Done = VdubPaths.isStepDone(project, 3),
                job = JobState.Idle
            )
        }
    }

    fun dismissMessage() = _state.update { it.copy(message = null) }

    fun cancel() {
        job?.cancel()
        job = null
        _state.update { it.copy(job = JobState.Idle, message = "Cancelled") }
    }

    fun downloadModel() = launchJob("Downloading emotion2vec") {
        downloader.download(ModelCatalog.EMOTION2VEC) { p ->
            progress(
                if (p.verifying) "Verifying emotion2vec" else "Downloading emotion2vec",
                p.fraction,
                if (p.verifying) p.fileName else buildString {
                    append(mb(p.bytes))
                    if (p.total > 0) append(" / ").append(mb(p.total))
                    if (p.mirror > 0) append("  (mirror ${p.mirror + 1})")
                }
            )
        }
        val f = EmotionClassifier.modelFile()
        _state.update {
            it.copy(
                modelPresent = EmotionClassifier.isModelPresent(),
                modelPath = f.absolutePath,
                modelSizeBytes = f.length(),
                job = JobState.Done("Model ready", mb(f.length()))
            )
        }
    }

    fun detect() = launchJob("Detecting emotion", foreground = true) {
        val script = repo.detectEmotions(_state.value.projectName) { done, total ->
            progress(
                "Detecting emotion",
                done.toFloat() / total.coerceAtLeast(1),
                "$done / $total clips"
            )
        }
        _state.update {
            it.copy(
                lines = script.lines,
                counts = script.counts,
                step3Done = true,
                job = JobState.Done(
                    "Emotion tagged",
                    script.counts.entries
                        .sortedByDescending { e -> e.value }
                        .joinToString(", ") { e -> "${e.key} ${e.value}" }
                )
            )
        }
    }

    fun setEmotion(utt: String, emotion: String) = viewModelScope.launch {
        val updated = repo.setLineEmotion(_state.value.projectName, utt, emotion) ?: return@launch
        _state.update { it.copy(lines = updated.lines, counts = updated.counts) }
    }

    fun exaggerationFor(emotion: String) = EmotionStyle.exaggeration(emotion)

    private fun launchJob(
        label: String,
        foreground: Boolean = false,
        block: suspend () -> Unit
    ) {
        if (_state.value.busy) return
        job?.cancel()
        job = viewModelScope.launch {
            _state.update { it.copy(job = JobState.Running(label), message = null) }
            if (foreground) DubbingService.start(context, label, "Starting…")
            try {
                runCatching { block() }.onFailure { e ->
                    if (e is kotlinx.coroutines.CancellationException) throw e
                    _state.update {
                        it.copy(
                            job = JobState.Failed(label, e.message ?: "failed"),
                            message = e.message
                        )
                    }
                }
            } finally {
                if (foreground) DubbingService.stop(context)
            }
        }
    }

    private fun progress(label: String, value: Float, detail: String) {
        _state.update { it.copy(job = JobState.Running(label, value, detail)) }
        DubbingService.update(
            context, label, detail,
            if (value >= 0f) (value * 100).toInt() else -1
        )
    }

    private fun mb(b: Long) =
        if (b < 1024 * 1024) "%.0f KB".format(b / 1024.0)
        else "%.0f MB".format(b / 1024.0 / 1024.0)
}
