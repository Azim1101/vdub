package com.azim.vdub.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.audio.VoiceEngine
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.local.VoicePrefs
import com.azim.vdub.data.model.EmotionStyle
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.repo.ProjectRepository
import com.azim.vdub.net.ModelDownloader
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/** One speaker and the clips that will be used to clone them. */
data class SpeakerPlan(
    val id: String,
    val displayName: String,
    val lineCount: Int,
    val referenceSeconds: Double,
    val referenceCount: Int,
    val colorIndex: Int
) {
    /** Zero-shot cloning degrades badly below a few seconds of reference. */
    val referenceWeak: Boolean get() = referenceSeconds < 3.0
}

data class Step5UiState(
    val projectName: String = "vdub_step",
    val engineId: String = ModelCatalog.CHATTERBOX_Q4.id,
    val engineName: String = "",
    val engineInstalled: Boolean = false,
    val engineSizeMb: Int = 0,
    val missingFiles: List<String> = emptyList(),
    val lines: List<SpeakerLine> = emptyList(),
    val speakers: List<SpeakerPlan> = emptyList(),
    val translatedCount: Int = 0,
    val spokenCount: Int = 0,
    val step4Done: Boolean = false,
    val job: JobState = JobState.Idle,
    val message: String? = null
) {
    val busy: Boolean get() = job is JobState.Running
    val total: Int get() = lines.size
    val untranslated: Int get() = (total - translatedCount).coerceAtLeast(0)
    val readyToSpeak: Boolean
        get() = engineInstalled && total > 0 && untranslated == 0 && !busy

    /** Rough wall-clock estimate: about a minute per line on a phone CPU. */
    val estimateMinutes: Int get() = total
}

@HiltViewModel
class Step5ViewModel @Inject constructor(
    private val repo: ProjectRepository,
    private val downloader: ModelDownloader,
    private val voicePrefs: VoicePrefs
) : ViewModel() {

    private val _state = MutableStateFlow(Step5UiState())
    val state: StateFlow<Step5UiState> = _state.asStateFlow()

    private var job: Job? = null

    fun load(project: String) = viewModelScope.launch {
        val engineId = voicePrefs.engineId.first()
        val engine = VoiceEngine.byId(engineId)
        val paths = VoiceEngine.pathsFor(engineId)

        val translated = repo.readTranslatedScript(project)
        val lines = translated?.lines
            ?: repo.readEmotionScript(project)?.lines
            ?: repo.readSpeakerScript(project)?.lines
            ?: emptyList()

        val names = repo.readSpeakerScript(project)?.names.orEmpty()
        val plans = lines.map { it.spk }.distinct().mapIndexed { i, spk ->
            val refs = repo.referenceClipsFor(project, spk)
            val secs = lines.filter { it.spk == spk }
                .sortedByDescending { l -> l.end - l.start }
                .take(refs.size)
                .sumOf { l -> l.end - l.start }
            SpeakerPlan(
                id = spk,
                displayName = names[spk] ?: spk,
                lineCount = lines.count { l -> l.spk == spk },
                referenceSeconds = secs,
                referenceCount = refs.size,
                colorIndex = i
            )
        }

        _state.update {
            it.copy(
                projectName = project,
                engineId = engineId,
                engineName = engine.name,
                engineInstalled = downloader.isInstalled(engine),
                engineSizeMb = engine.sizeMb,
                missingFiles = paths.missing.map { f -> f.name },
                lines = lines,
                speakers = plans,
                translatedCount = lines.count { l -> l.hi.isNotBlank() },
                spokenCount = VdubPaths.hiClipsDir(project)
                    .listFiles { f -> f.extension == "wav" }?.size ?: 0,
                step4Done = VdubPaths.isStepDone(project, 4),
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

    fun downloadEngine() = launchJob("Downloading voice engine") {
        val engine = VoiceEngine.byId(_state.value.engineId)
        downloader.download(engine) { p ->
            progress(
                if (p.verifying) "Verifying" else "Downloading voice engine",
                p.fraction,
                if (p.verifying) p.fileName else buildString {
                    if (p.fileCount > 1) append("${p.fileIndex + 1}/${p.fileCount}  ")
                    append(mb(p.bytes))
                    if (p.total > 0) append(" / ").append(mb(p.total))
                    if (p.mirror > 0) append("  (mirror ${p.mirror + 1})")
                }
            )
        }
        val paths = VoiceEngine.pathsFor(_state.value.engineId)
        _state.update {
            it.copy(
                engineInstalled = downloader.isInstalled(VoiceEngine.byId(it.engineId)),
                missingFiles = paths.missing.map { f -> f.name },
                job = JobState.Done("Voice engine ready", engine.name)
            )
        }
    }

    fun exaggerationFor(emotion: String) = EmotionStyle.exaggeration(emotion)

    /**
     * Synthesis is not implemented yet. Say so rather than starting a job that
     * cannot finish — the user has just downloaded up to 1.5 GB and deserves a
     * straight answer about what does and does not work.
     */
    fun speakNotReady() {
        _state.update {
            it.copy(
                message = "Speech generation is not wired up yet. Everything it " +
                    "needs is ready — engine, reference clips, Hindi text and " +
                    "emotion — but the inference loop is the next piece of work."
            )
        }
    }

    private fun launchJob(label: String, block: suspend () -> Unit) {
        if (_state.value.busy) return
        job?.cancel()
        job = viewModelScope.launch {
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

    private fun mb(b: Long) =
        if (b < 1024 * 1024) "%.0f KB".format(b / 1024.0)
        else "%.0f MB".format(b / 1024.0 / 1024.0)
}
