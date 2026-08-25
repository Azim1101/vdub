package com.azim.vdub.ui

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.model.TranslationSource
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

data class Step4UiState(
    val projectName: String = "vdub_step",
    val lines: List<SpeakerLine> = emptyList(),
    val source: TranslationSource = TranslationSource.NONE,
    val translatedCount: Int = 0,
    val nllbInstalled: Boolean = false,
    val speakerNames: Map<String, String> = emptyMap(),
    val exportedPath: String? = null,
    val step4Done: Boolean = false,
    val job: JobState = JobState.Idle,
    val message: String? = null
) {
    val busy: Boolean get() = job is JobState.Running
    val total: Int get() = lines.size
    val missing: Int get() = (total - translatedCount).coerceAtLeast(0)
    val complete: Boolean get() = total > 0 && missing == 0
    val progress: Float get() = if (total == 0) 0f else translatedCount.toFloat() / total
    fun speakerName(id: String) = speakerNames[id] ?: id
}

@HiltViewModel
class Step4ViewModel @Inject constructor(
    private val repo: ProjectRepository,
    private val downloader: ModelDownloader
) : ViewModel() {

    private val _state = MutableStateFlow(Step4UiState())
    val state: StateFlow<Step4UiState> = _state.asStateFlow()

    private var job: Job? = null

    fun load(project: String) = viewModelScope.launch {
        val translated = repo.readTranslatedScript(project)
        val fallback = translated?.lines
            ?: repo.readEmotionScript(project)?.lines
            ?: repo.readSpeakerScript(project)?.lines
            ?: emptyList()

        _state.update {
            it.copy(
                projectName = project,
                lines = fallback,
                source = runCatching {
                    TranslationSource.valueOf(translated?.source ?: "NONE")
                }.getOrDefault(TranslationSource.NONE),
                translatedCount = fallback.count { l -> l.hi.isNotBlank() },
                nllbInstalled = downloader.isInstalled(ModelCatalog.NLLB),
                speakerNames = repo.readSpeakerScript(project)?.names.orEmpty(),
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

    /** Upload a finished translation and skip machine translation entirely. */
    fun importSrt(uri: Uri) = launchJob("Reading translated SRT") {
        val script = repo.importTranslationSrt(_state.value.projectName, uri)
        applyResult(script.lines, TranslationSource.UPLOADED_SRT, "SRT")
    }

    fun importJson(uri: Uri) = launchJob("Reading translated JSON") {
        val script = repo.importTranslationJson(_state.value.projectName, uri)
        applyResult(script.lines, TranslationSource.UPLOADED_JSON, "JSON")
    }

    private fun applyResult(
        lines: List<SpeakerLine>,
        src: TranslationSource,
        what: String
    ) {
        val done = lines.count { it.hi.isNotBlank() }
        val missing = lines.size - done
        _state.update {
            it.copy(
                lines = lines,
                source = src,
                translatedCount = done,
                step4Done = missing == 0 && done > 0,
                job = JobState.Done(
                    "Translation loaded",
                    if (missing == 0) "$what · all $done lines"
                    else "$what · $done of ${lines.size} — $missing still empty"
                ),
                message = if (missing > 0) {
                    "$missing lines had no matching subtitle. Fill them in " +
                        "below, or run auto-translate for the rest."
                } else null
            )
        }
    }

    fun exportJson() = launchJob("Exporting JSON") {
        val f = repo.exportTranslationJson(_state.value.projectName)
        _state.update {
            it.copy(exportedPath = f.absolutePath, job = JobState.Done("Exported", f.name))
        }
    }

    fun exportSrt() = launchJob("Exporting SRT") {
        val f = repo.exportTranslationSrt(_state.value.projectName)
        _state.update {
            it.copy(exportedPath = f.absolutePath, job = JobState.Done("Exported", f.name))
        }
    }

    fun setLine(utt: String, hi: String) = viewModelScope.launch {
        val updated = repo.setLineTranslation(_state.value.projectName, utt, hi) ?: return@launch
        _state.update {
            it.copy(
                lines = updated.lines,
                translatedCount = updated.translatedCount,
                source = TranslationSource.MANUAL_EDIT,
                step4Done = updated.translatedCount == updated.lines.size
            )
        }
    }

    fun downloadNllb() = launchJob("Downloading NLLB") {
        downloader.download(ModelCatalog.NLLB) { p ->
            progress(
                if (p.verifying) "Verifying NLLB" else "Downloading NLLB",
                p.fraction,
                if (p.verifying) p.fileName else buildString {
                    append(mb(p.bytes))
                    if (p.total > 0) append(" / ").append(mb(p.total))
                    if (p.mirror > 0) append("  (mirror ${p.mirror + 1})")
                }
            )
        }
        _state.update {
            it.copy(
                nllbInstalled = downloader.isInstalled(ModelCatalog.NLLB),
                job = JobState.Done("NLLB ready", "")
            )
        }
    }

    fun autoTranslateNotReady() {
        _state.update {
            it.copy(
                message = "On-device NLLB translation is not wired up yet. " +
                    "Upload a translated SRT or JSON to continue."
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
