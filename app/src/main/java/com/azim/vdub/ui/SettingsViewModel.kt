package com.azim.vdub.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.net.ModelDownloader
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ModelRow(
    val model: ModelCatalog.Model,
    val installed: Boolean,
    val installedBytes: Long,
    val downloading: Boolean = false,
    val progress: Float = -1f,
    val detail: String = "",
    val error: String? = null
)

data class SettingsUiState(
    val rows: List<ModelRow> = emptyList(),
    val storageShared: Boolean = true,
    val modelsPath: String = "",
    val freeBytes: Long = 0L,
    val busyId: String? = null
) {
    val installedCount: Int get() = rows.count { it.installed }
    val totalInstalledBytes: Long get() = rows.sumOf { it.installedBytes }
    val requiredMissing: List<ModelCatalog.Model>
        get() = rows.filter { it.model.required && !it.installed }.map { it.model }
}

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val downloader: ModelDownloader
) : ViewModel() {

    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    private var job: Job? = null

    init { refresh() }

    fun refresh() {
        val rows = ModelCatalog.ALL.map { m ->
            ModelRow(
                model = m,
                installed = downloader.isInstalled(m),
                installedBytes = downloader.installedBytes(m)
            )
        }
        _state.update {
            it.copy(
                rows = rows,
                storageShared = VdubPaths.usingSharedStorage,
                modelsPath = VdubPaths.modelsDir.absolutePath,
                freeBytes = runCatching { VdubPaths.modelsDir.usableSpace }.getOrDefault(0L)
            )
        }
    }

    fun download(model: ModelCatalog.Model) {
        if (_state.value.busyId != null) return
        job?.cancel()
        _state.update { it.copy(busyId = model.id) }
        patch(model.id) { it.copy(downloading = true, progress = -1f, error = null) }

        job = viewModelScope.launch {
            runCatching {
                downloader.download(model) { p ->
                    patch(model.id) { row ->
                        row.copy(
                            progress = p.fraction,
                            detail = if (p.verifying) {
                                "Verifying ${p.fileName}"
                            } else {
                                buildString {
                                    if (p.fileCount > 1) {
                                        append("${p.fileIndex + 1}/${p.fileCount}  ")
                                    }
                                    append(mb(p.bytes))
                                    if (p.total > 0) append(" / ").append(mb(p.total))
                                    if (p.mirror > 0) append("  (mirror ${p.mirror + 1})")
                                }
                            }
                        )
                    }
                }
            }.onFailure { e ->
                if (e !is kotlinx.coroutines.CancellationException) {
                    patch(model.id) { it.copy(error = e.message ?: "download failed") }
                }
            }
            patch(model.id) { it.copy(downloading = false, detail = "") }
            _state.update { it.copy(busyId = null) }
            refresh()
        }
    }

    fun cancel() {
        job?.cancel()
        job = null
        _state.update { s ->
            s.copy(
                busyId = null,
                rows = s.rows.map { it.copy(downloading = false, detail = "") }
            )
        }
    }

    fun delete(model: ModelCatalog.Model) {
        if (_state.value.busyId != null) return
        downloader.delete(model)
        refresh()
    }

    private fun patch(id: String, block: (ModelRow) -> ModelRow) {
        _state.update { s ->
            s.copy(rows = s.rows.map { if (it.model.id == id) block(it) else it })
        }
    }

    private fun mb(b: Long) =
        if (b < 1024 * 1024) "%.0f KB".format(b / 1024.0)
        else "%.0f MB".format(b / 1024.0 / 1024.0)
}
