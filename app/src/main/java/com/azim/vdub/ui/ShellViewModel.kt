package com.azim.vdub.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.local.VoicePrefs
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

data class ShellState(
    /** Null until the resume decision has been made. */
    val screen: Int? = null,
    val project: String = "vdub_step",
    /** Set once when we jumped past Step 1, so the UI can say why. */
    val resumedAtStep: Int? = null
)

/**
 * Decides where the app opens.
 *
 * A half-finished project should reopen on the step that is actually
 * outstanding — pressing Next through four completed screens to get back to
 * work is the kind of friction that makes a long pipeline unusable.
 *
 * The decision is made from the S0x.done markers on disk rather than any
 * in-memory state, so it survives the process being killed mid-run.
 */
@HiltViewModel
class ShellViewModel @Inject constructor(
    private val prefs: VoicePrefs
) : ViewModel() {

    private val _state = MutableStateFlow(ShellState())
    val state: StateFlow<ShellState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            val last = prefs.lastProject.first()
            val resolved = withContext(Dispatchers.IO) {
                VdubPaths.ensureRoots()
                val name = last?.takeIf { VdubPaths.projectDir(it).exists() }
                    ?: VdubPaths.listProjects().firstOrNull()
                if (name == null) {
                    // Nothing on disk: start clean at Step 1.
                    Triple("vdub_step", 1, null)
                } else {
                    val step = VdubPaths.resumeStep(name)
                    Triple(name, step, if (step > 1) step else null)
                }
            }
            _state.update {
                it.copy(
                    project = resolved.first,
                    screen = resolved.second,
                    resumedAtStep = resolved.third
                )
            }
        }
    }

    fun goTo(screen: Int) = _state.update { it.copy(screen = screen, resumedAtStep = null) }

    fun setProject(name: String) {
        _state.update { it.copy(project = name) }
        viewModelScope.launch { prefs.setLastProject(name) }
    }

    /** Re-check markers after a step finishes, so Next lands correctly. */
    fun advanceFrom(step: Int) {
        val project = _state.value.project
        viewModelScope.launch {
            val next = withContext(Dispatchers.IO) {
                // Prefer the marker-derived answer; fall back to step + 1 when
                // the step does not write a marker until later.
                val resume = VdubPaths.resumeStep(project)
                maxOf(resume, step + 1).coerceAtMost(VdubPaths.LAST_STEP)
            }
            _state.update { it.copy(screen = next, resumedAtStep = null) }
        }
    }

    fun dismissResumeNotice() = _state.update { it.copy(resumedAtStep = null) }
}
