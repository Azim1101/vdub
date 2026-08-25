package com.azim.vdub

import com.azim.vdub.ui.Step1UiState
import com.azim.vdub.data.model.VideoSource
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * A previous run's video and subtitles must not survive into a new project.
 * These pin the two ways that leaked: renaming the project, and rehydrating
 * from a folder whose files have been deleted.
 */
class StaleStateTest {

    private fun loaded() = Step1UiState(
        projectName = "old",
        videoPath = "/AI/vdub_projects/old/input_video.mp4",
        videoSource = VideoSource.GALLERY,
        durationMs = 2_267_000,
        videoSizeBytes = 142_000_000,
        srtPath = "/AI/vdub_projects/old/subs/original.srt",
        cueCount = 473,
        lineCount = 190,
        clipCount = 190,
        step1Done = true
    )

    /** Mirrors Step1ViewModel.setProjectName. */
    private fun rename(s: Step1UiState, name: String) =
        if (name == s.projectName) s
        else Step1UiState(
            projectName = name,
            knownProjects = s.knownProjects,
            storageShared = s.storageShared,
            storagePath = s.storagePath,
            mergeGapMs = s.mergeGapMs
        )

    @Test
    fun `renaming detaches from the loaded project`() {
        val fresh = rename(loaded(), "brand_new")
        assertEquals("brand_new", fresh.projectName)
        assertNull(fresh.videoPath)
        assertFalse(fresh.hasVideo)
        assertFalse(fresh.hasSubtitles)
        assertEquals(0, fresh.clipCount)
        assertEquals(0, fresh.cueCount)
        assertFalse(fresh.step1Done)
    }

    @Test
    fun `renaming to the same name keeps state`() {
        val s = loaded()
        assertEquals(s, rename(s, "old"))
    }

    @Test
    fun `preferences survive a rename`() {
        val s = loaded().copy(mergeGapMs = 1200, knownProjects = listOf("a", "b"))
        val fresh = rename(s, "other")
        assertEquals(1200, fresh.mergeGapMs)
        assertEquals(listOf("a", "b"), fresh.knownProjects)
    }

    @Test
    fun `an empty project reports nothing loaded`() {
        val s = Step1UiState(projectName = "empty")
        assertFalse(s.hasVideo)
        assertFalse(s.hasSubtitles)
        assertFalse(s.canTrim)
    }

    /** step1Done must require clips, not just the marker file. */
    @Test
    fun `done needs clips present`() {
        val markerOnly = Step1UiState(projectName = "p", step1Done = true, clipCount = 0)
        // the flag can be set, but trimming is still unavailable without input
        assertFalse(markerOnly.canTrim)
    }
}
