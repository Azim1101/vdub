package com.azim.vdub

import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.ui.SpeakerPlan
import com.azim.vdub.ui.Step5UiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Step 5 must not offer to speak until everything it needs is genuinely
 * present — and must never quietly substitute the other voice engine.
 */
class Step5StateTest {

    private fun line(i: Int, hi: String = "नमस्ते", spk: String = "Speaker 1") =
        SpeakerLine("line_%04d".format(i), i.toDouble(), i + 2.0, "原文", spk, hi = hi)

    private fun state(
        lines: List<SpeakerLine>,
        installed: Boolean = true
    ) = Step5UiState(
        lines = lines,
        engineInstalled = installed,
        translatedCount = lines.count { it.hi.isNotBlank() }
    )

    @Test
    fun `ready only when engine present and everything translated`() {
        assertTrue(state((0..2).map { line(it) }).readyToSpeak)
    }

    @Test
    fun `not ready without the engine`() {
        assertFalse(state((0..2).map { line(it) }, installed = false).readyToSpeak)
    }

    @Test
    fun `not ready with untranslated lines`() {
        val lines = listOf(line(0), line(1, hi = ""), line(2))
        val s = state(lines)
        assertEquals(1, s.untranslated)
        assertFalse(s.readyToSpeak)
    }

    @Test
    fun `not ready with no lines at all`() {
        assertFalse(state(emptyList()).readyToSpeak)
    }

    /** A short reference produces a poor clone; flag it rather than hide it. */
    @Test
    fun `short reference audio is flagged`() {
        val weak = SpeakerPlan(
            "Speaker 2", "Chang Heng", 4, 1.8, 1, listOf("line_0007"), "नमस्ते", 1
        )
        val ok = SpeakerPlan(
            "Speaker 1", "Xiao Lanhua", 120, 12.4, 3,
            listOf("line_0001", "line_0004", "line_0009"), "जल्दी करो", 0
        )
        assertTrue(weak.referenceWeak)
        assertFalse(ok.referenceWeak)
    }

    @Test
    fun `estimate scales with line count`() {
        assertEquals(190, state((0 until 190).map { line(it) }).estimateMinutes)
    }
}
