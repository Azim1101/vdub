package com.azim.vdub

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Reopening a half-finished project must land on the outstanding step, not
 * make the user press Next through work that is already done.
 *
 * Mirrors VdubPaths.resumeStep, which reads S0x.done markers from disk and so
 * cannot run on the JVM.
 */
class ResumeTest {

    private val lastStep = 6

    private fun lastCompleted(done: Set<Int>): Int =
        (lastStep downTo 1).firstOrNull { done.contains(it) } ?: 0

    private fun resumeStep(done: Set<Int>): Int =
        (lastCompleted(done) + 1).coerceAtMost(lastStep)

    @Test
    fun `a fresh project opens at step one`() {
        assertEquals(1, resumeStep(emptySet()))
    }

    @Test
    fun `after trimming it opens at speakers`() {
        assertEquals(2, resumeStep(setOf(1)))
    }

    @Test
    fun `after translation it opens at voice`() {
        assertEquals(5, resumeStep(setOf(1, 2, 3, 4)))
    }

    /** Highest marker wins, so a re-run of an earlier step cannot send you back. */
    @Test
    fun `gaps do not drag the user backwards`() {
        assertEquals(5, resumeStep(setOf(1, 2, 4)))
        assertEquals(4, resumeStep(setOf(3)))
    }

    /** A finished project reopens on its last screen rather than nowhere. */
    @Test
    fun `a complete project stays on the final step`() {
        assertEquals(6, resumeStep(setOf(1, 2, 3, 4, 5, 6)))
        assertEquals(6, resumeStep(setOf(1, 2, 3, 4, 5)))
    }

    @Test
    fun `advancing never goes below the next step`() {
        // advanceFrom(step) = max(resumeStep, step + 1)
        fun advance(done: Set<Int>, from: Int) =
            maxOf(resumeStep(done), from + 1).coerceAtMost(lastStep)

        // step 4 writes its marker only when every line is translated
        assertEquals(5, advance(setOf(1, 2, 3), from = 4))
        assertEquals(5, advance(setOf(1, 2, 3, 4), from = 4))
    }
}
