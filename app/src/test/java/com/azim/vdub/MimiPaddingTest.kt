package com.azim.vdub

import com.azim.vdub.audio.MimiDecoder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Indri → Mimi bridge, which is where this engine can fail silently.
 *
 * Mimi's ONNX decoder takes a fixed 32 codebooks; Indri emits 8. Everything
 * here guards the two things that make feeding one to the other correct:
 * the interleaved token stream must be unpacked into the right grid, and the
 * 24 unused codebooks must be padded with the indices whose embeddings cancel
 * rather than with zeros.
 *
 * Measured on the real weights (`tools/probe_tts5.py`):
 *
 * | padding    | SNR vs the 8-codebook decode upstream performs |
 * |------------|------------------------------------------------|
 * | zeros      | 6.8 dB — audibly wrong                         |
 * | cancelling | 34.4 dB — inaudible                            |
 *
 * A well-meaning simplification to `IntArray(24)` would still run, still
 * produce audio, and quietly sound bad on every line. Hence a test.
 */
class MimiPaddingTest {

    @Test
    fun `padding covers exactly the unused codebooks`() {
        assertEquals(
            MimiDecoder.TOTAL_CODEBOOKS - MimiDecoder.USED_CODEBOOKS,
            MimiDecoder.PADDING_INDICES.size
        )
    }

    @Test
    fun `padding indices are inside the codebook`() {
        MimiDecoder.PADDING_INDICES.forEach {
            assertTrue("index $it out of range", it in 0 until MimiDecoder.CODEBOOK_SIZE)
        }
    }

    /**
     * The whole point. All-zeros is the obvious implementation and the wrong
     * one; if someone replaces the table with zeros this fails instead of the
     * audio quietly degrading.
     */
    @Test
    fun `padding is not all zeros`() {
        assertNotEquals(0, MimiDecoder.PADDING_INDICES.count { it != 0 })
        assertTrue(
            "padding looks like a zero fill",
            MimiDecoder.PADDING_INDICES.count { it != 0 } >= 20
        )
    }

    /**
     * Pins the exact table. These indices are a property of Mimi's weights,
     * derived by minimising ‖Σ codebook_q[i_q]‖ (1.383 → 0.045); they are not
     * arbitrary and must not be "tidied".
     */
    @Test
    fun `padding table matches the values derived from mimi weights`() {
        assertEquals(
            listOf(
                1437, 374, 662, 1190, 1908, 1714, 220, 610,
                32, 1642, 1736, 1402, 692, 1897, 1332, 1774,
                724, 591, 1538, 483, 35, 14, 332, 1833
            ),
            MimiDecoder.PADDING_INDICES.toList()
        )
    }

    // ------------------------------------------------------------- layout

    /**
     * Indri interleaves: token *i* belongs to codebook `i % 8`. Unpacking it
     * as 8 contiguous runs instead would produce audio — of the wrong thing.
     */
    @Test
    fun `layout de-interleaves tokens across codebooks`() {
        // Frame 0: one token per codebook, each carrying its own base offset.
        // Frame 1: the same, plus one, so position is visible in the result.
        val frames = 2
        val tokens = IntArray(frames * MimiDecoder.USED_CODEBOOKS) { i ->
            val cb = i % MimiDecoder.USED_CODEBOOKS
            val frame = i / MimiDecoder.USED_CODEBOOKS
            cb * MimiDecoder.CODEBOOK_SIZE + 100 + frame
        }

        val grid = MimiDecoder.layout(tokens)

        assertEquals(MimiDecoder.TOTAL_CODEBOOKS, grid.size)
        grid.forEach { assertEquals(frames, it.size) }
        for (cb in 0 until MimiDecoder.USED_CODEBOOKS) {
            assertEquals("codebook $cb frame 0", 100, grid[cb][0])
            assertEquals("codebook $cb frame 1", 101, grid[cb][1])
        }
    }

    @Test
    fun `layout fills unused codebooks with the cancelling indices`() {
        val tokens = IntArray(MimiDecoder.USED_CODEBOOKS) { it * MimiDecoder.CODEBOOK_SIZE }
        val grid = MimiDecoder.layout(tokens)

        for (row in MimiDecoder.USED_CODEBOOKS until MimiDecoder.TOTAL_CODEBOOKS) {
            val expected = MimiDecoder.PADDING_INDICES[row - MimiDecoder.USED_CODEBOOKS]
            grid[row].forEach { assertEquals("row $row", expected, it) }
        }
    }

    /** A partial trailing frame is dropped rather than decoded as noise. */
    @Test
    fun `layout ignores a partial trailing frame`() {
        val tokens = IntArray(MimiDecoder.USED_CODEBOOKS * 2 + 3) { it }
        assertEquals(2, MimiDecoder.layout(tokens)[0].size)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `layout rejects fewer tokens than one frame`() {
        MimiDecoder.layout(IntArray(MimiDecoder.USED_CODEBOOKS - 1))
    }

    /**
     * A token outside its codebook's band means the sampling mask failed. It
     * is clamped, because indexing another codebook's entry is worse than
     * clipping to an edge of the right one.
     */
    @Test
    fun `layout clamps a token from outside its band`() {
        val tokens = IntArray(MimiDecoder.USED_CODEBOOKS) { 0 }
        tokens[3] = 99_999
        val grid = MimiDecoder.layout(tokens)
        assertEquals(MimiDecoder.CODEBOOK_SIZE - 1, grid[3][0])

        val negative = IntArray(MimiDecoder.USED_CODEBOOKS) { 0 }
        negative[5] = -40
        assertEquals(0, MimiDecoder.layout(negative)[5][0])
    }

    /** 8 tokens is one 80 ms frame; the rate the timing stage assumes. */
    @Test
    fun `frame rate matches the codec`() {
        assertEquals(12.5, MimiDecoder.FRAME_RATE, 1e-9)
        assertEquals(
            MimiDecoder.SAMPLE_RATE.toDouble() / MimiDecoder.FRAME_RATE,
            1920.0,
            1e-9
        )
    }
}
