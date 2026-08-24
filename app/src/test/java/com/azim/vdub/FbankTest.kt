package com.azim.vdub

import com.azim.vdub.audio.Fbank
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class FbankTest {

    private fun tone(hz: Double, seconds: Double, amp: Float = 16000f): FloatArray {
        val n = (Fbank.SAMPLE_RATE * seconds).toInt()
        return FloatArray(n) { i ->
            (amp * sin(2 * PI * hz * i / Fbank.SAMPLE_RATE)).toFloat()
        }
    }

    @Test
    fun `frame count matches kaldi snip_edges`() {
        // kaldi: 1s at 16k with 25ms/10ms -> 98 frames
        assertEquals(98, Fbank.numFrames(16_000))
        assertEquals(0, Fbank.numFrames(100))
        assertEquals(1, Fbank.numFrames(Fbank.frameLength))
    }

    @Test
    fun `produces 80 bins per frame`() {
        val feats = Fbank.compute(tone(440.0, 0.5))
        assertTrue(feats.isNotEmpty())
        feats.forEach { assertEquals(Fbank.NUM_BINS, it.size) }
        assertEquals(Fbank.numFrames((0.5 * Fbank.SAMPLE_RATE).toInt()), feats.size)
    }

    @Test
    fun `cmn zeroes the per-bin mean`() {
        val feats = Fbank.compute(tone(440.0, 0.5), applyCmn = true)
        for (b in 0 until Fbank.NUM_BINS) {
            val mean = feats.sumOf { it[b].toDouble() } / feats.size
            assertTrue("bin $b mean $mean", abs(mean) < 1e-3)
        }
    }

    @Test
    fun `energy lands in the expected mel bin`() {
        // Higher tone must peak in a higher bin than a lower tone.
        val low = Fbank.compute(tone(300.0, 0.4), applyCmn = false)
        val high = Fbank.compute(tone(3000.0, 0.4), applyCmn = false)
        val lowPeak = low[10].indices.maxByOrNull { low[10][it] }!!
        val highPeak = high[10].indices.maxByOrNull { high[10][it] }!!
        assertTrue("low=$lowPeak high=$highPeak", highPeak > lowPeak)
        assertTrue("300Hz should sit low", lowPeak < 25)
        assertTrue("3kHz should sit high", highPeak > 40)
    }

    @Test
    fun `silence does not produce NaN`() {
        val feats = Fbank.compute(FloatArray(8000), applyCmn = true)
        feats.forEach { row ->
            row.forEach { assertTrue("NaN/Inf in fbank", it.isFinite()) }
        }
    }

    @Test
    fun `too short input yields no frames`() {
        assertEquals(0, Fbank.compute(FloatArray(100)).size)
    }
}
