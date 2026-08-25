package com.azim.vdub

import com.azim.vdub.audio.DubTimeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class DubTimelineTest {

    private val sr = 24_000

    private fun tone(hz: Double, seconds: Double): FloatArray =
        FloatArray((seconds * sr).toInt()) { i ->
            (0.5 * sin(2 * PI * hz * i / sr)).toFloat()
        }

    /** Zero crossings per second — a cheap proxy for pitch. */
    private fun zcr(x: FloatArray): Double {
        if (x.size < 2) return 0.0
        var c = 0
        for (i in 1 until x.size) if ((x[i - 1] < 0) != (x[i] < 0)) c++
        return c / (x.size.toDouble() / sr)
    }

    @Test
    fun `stretch hits the requested duration`() {
        val src = tone(220.0, 2.0)
        listOf(0.8f, 1.25f, 1.5f, 2.0f).forEach { speed ->
            val out = DubTimeline.timeStretch(src, speed)
            val want = src.size / speed
            assertTrue(
                "speed $speed gave ${out.size}, wanted ~$want",
                abs(out.size - want) / want < 0.05
            )
        }
    }

    /**
     * The whole point of overlap-add rather than resampling: a line that has
     * to be sped up must not turn into a chipmunk.
     */
    @Test
    fun `stretch preserves pitch`() {
        val src = tone(220.0, 2.0)
        val base = zcr(src)
        listOf(1.25f, 1.5f, 2.0f).forEach { speed ->
            val out = DubTimeline.timeStretch(src, speed)
            val mid = out.copyOfRange(out.size / 4, out.size * 3 / 4)
            val drift = abs(zcr(mid) - base) / base
            assertTrue("speed $speed shifted pitch by ${drift * 100}%", drift < 0.15)
        }
    }

    @Test
    fun `speed of one is a passthrough`() {
        val src = tone(300.0, 0.5)
        assertEquals(src.size, DubTimeline.timeStretch(src, 1.0f).size)
    }

    @Test
    fun `clips land at their start time`() {
        val clip = DubTimeline.Clip(0, 1.0, 2.0, tone(200.0, 1.0))
        val r = DubTimeline.assemble(listOf(clip), 5.0, sr)
        assertEquals(5 * sr, r.samples.size)
        // silence before, audio at 1 s
        assertTrue(r.samples.take(sr / 2).all { it == 0f })
        assertTrue(r.samples.copyOfRange(sr + 100, sr + 200).any { it != 0f })
    }

    /** A long line must not stamp over the next one's opening words. */
    @Test
    fun `overlapping clips are pushed apart`() {
        val a = DubTimeline.Clip(0, 0.0, 1.0, tone(200.0, 3.0))   // way too long
        val b = DubTimeline.Clip(1, 1.0, 2.0, tone(400.0, 0.5))
        val r = DubTimeline.assemble(listOf(a, b), 10.0, sr)
        val pa = r.placements.first { it.lineId == 0 }
        val pb = r.placements.first { it.lineId == 1 }
        assertTrue("b starts before a ends", pb.startSec >= pa.startSec)
    }

    @Test
    fun `overflow is reported not hidden`() {
        val long = DubTimeline.Clip(0, 0.0, 0.5, tone(200.0, 4.0))
        val r = DubTimeline.assemble(listOf(long), 10.0, sr)
        val p = r.placements.single()
        assertTrue("should overflow", !p.fitted)
        assertTrue(r.overflowing.isNotEmpty())
    }

    @Test
    fun `speed is clamped to a natural range`() {
        val long = DubTimeline.Clip(0, 0.0, 0.2, tone(200.0, 10.0))
        val p = DubTimeline.assemble(listOf(long), 20.0, sr).placements.single()
        assertTrue(p.speed <= DubTimeline.MAX_SPEED)
        assertTrue(p.speed >= DubTimeline.MIN_SPEED)
    }

    @Test
    fun `mix keeps levels in range`() {
        val a = FloatArray(1000) { 0.9f }
        val b = FloatArray(1000) { 0.9f }
        DubTimeline.mix(a, b).forEach {
            assertTrue("clipped out of range: $it", it in -1f..1f)
        }
    }

    @Test
    fun `empty input is handled`() {
        assertEquals(0, DubTimeline.timeStretch(FloatArray(0), 1.5f).size)
        val r = DubTimeline.assemble(emptyList(), 3.0, sr)
        assertEquals(3 * sr, r.samples.size)
        assertTrue(r.placements.isEmpty())
    }
}
