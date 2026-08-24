package com.azim.vdub

import com.azim.vdub.audio.WavIo
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import kotlin.math.sin

class WavIoTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val sr = 16_000

    private fun tone(seconds: Double): ByteArray {
        val n = (seconds * sr).toInt()
        val out = ByteArray(n * 2)
        for (i in 0 until n) {
            val v = (20_000 * sin(2 * Math.PI * 440 * i / sr)).toInt().toShort()
            out[i * 2] = (v.toInt() and 0xFF).toByte()
            out[i * 2 + 1] = ((v.toInt() shr 8) and 0xFF).toByte()
        }
        return out
    }

    @Test
    fun `header round trips`() {
        val f = tmp.newFile("a.wav")
        val pcm = tone(1.5)
        WavIo.writePcm16(f, pcm, sr, 1)
        val fmt = WavIo.readFormat(f)
        assertEquals(sr, fmt.sampleRate)
        assertEquals(1, fmt.channels)
        assertEquals(16, fmt.bitsPerSample)
        assertEquals(44L, fmt.dataOffset)
        assertEquals(pcm.size.toLong(), fmt.dataBytes)
        assertEquals(1.5, fmt.durationSec, 1e-9)
    }

    @Test
    fun `slice is byte identical to the source range`() {
        val src = tmp.newFile("src.wav")
        val pcm = tone(10.0)
        WavIo.writePcm16(src, pcm, sr, 1)
        val fmt = WavIo.readFormat(src)

        val start = 2.0
        val end = 4.5
        val dst = tmp.newFile("clip.wav")
        val frames = WavIo.sliceToFile(src, fmt, start, end, dst)

        assertEquals(((end - start) * sr).toInt(), frames)
        val expected = pcm.copyOfRange((start * sr).toInt() * 2, (end * sr).toInt() * 2)
        val actual = dst.readBytes().copyOfRange(WavIo.HEADER_BYTES, dst.length().toInt())
        assertArrayEquals(expected, actual)
    }

    @Test
    fun `slice clamps at both ends`() {
        val src = tmp.newFile("src2.wav")
        WavIo.writePcm16(src, tone(5.0), sr, 1)
        val fmt = WavIo.readFormat(src)

        val head = tmp.newFile("head.wav")
        assertEquals((0.8 * sr).toInt(), WavIo.sliceToFile(src, fmt, -0.5, 0.8, head))

        val tail = tmp.newFile("tail.wav")
        assertEquals((0.5 * sr).toInt(), WavIo.sliceToFile(src, fmt, 4.5, 9.0, tail))

        val empty = tmp.newFile("empty.wav")
        assertEquals(0, WavIo.sliceToFile(src, fmt, 3.0, 3.0, empty))
        assertEquals(0L, WavIo.readFormat(empty).dataBytes)
    }
}
