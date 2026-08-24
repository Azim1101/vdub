package com.azim.vdub

import com.azim.vdub.subtitle.SrtParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SrtParserTest {

    private val sample = """
        1
        00:00:01,200 --> 00:00:02,400
        I told you already,

        2
        00:00:02,550 --> 00:00:04,100
        this is not what we agreed on.

        3
        00:00:07,000 --> 00:00:09,250
        <i>The rain stopped an hour ago.</i>

        4
        00:00:09,400 --> 00:00:11,000
        [NARRATOR] We can still fix it
    """.trimIndent()

    @Test
    fun `parses cues and strips tags`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        assertEquals(4, cues.size)
        assertEquals(1200L, cues[0].startMs)
        assertEquals(2400L, cues[0].endMs)
        assertEquals("The rain stopped an hour ago.", cues[2].text)
        assertEquals("We can still fix it", cues[3].text)
    }

    @Test
    fun `merges continuation cues but splits on sentence end`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        val lines = SrtParser.mergeToLines(cues, maxGapMs = 400)
        assertEquals(3, lines.size)
        assertEquals("I told you already, this is not what we agreed on.", lines[0].text)
        assertEquals(1.2, lines[0].start, 1e-6)
        assertEquals(4.1, lines[0].end, 1e-6)
    }

    @Test
    fun `merge never loses text`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        val lines = SrtParser.mergeToLines(cues)
        val a = cues.joinToString("") { it.text }.filterNot { it.isWhitespace() }
        val b = lines.joinToString("") { it.text }.filterNot { it.isWhitespace() }
        assertEquals(a, b)
    }

    @Test
    fun `ids are sequential and clip paths line up`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        val lines = SrtParser.mergeToLines(cues)
        lines.forEachIndexed { i, l ->
            assertEquals(i, l.id)
            assertEquals("clips/line_%04d.wav".format(i), l.clip)
            assertTrue(l.end > l.start)
        }
    }

    @Test
    fun `larger gap folds more cues together`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        val tight = SrtParser.mergeToLines(cues, maxGapMs = 50)
        val loose = SrtParser.mergeToLines(cues, maxGapMs = 400)
        assertTrue(loose.size <= tight.size)
    }

    @Test
    fun `round trips through srt`() {
        val cues = SrtParser.parse(sample.byteInputStream())
        val lines = SrtParser.mergeToLines(cues)
        val srt = SrtParser.toSrt(lines)
        val reparsed = SrtParser.parse(srt.byteInputStream())
        assertEquals(lines.size, reparsed.size)
        assertEquals(lines[0].text, reparsed[0].text)
        assertEquals((lines[0].start * 1000).toLong(), reparsed[0].startMs)
    }
}
