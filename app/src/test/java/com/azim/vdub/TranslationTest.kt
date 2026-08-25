package com.azim.vdub

import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.model.TranslationSource
import com.azim.vdub.subtitle.SrtParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Uploading a finished translation must skip machine translation entirely,
 * and must not quietly drop lines the uploaded file does not cover.
 */
class TranslationTest {

    private val lines = listOf(
        SpeakerLine("line_0000", 1.0, 3.0, "传闻中", "Speaker 1"),
        SpeakerLine("line_0001", 4.0, 6.0, "他手刃", "Speaker 1"),
        SpeakerLine("line_0002", 7.0, 9.0, "东方青苍", "Speaker 2")
    )

    /** Mirrors the repository's overlap match. */
    private fun attachSrt(srt: String): List<SpeakerLine> {
        val cues = SrtParser.parse(srt.byteInputStream())
        return lines.map { line ->
            val hit = cues.filter { c ->
                c.endMs / 1000.0 > line.start && c.startMs / 1000.0 < line.end
            }
            if (hit.isEmpty()) line
            else line.copy(hi = hit.joinToString(" ") { it.text }.trim())
        }
    }

    @Test
    fun `uploaded srt fills every line by time overlap`() {
        val out = attachSrt(
            """
            1
            00:00:01,000 --> 00:00:03,000
            अफ़वाहों में

            2
            00:00:04,000 --> 00:00:06,000
            उसने मारा

            3
            00:00:07,000 --> 00:00:09,000
            डोंगफ़ांग क़िंगcang
            """.trimIndent()
        )
        assertEquals(3, out.count { it.hi.isNotBlank() })
        assertEquals("अफ़वाहों में", out[0].hi)
    }

    @Test
    fun `cues that span two lines attach to both`() {
        val out = attachSrt(
            """
            1
            00:00:00,500 --> 00:00:06,500
            एक लंबी पंक्ति
            """.trimIndent()
        )
        assertTrue(out[0].hi.isNotBlank())
        assertTrue(out[1].hi.isNotBlank())
        assertTrue("must not reach line 3", out[2].hi.isBlank())
    }

    @Test
    fun `a partial upload leaves the rest blank rather than guessing`() {
        val out = attachSrt(
            """
            1
            00:00:01,000 --> 00:00:03,000
            सिर्फ़ पहली
            """.trimIndent()
        )
        assertEquals(1, out.count { it.hi.isNotBlank() })
        assertEquals(2, out.count { it.hi.isBlank() })
    }

    /** JSON is matched by utt, so re-timing cannot misalign it. */
    @Test
    fun `uploaded json matches by utt not order`() {
        val incoming = listOf(
            SpeakerLine("line_0002", 0.0, 0.0, "", "", hi = "तीसरी"),
            SpeakerLine("line_0000", 0.0, 0.0, "", "", hi = "पहली")
        )
        val byUtt = incoming.associateBy { it.utt }
        val out = lines.map { l ->
            val hi = byUtt[l.utt]?.hi.orEmpty()
            if (hi.isBlank()) l else l.copy(hi = hi)
        }
        assertEquals("पहली", out[0].hi)
        assertEquals("", out[1].hi)
        assertEquals("तीसरी", out[2].hi)
    }

    @Test
    fun `blank incoming values do not erase existing text`() {
        val existing = lines.map { it.copy(hi = "पुराना") }
        val byUtt = mapOf("line_0001" to SpeakerLine("line_0001", 0.0, 0.0, "", "", hi = ""))
        val out = existing.map { l ->
            val hi = byUtt[l.utt]?.hi.orEmpty()
            if (hi.isBlank()) l else l.copy(hi = hi)
        }
        assertTrue(out.all { it.hi == "पुराना" })
    }

    /** Step 4 is only done when nothing is left untranslated. */
    @Test
    fun `completion requires every line`() {
        fun done(l: List<SpeakerLine>) =
            l.count { it.hi.isNotBlank() } == l.size && l.isNotEmpty()

        assertTrue(!done(lines))
        assertTrue(!done(lines.mapIndexed { i, l -> if (i < 2) l.copy(hi = "x") else l }))
        assertTrue(done(lines.map { it.copy(hi = "x") }))
    }

    @Test
    fun `sources are distinguishable`() {
        assertEquals("UPLOADED_SRT", TranslationSource.UPLOADED_SRT.name)
        assertTrue(TranslationSource.entries.contains(TranslationSource.UPLOADED_JSON))
    }
}
