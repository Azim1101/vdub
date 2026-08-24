package com.azim.vdub

import com.azim.vdub.audio.EmotionClassifier
import com.azim.vdub.data.model.EmotionStyle
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class EmotionTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun `head json parses weight bias and labels`() {
        val f = tmp.newFile("head.json")
        f.writeText(
            """
            {"weight": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
             "bias": [0.01, 0.02],
             "labels": ["生气/angry", "开心/happy"]}
            """.trimIndent()
        )
        val (w, b, labels) = EmotionClassifier.parseHead(f)
        assertEquals(2, w.size)
        assertEquals(3, w[0].size)
        assertEquals(0.6f, w[1][2], 1e-6f)
        assertEquals(0.02f, b[1], 1e-6f)
        assertEquals(listOf("angry", "happy"), labels)
    }

    @Test
    fun `mismatched head is rejected`() {
        val f = tmp.newFile("bad.json")
        f.writeText("""{"weight": [[1.0]], "bias": [0.0, 0.0], "labels": ["a", "b"]}""")
        val e = runCatching { EmotionClassifier.parseHead(f) }.exceptionOrNull()
        assertTrue("expected a mismatch error, got $e", e is IllegalStateException)
    }

    @Test
    fun `labels keep the english side`() {
        assertEquals("angry", EmotionClassifier.cleanLabel("生气/angry"))
        assertEquals("happy", EmotionClassifier.cleanLabel("happy"))
        assertEquals("unk", EmotionClassifier.cleanLabel("<unk>"))
    }

    @Test
    fun `exaggeration matches the pipeline notes`() {
        assertEquals(1.4f, EmotionStyle.exaggeration("angry"), 1e-6f)
        assertEquals(1.1f, EmotionStyle.exaggeration("happy"), 1e-6f)
        assertEquals(1.0f, EmotionStyle.exaggeration("neutral"), 1e-6f)
    }

    @Test
    fun `unknown emotion falls back to neutral strength`() {
        assertEquals(1.0f, EmotionStyle.exaggeration("wibble"), 1e-6f)
        assertEquals(1.0f, EmotionStyle.exaggeration(null), 1e-6f)
    }

    @Test
    fun `normalise maps junk to neutral and is case insensitive`() {
        assertEquals("neutral", EmotionStyle.normalise(""))
        assertEquals("neutral", EmotionStyle.normalise("NOPE"))
        assertEquals("angry", EmotionStyle.normalise("ANGRY"))
        assertEquals("sad", EmotionStyle.normalise("  Sad "))
    }

    @Test
    fun `every known emotion has a strength`() {
        EmotionStyle.KNOWN.forEach {
            assertTrue("$it has no strength", EmotionStyle.exaggeration(it) > 0f)
        }
    }
}
