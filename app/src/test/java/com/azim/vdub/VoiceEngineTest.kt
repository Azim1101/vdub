package com.azim.vdub

import com.azim.vdub.audio.VoiceEngine
import com.azim.vdub.core.ModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The two packs share one inference path. Everything that differs between
 * them must be resolved here, not branched on at the call site.
 */
class VoiceEngineTest {

    private fun leaf(s: String) = s.substringAfterLast('/')

    @Test
    fun `q4 pack uses the plain language model name`() {
        val n = VoiceEngine.namesFor("chatterbox_q4")
        assertEquals("language_model.onnx", leaf(n.languageModel))
        assertEquals("embed_tokens.onnx", leaf(n.embedTokens))
        assertEquals("speech_encoder.onnx", leaf(n.speechEncoder))
        assertEquals("conditional_decoder.onnx", leaf(n.conditionalDecoder))
    }

    /** The mix graph references its sidecar by this exact name. */
    @Test
    fun `mix pack keeps the q4 language model name`() {
        val n = VoiceEngine.namesFor("chatterbox_mix")
        assertEquals("language_model_q4.onnx", leaf(n.languageModel))
        // the other three keep the shared names
        assertEquals("embed_tokens.onnx", leaf(n.embedTokens))
        assertEquals("speech_encoder.onnx", leaf(n.speechEncoder))
        assertEquals("conditional_decoder.onnx", leaf(n.conditionalDecoder))
    }

    @Test
    fun `packs resolve into different folders`() {
        val a = VoiceEngine.namesFor("chatterbox_q4")
        val b = VoiceEngine.namesFor("chatterbox_mix")
        assertNotEquals(
            a.speechEncoder.substringBeforeLast('/'),
            b.speechEncoder.substringBeforeLast('/')
        )
        assertNotEquals(a.speechEncoder, b.speechEncoder)
    }

    @Test
    fun `every resolved graph is declared in the catalog`() {
        ModelCatalog.VOICE_ENGINES.forEach { engine ->
            val declared = engine.files.map { it.localName }.toSet()
            val n = VoiceEngine.namesFor(engine.id)
            (n.graphs + n.tokenizer).forEach { rel ->
                assertTrue("${engine.id}: $rel not in catalog", declared.contains(rel))
            }
        }
    }

    /** An unknown id must not crash the voice stage. */
    @Test
    fun `unknown id falls back to the first engine`() {
        assertEquals(
            ModelCatalog.VOICE_ENGINES.first().id,
            VoiceEngine.byId("nope").id
        )
    }

    /** 2.0 makes the quantized LLM loop forever; both packs need 1.2. */
    @Test
    fun `repetition penalty avoids the loop`() {
        assertEquals(1.2f, VoiceEngine.REPETITION_PENALTY, 1e-6f)
        assertTrue(VoiceEngine.REPETITION_PENALTY < 2.0f)
    }

    /** Each pack resolves a full set of graphs, none left as a bare guess. */
    @Test
    fun `every engine resolves a complete graph set`() {
        ModelCatalog.VOICE_ENGINES.forEach { engine ->
            val n = VoiceEngine.namesFor(engine.id)
            assertEquals(4, n.graphs.size)
            assertTrue(n.graphs.all { it.contains('/') })
            assertTrue(n.tokenizer.endsWith("tokenizer.json"))
        }
    }
}
