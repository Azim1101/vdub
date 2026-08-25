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

    @Test
    fun `q4 pack uses the plain language model name`() {
        val p = VoiceEngine.pathsFor("chatterbox_q4")
        assertEquals("language_model.onnx", p.languageModel.name)
        assertEquals("embed_tokens.onnx", p.embedTokens.name)
        assertEquals("speech_encoder.onnx", p.speechEncoder.name)
        assertEquals("conditional_decoder.onnx", p.conditionalDecoder.name)
    }

    /** The mix graph references its sidecar by this exact name. */
    @Test
    fun `mix pack keeps the q4 language model name`() {
        val p = VoiceEngine.pathsFor("chatterbox_mix")
        assertEquals("language_model_q4.onnx", p.languageModel.name)
        // the other three keep the shared names
        assertEquals("embed_tokens.onnx", p.embedTokens.name)
        assertEquals("speech_encoder.onnx", p.speechEncoder.name)
        assertEquals("conditional_decoder.onnx", p.conditionalDecoder.name)
    }

    @Test
    fun `packs resolve into different folders`() {
        val a = VoiceEngine.pathsFor("chatterbox_q4")
        val b = VoiceEngine.pathsFor("chatterbox_mix")
        assertNotEquals(a.speechEncoder.parent, b.speechEncoder.parent)
        assertNotEquals(a.speechEncoder.absolutePath, b.speechEncoder.absolutePath)
    }

    @Test
    fun `every resolved graph is declared in the catalog`() {
        ModelCatalog.VOICE_ENGINES.forEach { engine ->
            val declared = engine.files.map { it.localName }.toSet()
            val p = VoiceEngine.pathsFor(engine.id)
            (p.graphs + p.tokenizer).forEach { f ->
                val rel = "${f.parentFile.name}/${f.name}"
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

    @Test
    fun `nothing is installed on a bare filesystem`() {
        // no files exist under the test's models dir
        assertTrue(VoiceEngine.pathsFor("chatterbox_q4").missing.isNotEmpty())
    }
}
