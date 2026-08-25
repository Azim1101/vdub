package com.azim.vdub

import com.azim.vdub.core.ModelCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {

    @Test
    fun `every file has https mirrors and a distinct local name`() {
        val names = mutableSetOf<String>()
        ModelCatalog.ALL.forEach { m ->
            assertTrue("${m.id} has no files", m.files.isNotEmpty())
            m.files.forEach { f ->
                assertTrue(
                    "${m.id}/${f.localName} needs >= 2 mirrors",
                    f.urls.size >= 2
                )
                f.urls.forEach { u ->
                    assertTrue("not https: $u", u.startsWith("https://"))
                }
                assertTrue(
                    "duplicate local name ${f.localName}",
                    names.add(f.localName)
                )
            }
        }
    }

    @Test
    fun `model ids are unique`() {
        val ids = ModelCatalog.ALL.map { it.id }
        assertEquals(ids.size, ids.distinct().size)
    }

    @Test
    fun `onnx files are declared as onnx and json as json`() {
        ModelCatalog.ALL.flatMap { it.files }.forEach { f ->
            when {
                f.localName.endsWith(".onnx") ->
                    assertEquals(f.localName, ModelCatalog.Kind.ONNX, f.kind)
                f.localName.endsWith(".json") ->
                    assertEquals(f.localName, ModelCatalog.Kind.JSON, f.kind)
            }
        }
    }

    /**
     * The whole reason no server is needed: stages run one at a time, so the
     * memory ceiling is the largest model, not the sum.
     */
    @Test
    fun `peak ram is one stage, not the sum`() {
        assertEquals(ModelCatalog.ALL.maxOf { it.ramBytes }, ModelCatalog.peakRamBytes)
        assertTrue(ModelCatalog.peakRamBytes < ModelCatalog.totalBytes)
    }

    /** Steps 1-4 must stay comfortable on a 6 GB phone. */
    @Test
    fun `runnable stages stay under 1gb`() {
        assertTrue(
            "peak ${ModelCatalog.peakRunnableRamBytes / 1024 / 1024} MB too big",
            ModelCatalog.peakRunnableRamBytes < 1_000_000_000L
        )
    }

    /**
     * A quantized model that dequantizes on the fly needs more RAM than it
     * occupies on disk, so the catalog must not equate the two.
     */
    @Test
    fun `chatterbox declares more ram than its file size`() {
        val cb = ModelCatalog.byId("chatterbox_hi")!!
        assertTrue(cb.ramBytes > cb.sizeBytes)
        assertTrue(cb.ramBytes > 1_500_000_000L)
    }

    @Test
    fun `campplus is required and speaker stage`() {
        val m = ModelCatalog.byId("campplus")!!
        assertTrue(m.required)
        assertEquals(ModelCatalog.Stage.DIARIZATION, m.stage)
        assertEquals("campplus.onnx", m.files.first().localName)
    }

    @Test
    fun `asr is optional since an SRT can be supplied instead`() {
        assertTrue(!ModelCatalog.byId("sensevoice")!!.required)
    }

    @Test
    fun `every stage has at least one model`() {
        ModelCatalog.Stage.entries.forEach { stage ->
            assertTrue("no model for $stage", ModelCatalog.forStage(stage).isNotEmpty())
        }
    }

    @Test
    fun `tts is the project chatterbox model, not a substitute`() {
        val tts = ModelCatalog.forStage(ModelCatalog.Stage.TTS)
        assertEquals(1, tts.size)
        val m = tts.first()
        assertEquals("chatterbox_hi", m.id)
        // must come from the project's own repo
        assertTrue(
            m.files.all { f -> f.urls.any { it.contains("vdub-hindi-dubbing-lite") } }
        )
        // voice cloning is the whole point
        assertTrue(m.description.contains("Clones", ignoreCase = true))
    }

    /**
     * Chatterbox ships safetensors, which ONNX Runtime cannot load. The flag
     * must say so, otherwise Settings would present it as ready and Step 5
     * would fail at run time instead.
     */
    @Test
    fun `safetensors models are marked not runnable`() {
        val cb = ModelCatalog.byId("chatterbox_hi")!!
        assertEquals(ModelCatalog.Runtime.SAFETENSORS, cb.runtime)
        assertTrue(!cb.runnable)

        listOf("campplus", "emotion2vec", "nllb", "sensevoice").forEach {
            assertTrue("$it should be runnable", ModelCatalog.byId(it)!!.runnable)
        }
    }

    @Test
    fun `sizes are plausible`() {
        ModelCatalog.ALL.forEach { m ->
            assertTrue("${m.id} size", m.sizeBytes > 1_000_000)
            val declared = m.files.sumOf { it.approxBytes }
            // model size should be within 2x of the sum of its parts
            assertTrue(
                "${m.id}: declared ${m.sizeMb}MB vs files ${declared / 1024 / 1024}MB",
                declared in (m.sizeBytes / 2)..(m.sizeBytes * 2)
            )
        }
    }
}
