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
    fun `peak ram is the largest model and fits a 6gb phone`() {
        assertEquals(ModelCatalog.ALL.maxOf { it.sizeBytes }, ModelCatalog.peakRamBytes)
        assertTrue(ModelCatalog.peakRamBytes < ModelCatalog.totalBytes)
        // comfortably under what a 6 GB device gives one app
        assertTrue(
            "peak ${ModelCatalog.peakRamBytes / 1024 / 1024} MB too big",
            ModelCatalog.peakRamBytes < 1_000_000_000L
        )
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
