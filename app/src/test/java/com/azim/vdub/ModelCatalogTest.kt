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

    /**
     * Analysis stages (everything before speaking) must stay light — those run
     * over all 190 clips, so a heavy one would be felt on every line.
     */
    @Test
    fun `analysis stages stay under 1gb`() {
        val analysis = ModelCatalog.ALL
            .filter { it.runnable && it.stage != ModelCatalog.Stage.TTS }
        assertTrue(analysis.isNotEmpty())
        val peak = analysis.maxOf { it.ramBytes }
        assertTrue("peak ${peak / 1024 / 1024} MB too big", peak < 1_000_000_000L)
    }

    /**
     * Speaking is the heavy stage and always will be. It must still fit a
     * 6 GB phone with room for the OS and the app itself.
     */
    @Test
    fun `voice stage fits a 6gb phone`() {
        val tts = ModelCatalog.forStage(ModelCatalog.Stage.TTS).filter { it.runnable }
        assertTrue(tts.isNotEmpty())
        tts.forEach {
            assertTrue(
                "${it.id} needs ${it.ramMb} MB",
                it.ramBytes < 2_500_000_000L
            )
        }
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

    /** Sizes must not silently drift from what is really hosted. */
    @Test
    fun `chatterbox onnx totals roughly its declared size`() {
        val m = ModelCatalog.byId("chatterbox_onnx")!!
        val parts = m.files.sumOf { it.approxBytes }
        assertTrue(
            "declared ${m.sizeMb} MB vs parts ${parts / 1024 / 1024} MB",
            parts in (m.sizeBytes * 9 / 10)..(m.sizeBytes * 11 / 10)
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
    fun `tts stays chatterbox, and voice cloning is the point`() {
        val tts = ModelCatalog.forStage(ModelCatalog.Stage.TTS)
        assertTrue(tts.isNotEmpty())
        assertTrue(tts.all { it.name.contains("Chatterbox", ignoreCase = true) })
        assertTrue(tts.all { it.description.contains("Clones", ignoreCase = true) })
    }

    /**
     * At least one TTS option must actually be executable, otherwise Step 5
     * can never run.
     */
    @Test
    fun `a runnable voice model exists`() {
        val runnable = ModelCatalog.forStage(ModelCatalog.Stage.TTS).filter { it.runnable }
        assertTrue("no runnable TTS", runnable.isNotEmpty())
        assertEquals("chatterbox_onnx", runnable.first().id)
    }

    /**
     * ONNX external-data sidecars are raw weights, not protobuf, so they must
     * not be validated as ONNX or every large model would fail to install.
     */
    @Test
    fun `onnx data sidecars are typed as data`() {
        ModelCatalog.ALL.flatMap { it.files }
            .filter { it.localName.endsWith(".onnx_data") }
            .also { assertTrue("expected sidecars", it.isNotEmpty()) }
            .forEach { assertEquals(it.localName, ModelCatalog.Kind.ONNX_DATA, it.kind) }
    }

    /** A sidecar is useless without the graph that references it. */
    @Test
    fun `every sidecar has its graph`() {
        ModelCatalog.ALL.forEach { m ->
            val names = m.files.map { it.localName }.toSet()
            names.filter { it.endsWith(".onnx_data") }.forEach { data ->
                val graph = data.removeSuffix("_data")
                assertTrue("$data has no $graph", names.contains(graph))
            }
        }
    }

    /**
     * Chatterbox ships safetensors, which ONNX Runtime cannot load. The flag
     * must say so, otherwise Settings would present it as ready and Step 5
     * would fail at run time instead.
     */
    @Test
    fun `safetensors models are marked not runnable`() {
        val cb = ModelCatalog.byId("chatterbox_hi")!!
        assertTrue(!cb.required)
        assertEquals(ModelCatalog.Runtime.SAFETENSORS, cb.runtime)
        assertTrue(!cb.runnable)

        listOf("campplus", "emotion2vec", "nllb", "sensevoice", "chatterbox_onnx").forEach {
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
