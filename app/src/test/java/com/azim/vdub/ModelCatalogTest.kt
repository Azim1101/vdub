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

    /**
     * The voice stage is the only one that runs after translation, so it must
     * be reachable: a required, runnable model in the TTS stage.
     */
    @Test
    fun `pipeline has a runnable model for every required stage`() {
        ModelCatalog.Stage.entries.forEach { stage ->
            val usable = ModelCatalog.forStage(stage).filter { it.runnable }
            assertTrue("no runnable model for $stage", usable.isNotEmpty())
        }
    }

    /** Sizes must not silently drift from what is really hosted. */
    @Test
    fun `voice engine sizes match the sum of their files`() {
        ModelCatalog.VOICE_ENGINES.forEach { m ->
            val parts = m.files.sumOf { it.approxBytes }
            assertTrue(
                "${m.id}: declared ${m.sizeMb} MB vs parts ${parts / 1024 / 1024} MB",
                parts in (m.sizeBytes * 9 / 10)..(m.sizeBytes * 11 / 10)
            )
        }
    }

    /**
     * Peak RAM is one engine, never both — they are mutually exclusive, so
     * loading is sequential and the ceiling is the larger of the two.
     */
    @Test
    fun `peak voice ram is the larger engine, not the sum`() {
        val engines = ModelCatalog.VOICE_ENGINES
        val sum = engines.sumOf { it.ramBytes }
        val peak = engines.maxOf { it.ramBytes }
        assertTrue(peak < sum)
        assertEquals(peak, ModelCatalog.forStage(ModelCatalog.Stage.TTS)
            .filter { it.runnable }.maxOf { it.ramBytes })
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
    fun `both voice engines are runnable and distinct`() {
        val engines = ModelCatalog.VOICE_ENGINES
        assertEquals(2, engines.size)
        assertEquals(listOf("chatterbox_q4", "chatterbox_mix"), engines.map { it.id })
        assertTrue(engines.all { it.runnable })
        assertEquals(engines.size, engines.map { it.id }.distinct().size)
    }

    /**
     * The two packs use identical filenames for entirely different weights, so
     * they must live in separate folders or one silently overwrites the other.
     */
    @Test
    fun `voice engines do not share any file path`() {
        val (a, b) = ModelCatalog.VOICE_ENGINES
        val pathsA = a.files.map { it.localName }.toSet()
        val pathsB = b.files.map { it.localName }.toSet()
        assertTrue("packs overlap: ${pathsA intersect pathsB}", (pathsA intersect pathsB).isEmpty())
    }

    /** Each pack must ship the four graphs the pipeline runs. */
    @Test
    fun `each voice engine ships all four graphs`() {
        ModelCatalog.VOICE_ENGINES.forEach { engine ->
            val leaf = engine.files.map { it.localName.substringAfterLast('/') }
            listOf("embed_tokens.onnx", "speech_encoder.onnx", "conditional_decoder.onnx")
                .forEach { g -> assertTrue("${engine.id} missing $g", leaf.contains(g)) }
            assertTrue(
                "${engine.id} has no language model",
                leaf.any { it.startsWith("language_model") }
            )
            assertTrue("${engine.id} has no tokenizer", leaf.contains("tokenizer.json"))
        }
    }

    /**
     * In the mix pack only the language model is quantized; the graphs that
     * decide clone identity and audio quality must stay full precision, which
     * shows up as their FP32 sidecar sizes.
     */
    @Test
    fun `mix pack keeps encoder and decoder full precision`() {
        val mix = ModelCatalog.byId("chatterbox_mix")!!
        fun bytesOf(prefix: String) = mix.files
            .filter { it.localName.substringAfterLast('/').startsWith(prefix) }
            .sumOf { it.approxBytes }

        assertTrue("encoder looks quantized", bytesOf("speech_encoder") > 500_000_000L)
        assertTrue("decoder looks quantized", bytesOf("conditional_decoder") > 400_000_000L)
        // and the LLM is the one that shrank
        assertTrue("llm not quantized", bytesOf("language_model") < 400_000_000L)
    }

    /**
     * An external-data graph hardcodes its sidecar filename, so the local name
     * must keep the upstream spelling — renaming language_model_q4 would break
     * loading at run time, not at download time.
     */
    @Test
    fun `mix pack keeps upstream language model filename`() {
        val mix = ModelCatalog.byId("chatterbox_mix")!!
        val names = mix.files.map { it.localName.substringAfterLast('/') }
        assertTrue(names.contains("language_model_q4.onnx"))
        assertTrue(names.contains("language_model_q4.onnx_data"))
    }

    /**
     * If a sidecar is ever reintroduced it must be typed as data — it holds raw
     * weights, not protobuf, so the ONNX header check would reject it.
     */
    @Test
    fun `onnx data sidecars are typed as data`() {
        ModelCatalog.ALL.flatMap { it.files }
            .filter { it.localName.endsWith(".onnx_data") }
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
     * The voice model is deliberately the single-file q4 build: external-data
     * sidecars must sit beside their graph, which is fragile on Android
     * storage and doubles the download.
     */
    @Test
    fun `voice model needs no external data files`() {
        val m = ModelCatalog.byId("chatterbox_onnx")!!
        assertTrue(m.files.none { it.localName.endsWith(".onnx_data") })
        assertTrue(m.files.any { it.localName.endsWith("language_model.onnx") })
    }

    /** The four graphs the pipeline runs in sequence must all be present. */
    @Test
    fun `voice model ships all four graphs and its tokenizer`() {
        val names = ModelCatalog.byId("chatterbox_onnx")!!.files.map { it.localName }
        listOf(
            "embed_tokens.onnx", "language_model.onnx",
            "speech_encoder.onnx", "conditional_decoder.onnx", "tokenizer.json"
        ).forEach { needed ->
            assertTrue("missing $needed", names.any { it.endsWith(needed) })
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

        listOf("campplus", "emotion2vec", "nllb", "sensevoice",
            "chatterbox_q4", "chatterbox_mix").forEach {
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
