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
    fun `voice cloning stays reachable`() {
        val tts = ModelCatalog.forStage(ModelCatalog.Stage.TTS)
        assertTrue(tts.isNotEmpty())
        // Cloning the original actor is the point of the pipeline, so at least
        // one selectable engine must still do it however many are added.
        val cloning = ModelCatalog.CLONING_ENGINES
        assertTrue("no cloning engine left", cloning.isNotEmpty())
        assertTrue(cloning.all { it.runnable })
        assertTrue(
            cloning.any { it.description.contains("Clones", ignoreCase = true) }
        )
    }

    /**
     * At least one TTS option must actually be executable, otherwise Step 5
     * can never run.
     */
    @Test
    fun `every voice engine is runnable and distinct`() {
        val engines = ModelCatalog.VOICE_ENGINES
        assertEquals(
            listOf("chatterbox_q4", "chatterbox_mix", "dhvaani", "indri"),
            engines.map { it.id }
        )
        // An engine the app cannot execute must never be offered as a choice —
        // chatterbox_hi is in the catalog for visibility but is not selectable.
        assertTrue(engines.all { it.runnable })
        assertEquals(engines.size, engines.map { it.id }.distinct().size)
        assertTrue(ModelCatalog.CHATTERBOX_HI !in engines)
    }

    /**
     * The two packs use identical filenames for entirely different weights, so
     * they must live in separate folders or one silently overwrites the other.
     */
    @Test
    fun `voice engines do not share any file path`() {
        val engines = ModelCatalog.VOICE_ENGINES
        for (i in engines.indices) {
            for (j in i + 1 until engines.size) {
                val a = engines[i].files.map { it.localName }.toSet()
                val b = engines[j].files.map { it.localName }.toSet()
                assertTrue(
                    "${engines[i].id} and ${engines[j].id} overlap: ${a intersect b}",
                    (a intersect b).isEmpty()
                )
            }
        }
    }

    /** Each pack must ship the four graphs the pipeline runs. */
    @Test
    fun `each chatterbox pack ships all four graphs`() {
        ModelCatalog.VOICE_ENGINES
            .filter { it.id.startsWith("chatterbox") }
            .forEach { engine ->
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
     * Whatever the shape of an engine, it must ship at least one graph to run
     * and its files must all be reachable — an entry that downloads nothing
     * executable would install "successfully" and then fail to open.
     */
    @Test
    fun `every voice engine ships something runnable`() {
        ModelCatalog.VOICE_ENGINES.forEach { engine ->
            assertTrue(
                "${engine.id} has no onnx graph",
                engine.files.any { it.localName.endsWith(".onnx") }
            )
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

    // ------------------------------------------------ the added TTS engines

    /**
     * DhVaani's two numpy archives are not ONNX graphs, and declaring them as
     * ONNX would make the downloader reject them for a bad protobuf header.
     */
    @Test
    fun `npz files are declared as npz`() {
        ModelCatalog.ALL.flatMap { it.files }
            .filter { it.localName.endsWith(".npz") }
            .also { assertTrue("no npz files declared", it.isNotEmpty()) }
            .forEach { assertEquals(it.localName, ModelCatalog.Kind.NPZ, it.kind) }
    }

    @Test
    fun `txt files are declared as text`() {
        ModelCatalog.ALL.flatMap { it.files }
            .filter { it.localName.endsWith(".txt") }
            .forEach { assertEquals(it.localName, ModelCatalog.Kind.TEXT, it.kind) }
    }

    /**
     * Every engine keeps its own folder. The four engines share leaf names
     * (`vocab.json`, `tokenizer.json`, `language_model.onnx`), so a flat
     * layout would have one silently overwrite another's weights.
     */
    @Test
    fun `each voice engine downloads into its own folder`() {
        val folders = ModelCatalog.VOICE_ENGINES.map { engine ->
            val dirs = engine.files.map { it.localName.substringBeforeLast('/') }.distinct()
            assertEquals("${engine.id} spans folders $dirs", 1, dirs.size)
            dirs.first()
        }
        assertEquals("engines share a folder", folders.size, folders.distinct().size)
    }

    /**
     * Indri's decoder comes from Kyutai's Mimi repo, not from Indri's own —
     * Indri ships the language model only. If that entry ever disappears the
     * engine downloads happily and then cannot produce a waveform at all.
     */
    @Test
    fun `indri ships a mimi decoder alongside its language model`() {
        val indri = ModelCatalog.INDRI_TTS
        val leaves = indri.files.map { it.localName.substringAfterLast('/') }
        assertTrue("no mimi decoder", leaves.contains("mimi_decoder.onnx"))
        assertTrue("no language model", leaves.contains("indri_lm.onnx"))

        val decoder = indri.files.first { it.localName.endsWith("mimi_decoder.onnx") }
        assertTrue(
            "decoder must come from the mimi repo",
            decoder.urls.all { it.contains("kyutai-mimi-ONNX") }
        )
        // fp16: int8's own quantization error (15.6 dB) exceeds the padding
        // error it would be masking. See tools/probe_tts5.py.
        assertTrue(
            "decoder should be the fp16 export",
            decoder.urls.all { it.contains("fp16") }
        )
    }

    /** DhVaani needs all three graphs plus both npz files and its vocabulary. */
    @Test
    fun `dhvaani ships every piece its pipeline needs`() {
        val leaves = ModelCatalog.DHVAANI_TTS.files
            .map { it.localName.substringAfterLast('/') }
            .toSet()
        listOf(
            "text_encoder.onnx", "fm_decoder.onnx", "vocoder_backbone.onnx",
            "vocos_head.npz", "mel_fb.npz", "tokens.txt"
        ).forEach { assertTrue("dhvaani is missing $it", it in leaves) }
    }

    /** Indri is research-only and cannot be presented as freely usable. */
    @Test
    fun `restrictive licences are stated`() {
        assertTrue(
            ModelCatalog.INDRI_TTS.license.contains("research", ignoreCase = true) ||
                ModelCatalog.INDRI_TTS.license.contains("CC-BY-SA")
        )
        assertTrue(ModelCatalog.NLLB.license.contains("NC"))
    }

    /**
     * The point of adding DhVaani: it is far smaller than Chatterbox, and if a
     * future edit inflates it past that the reason to offer it is gone.
     */
    @Test
    fun `dhvaani stays much smaller than chatterbox`() {
        assertTrue(
            "dhvaani ${ModelCatalog.DHVAANI_TTS.sizeMb} MB vs " +
                "chatterbox ${ModelCatalog.CHATTERBOX_Q4.sizeMb} MB",
            ModelCatalog.DHVAANI_TTS.sizeBytes < ModelCatalog.CHATTERBOX_Q4.sizeBytes / 2
        )
        assertTrue(
            ModelCatalog.DHVAANI_TTS.ramBytes < ModelCatalog.CHATTERBOX_Q4.ramBytes
        )
    }
}
