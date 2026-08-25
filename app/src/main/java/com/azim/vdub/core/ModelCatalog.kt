package com.azim.vdub.core

/**
 * Every model the pipeline can use, with real, verified download URLs.
 *
 * ## Why everything fits on a 6 GB phone
 *
 * The "4 GB of models cannot fit" arithmetic adds the models together, but the
 * pipeline is *sequential*: diarization finishes before emotion starts, which
 * finishes before translation starts. Only one model is ever in memory, so the
 * ceiling is the **largest** model (~600 MB), not the sum. Each stage closes
 * its ONNX session before the next opens. That is why no server is needed.
 *
 * Disk is the real cost (~1.6 GB if you fetch everything), which is why models
 * are opt-in downloads rather than bundled in the APK.
 */
object ModelCatalog {

    enum class Stage(val step: Int, val label: String) {
        ASR(2, "Transcribe"),
        DIARIZATION(3, "Speakers"),
        EMOTION(4, "Emotion"),
        TRANSLATION(5, "Translate"),
        TTS(6, "Voice")
    }

    /**
     * @param required  pipeline cannot run this stage without it
     * @param files     every file that must land on disk (model + tokenizer…)
     */
    /** How the weights must be executed. */
    enum class Runtime {
        /** Loadable by ONNX Runtime Mobile — works today. */
        ONNX,
        /** PyTorch/safetensors — needs an inference path that does not exist yet. */
        SAFETENSORS
    }

    data class Model(
        val id: String,
        val name: String,
        val stage: Stage,
        val sizeBytes: Long,
        val description: String,
        val files: List<ModelFile>,
        val required: Boolean = true,
        val license: String = "",
        val note: String = "",
        val runtime: Runtime = Runtime.ONNX,
        /**
         * Working-set RAM while this stage runs. Usually close to the file
         * size, but a quantized model that dequantizes on the fly needs far
         * more than it occupies on disk — Chatterbox is 628 MB of weights and
         * ~1.9 GB live. Defaults to the file size when not stated.
         */
        val runtimeRamBytes: Long = 0L
    ) {
        val sizeMb: Int get() = (sizeBytes / 1024 / 1024).toInt()

        /** True when the app can actually run this model today. */
        val runnable: Boolean get() = runtime == Runtime.ONNX

        val ramBytes: Long get() = if (runtimeRamBytes > 0) runtimeRamBytes else sizeBytes
        val ramMb: Int get() = (ramBytes / 1024 / 1024).toInt()
    }

    data class ModelFile(
        /** Saved as /AI/models/{localName} */
        val localName: String,
        /** Mirrors tried in order. */
        val urls: List<String>,
        val approxBytes: Long,
        /** onnx = protobuf header checked; json/txt = text sanity check. */
        val kind: Kind = Kind.ONNX
    )

    enum class Kind { ONNX, ONNX_DATA, JSON, TEXT, BIN }

    /** Same as [hf]; named separately only for readability at the call sites. */
    private fun hfx(repo: String, path: String): List<String> = hf(repo, path)

    private fun hf(repo: String, path: String): List<String> = listOf(
        "https://huggingface.co/$repo/resolve/main/$path?download=true",
        "https://hf-mirror.com/$repo/resolve/main/$path?download=true"
    )

    // ------------------------------------------------------------ Step 3

    val CAMPPLUS = Model(
        id = "campplus",
        name = "CAM++ speaker embedding",
        stage = Stage.DIARIZATION,
        sizeBytes = 28_283_928L,
        description = "192-dim voice fingerprint per clip — tells speakers apart.",
        license = "Apache-2.0",
        files = listOf(
            ModelFile(
                localName = "campplus.onnx",
                urls = hf(
                    "welcomyou/campplus-3dspeaker-200k-onnx",
                    "campplus_cn_en_common_200k.onnx"
                ) + hf("Luigi/campplus-zh-en-onnx", "campplus_zh_en_fp32.onnx"),
                approxBytes = 28_283_928L
            )
        )
    )

    // ------------------------------------------------------------ Step 2

    val SENSEVOICE = Model(
        id = "sensevoice",
        name = "SenseVoice ASR",
        stage = Stage.ASR,
        sizeBytes = 249_000_000L,
        description = "Speech → text for zh/en/ja/ko/yue. Only needed if you " +
            "have no SRT file.",
        license = "Apache-2.0",
        required = false,
        files = listOf(
            ModelFile(
                localName = "sensevoice.onnx",
                urls = hf(
                    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                    "model.int8.onnx"
                ),
                approxBytes = 249_000_000L
            ),
            ModelFile(
                localName = "sensevoice_tokens.txt",
                urls = hf(
                    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                    "tokens.txt"
                ),
                approxBytes = 308_000L,
                kind = Kind.TEXT
            )
        )
    )

    // ------------------------------------------------------------ Step 4

    val EMOTION2VEC = Model(
        id = "emotion2vec",
        name = "emotion2vec+ base",
        stage = Stage.EMOTION,
        sizeBytes = 373_159_295L,
        description = "9 emotions per line (angry, happy, sad…) — drives how " +
            "expressively each line is spoken.",
        license = "MIT",
        files = listOf(
            ModelFile(
                localName = "emotion2vec.onnx",
                urls = hf(
                    "ziyu12345/emotion2vec_plus_base_onnx",
                    "emotion2vec_plus_base.onnx"
                ),
                approxBytes = 373_000_000L
            ),
            ModelFile(
                localName = "emotion2vec_head.json",
                urls = hf(
                    "ziyu12345/emotion2vec_plus_base_onnx",
                    "emotion2vec_head.json"
                ),
                approxBytes = 200_000L,
                kind = Kind.JSON
            )
        )
    )

    // ------------------------------------------------------------ Step 5

    val NLLB = Model(
        id = "nllb",
        name = "NLLB-200 distilled 600M",
        stage = Stage.TRANSLATION,
        sizeBytes = 620_000_000L,
        description = "Translates into Hindi (and 200 other languages) on-device.",
        license = "CC-BY-NC-4.0 — non-commercial",
        note = "Largest model: ~600 MB RAM while translating.",
        files = listOf(
            ModelFile(
                localName = "nllb_encoder.onnx",
                urls = hf("Xenova/nllb-200-distilled-600M", "onnx/encoder_model_quantized.onnx"),
                approxBytes = 340_000_000L
            ),
            ModelFile(
                localName = "nllb_decoder.onnx",
                urls = hf(
                    "Xenova/nllb-200-distilled-600M",
                    "onnx/decoder_model_merged_quantized.onnx"
                ),
                approxBytes = 280_000_000L
            ),
            ModelFile(
                localName = "nllb_tokenizer.json",
                urls = hf("Xenova/nllb-200-distilled-600M", "tokenizer.json"),
                approxBytes = 17_000_000L,
                kind = Kind.JSON
            )
        )
    )

    // ------------------------------------------------------------ Step 6

    /**
     * Chatterbox Hindi INT8 — the project's own TTS (Bbkblo/vdub-hindi-dubbing-lite).
     *
     * This is the voice-cloning model the pipeline was designed around: it
     * clones the original actor from their clip instead of using a preset
     * voice, and takes an `exaggeration` knob that the Step 3 emotion labels
     * feed directly (ANGRY 1.4 / HAPPY 1.1 / SAD 0.4 / NEUTRAL 0.5).
     *
     * Important: these weights are **safetensors + PyTorch**, not ONNX, and
     * the vocoder (S3Gen, ~1.06 GB) lives upstream and is fetched separately.
     * ONNX Runtime cannot load them, so Step 5 needs an inference path for
     * this format — see [runtime].
     */
    /**
     * Chatterbox Multilingual — ONNX export, the runnable path to voice cloning.
     *
     * Same model family as the project's own lite repo, but exported to ONNX by
     * onnx-community, so ONNX Runtime Mobile can actually execute it. Hindi is
     * one of its 23 languages, cloning is zero-shot from the speaker's own
     * clip, and it has the exaggeration control the Step 3 emotion feeds.
     *
     * Four graphs run in sequence: embed_tokens -> language_model (q4) ->
     * speech_encoder -> conditional_decoder. Weights live in .onnx_data
     * sidecars, which must sit next to their .onnx file.
     */
    val CHATTERBOX_ONNX = Model(
        id = "chatterbox_onnx",
        name = "Chatterbox Multilingual (voice cloning)",
        stage = Stage.TTS,
        sizeBytes = 1_555_000_000L,
        description = "Clones each speaker from their own clips and speaks the " +
            "Hindi lines. Zero-shot — no training, no reference text.",
        license = "MIT (ResembleAI) — output carries a PerTh watermark",
        note = "Largest download. Runs the language model in q4 to keep RAM near " +
            "1.5 GB; expect roughly a minute per line on a phone CPU.",
        files = listOf(
            ModelFile(
                localName = "chatterbox/embed_tokens.onnx",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx"),
                approxBytes = 13_286L
            ),
            ModelFile(
                localName = "chatterbox/embed_tokens.onnx_data",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/embed_tokens.onnx_data"),
                approxBytes = 68_390_912L,
                kind = Kind.ONNX_DATA
            ),
            ModelFile(
                localName = "chatterbox/language_model_q4.onnx",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx"),
                approxBytes = 227_911L
            ),
            ModelFile(
                localName = "chatterbox/language_model_q4.onnx_data",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model_q4.onnx_data"),
                approxBytes = 353_621_248L,
                kind = Kind.ONNX_DATA
            ),
            ModelFile(
                localName = "chatterbox/speech_encoder.onnx",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx"),
                approxBytes = 1_184_608L
            ),
            ModelFile(
                localName = "chatterbox/speech_encoder.onnx_data",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/speech_encoder.onnx_data"),
                approxBytes = 591_274_880L,
                kind = Kind.ONNX_DATA
            ),
            ModelFile(
                localName = "chatterbox/conditional_decoder.onnx",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx"),
                approxBytes = 6_350_448L
            ),
            ModelFile(
                localName = "chatterbox/conditional_decoder.onnx_data",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "onnx/conditional_decoder.onnx_data"),
                approxBytes = 533_970_816L,
                kind = Kind.ONNX_DATA
            ),
            ModelFile(
                localName = "chatterbox/tokenizer.json",
                urls = hfx("onnx-community/chatterbox-multilingual-ONNX", "tokenizer.json"),
                approxBytes = 25_470L,
                kind = Kind.JSON
            )
        ),
        runtimeRamBytes = 1_500_000_000L
    )

    val CHATTERBOX_HI = Model(
        id = "chatterbox_hi",
        name = "Chatterbox Hindi INT8 (PyTorch, not runnable yet)",
        stage = Stage.TTS,
        required = false,
        sizeBytes = 658_584_623L,
        description = "Clones each speaker's voice from their own clips and " +
            "speaks the Hindi lines. Emotion sets the exaggeration.",
        license = "MIT (ResembleAI) — output carries a PerTh watermark",
        note = "Q8_0 quantized Llama-520M. Needs the S3Gen vocoder (~1.06 GB) " +
            "as well; ~1.9 GB RAM while speaking.",
        runtime = Runtime.SAFETENSORS,
        runtimeRamBytes = 1_900_000_000L,
        files = listOf(
            ModelFile(
                localName = "chatterbox/t3_hi_int8.safetensors",
                urls = hf(
                    "Bbkblo/vdub-hindi-dubbing-lite",
                    "models/chatterbox_hi_lite/t3_hi_int8.safetensors"
                ),
                approxBytes = 652_816_008L,
                kind = Kind.BIN
            ),
            ModelFile(
                localName = "chatterbox/ve.pt",
                urls = hf(
                    "Bbkblo/vdub-hindi-dubbing-lite",
                    "models/chatterbox_hi_lite/ve.pt"
                ),
                approxBytes = 5_698_626L,
                kind = Kind.BIN
            ),
            ModelFile(
                localName = "chatterbox/grapheme_mtl_merged_expanded_v1.json",
                urls = hf(
                    "Bbkblo/vdub-hindi-dubbing-lite",
                    "models/chatterbox_hi_lite/grapheme_mtl_merged_expanded_v1.json"
                ),
                approxBytes = 69_989L,
                kind = Kind.JSON
            ),
            ModelFile(
                localName = "chatterbox/quant_manifest.json",
                urls = hf(
                    "Bbkblo/vdub-hindi-dubbing-lite",
                    "models/chatterbox_hi_lite/quant_manifest.json"
                ),
                approxBytes = 60_258L,
                kind = Kind.JSON
            )
        )
    )

    val ALL = listOf(CAMPPLUS, SENSEVOICE, EMOTION2VEC, NLLB, CHATTERBOX_ONNX, CHATTERBOX_HI)

    fun byId(id: String): Model? = ALL.firstOrNull { it.id == id }

    fun forStage(stage: Stage): List<Model> = ALL.filter { it.stage == stage }

    /** Total disk if the user downloads everything. */
    val totalBytes: Long get() = ALL.sumOf { it.sizeBytes }

    /**
     * Peak RAM is the heaviest single stage, since stages run one at a time —
     * not the sum. Uses working-set RAM, which for a dequantizing model is
     * larger than its file.
     */
    val peakRamBytes: Long get() = ALL.maxOf { it.ramBytes }

    /** Peak across the stages that actually run today (ONNX only). */
    val peakRunnableRamBytes: Long get() =
        ALL.filter { it.runnable }.maxOf { it.ramBytes }
}
