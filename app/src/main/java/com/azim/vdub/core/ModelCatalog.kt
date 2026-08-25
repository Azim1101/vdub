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

    private const val CB_Q4 = "verify01234/chatterbox-multilingual-ONNX-q4"

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
     * Chatterbox Multilingual — Q4 ONNX, the runnable voice-cloning path.
     *
     * Same model family as the project's lite repo, but a real ONNX export so
     * ONNX Runtime Mobile can execute it. Hindi is one of its 23 languages,
     * cloning is zero-shot from the speaker's own clip, and the exaggeration
     * control is what Step 3's emotion labels feed.
     *
     * This q4 build is single-file: the upstream export splits weights into
     * .onnx_data sidecars that must sit beside their graph, which is fragile
     * on Android storage. Here each graph is self-contained, and the whole set
     * is 790 MB instead of 1.45 GB.
     *
     * Four graphs run in sequence:
     *   embed_tokens -> language_model -> speech_encoder -> conditional_decoder
     *
     * Generation must use repetition_penalty 1.2 — the upstream default of 2.0
     * sends this quantized build into an infinite loop.
     */
    val CHATTERBOX_ONNX = Model(
        id = "chatterbox_onnx",
        name = "Chatterbox Multilingual (voice cloning)",
        stage = Stage.TTS,
        sizeBytes = 830_000_000L,
        description = "Clones each speaker from their own clips and speaks the " +
            "Hindi lines. Zero-shot — no training, no reference text.",
        license = "MIT (ResembleAI) — output carries a PerTh watermark",
        note = "Q4 single-file build. Roughly 1 GB RAM and about a minute per " +
            "line on a phone CPU.",
        files = listOf(
            ModelFile(
                localName = "chatterbox/embed_tokens.onnx",
                urls = hf(CB_Q4, "onnx/embed_tokens.onnx"),
                approxBytes = 68_420_479L
            ),
            ModelFile(
                localName = "chatterbox/language_model.onnx",
                urls = hf(CB_Q4, "onnx/language_model.onnx"),
                approxBytes = 353_810_438L
            ),
            ModelFile(
                localName = "chatterbox/speech_encoder.onnx",
                urls = hf(CB_Q4, "onnx/speech_encoder.onnx"),
                approxBytes = 180_077_492L
            ),
            ModelFile(
                localName = "chatterbox/conditional_decoder.onnx",
                urls = hf(CB_Q4, "onnx/conditional_decoder.onnx"),
                approxBytes = 225_572_798L
            ),
            ModelFile(
                localName = "chatterbox/tokenizer.json",
                urls = hf(CB_Q4, "tokenizer.json"),
                approxBytes = 71_798L,
                kind = Kind.JSON
            ),
            ModelFile(
                localName = "chatterbox/tokenizer_config.json",
                urls = hf(CB_Q4, "tokenizer_config.json"),
                approxBytes = 244L,
                kind = Kind.JSON
            ),
            ModelFile(
                localName = "chatterbox/generation_config.json",
                urls = hf(CB_Q4, "generation_config.json"),
                approxBytes = 93L,
                kind = Kind.JSON
            ),
            // Needed to tokenize Chinese source text; also the fallback voice
            // when a speaker's own clips are unusable.
            ModelFile(
                localName = "chatterbox/Cangjie5_TC.json",
                urls = hf(CB_Q4, "Cangjie5_TC.json"),
                approxBytes = 1_920_163L,
                kind = Kind.JSON
            ),
            ModelFile(
                localName = "chatterbox/default_voice.wav",
                urls = hf(CB_Q4, "default_voice.wav"),
                approxBytes = 714_320L,
                kind = Kind.BIN
            )
        ),
        runtimeRamBytes = 1_100_000_000L
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

    /**
     * Peak for everything before speaking. These run over every clip, so they
     * are the figure that matters for day-to-day use.
     */
    val peakAnalysisRamBytes: Long get() =
        ALL.filter { it.runnable && it.stage != Stage.TTS }.maxOf { it.ramBytes }
}
