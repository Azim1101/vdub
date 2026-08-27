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

    enum class Kind { ONNX, ONNX_DATA, JSON, TEXT, BIN, NPZ }

    private const val CB_Q4 = "verify01234/chatterbox-multilingual-ONNX-q4"
    private const val CB_MIX = "onnx-community/chatterbox-multilingual-ONNX"
    private const val DHVAANI = "Bbkblo/DhVaani-0.5-ONNX"
    private const val INDRI = "Bbkblo/indri-0.1-124m-tts-ONNX"
    private const val MIMI = "onnx-community/kyutai-mimi-ONNX"

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
    /**
     * Voice engine A — everything quantized to Q4, single-file.
     *
     * Smallest and fastest. Each graph is self-contained, so there are no
     * external-data sidecars to keep adjacent. Quantizing the speech encoder
     * and vocoder costs some clone fidelity and audio detail, which is the
     * trade for half the size of [CHATTERBOX_MIX].
     *
     * Generation must use repetition_penalty 1.2 — the upstream default of 2.0
     * sends this build into an infinite loop.
     */
    val CHATTERBOX_Q4 = Model(
        id = "chatterbox_q4",
        name = "Chatterbox Q4 (small + fast)",
        stage = Stage.TTS,
        sizeBytes = 830_000_000L,
        description = "Clones each speaker from their own clips and speaks the " +
            "Hindi lines. Everything Q4 — smallest download, quickest.",
        license = "MIT (ResembleAI) — output carries a PerTh watermark",
        note = "~1.1 GB RAM. Clone and audio are a little softer than the mix pack.",
        files = listOf(
            ModelFile("chatterbox_q4/embed_tokens.onnx",
                hf(CB_Q4, "onnx/embed_tokens.onnx"), 68_420_479L),
            ModelFile("chatterbox_q4/language_model.onnx",
                hf(CB_Q4, "onnx/language_model.onnx"), 353_810_438L),
            ModelFile("chatterbox_q4/speech_encoder.onnx",
                hf(CB_Q4, "onnx/speech_encoder.onnx"), 180_077_492L),
            ModelFile("chatterbox_q4/conditional_decoder.onnx",
                hf(CB_Q4, "onnx/conditional_decoder.onnx"), 225_572_798L),
            ModelFile("chatterbox_q4/tokenizer.json",
                hf(CB_Q4, "tokenizer.json"), 71_798L, Kind.JSON),
            ModelFile("chatterbox_q4/tokenizer_config.json",
                hf(CB_Q4, "tokenizer_config.json"), 244L, Kind.JSON),
            ModelFile("chatterbox_q4/generation_config.json",
                hf(CB_Q4, "generation_config.json"), 93L, Kind.JSON),
            ModelFile("chatterbox_q4/Cangjie5_TC.json",
                hf(CB_Q4, "Cangjie5_TC.json"), 1_920_163L, Kind.JSON),
            ModelFile("chatterbox_q4/default_voice.wav",
                hf(CB_Q4, "default_voice.wav"), 714_320L, Kind.BIN)
        ),
        runtimeRamBytes = 1_100_000_000L
    )

    /**
     * Voice engine B — mixed precision, the quality-per-byte choice.
     *
     * The two graphs that decide how the result *sounds* stay FP32:
     *   speech_encoder      reads the reference clip -> speaker identity
     *   conditional_decoder turns tokens into the waveform -> audio detail
     * Only the language model is Q4, and that is where the bulk of the weight
     * sits (1984 MB -> 337 MB), so the size falls by more than half while the
     * parts that carry the voice are untouched.
     *
     * These graphs use ONNX external data: each .onnx has a matching
     * .onnx_data that must sit beside it under the exact upstream filename,
     * because the reference is baked into the graph. Hence language_model_q4
     * keeps its name rather than being normalised.
     *
     * Kept in its own folder: the filenames collide with [CHATTERBOX_Q4] but
     * the contents are entirely different, so one would overwrite the other.
     */
    val CHATTERBOX_MIX = Model(
        id = "chatterbox_mix",
        name = "Chatterbox mix (Q4 LLM)",
        stage = Stage.TTS,
        sizeBytes = 1_560_000_000L,
        description = "Same cloning, better fidelity. Clone and vocoder stay " +
            "full precision; only the language model is Q4.",
        license = "MIT (ResembleAI) — output carries a PerTh watermark",
        note = "Recommended. ~1.6 GB download, ~1.6 GB RAM while speaking.",
        files = listOf(
            // FP32 — reads the reference clip, decides the cloned identity
            ModelFile("chatterbox_mix/speech_encoder.onnx",
                hf(CB_MIX, "onnx/speech_encoder.onnx"), 1_184_608L),
            ModelFile("chatterbox_mix/speech_encoder.onnx_data",
                hf(CB_MIX, "onnx/speech_encoder.onnx_data"), 591_274_880L, Kind.ONNX_DATA),
            // FP32 — token embeddings
            ModelFile("chatterbox_mix/embed_tokens.onnx",
                hf(CB_MIX, "onnx/embed_tokens.onnx"), 13_286L),
            ModelFile("chatterbox_mix/embed_tokens.onnx_data",
                hf(CB_MIX, "onnx/embed_tokens.onnx_data"), 68_390_912L, Kind.ONNX_DATA),
            // Q4 — the size cut. Filename must match the sidecar reference.
            ModelFile("chatterbox_mix/language_model_q4.onnx",
                hf(CB_MIX, "onnx/language_model_q4.onnx"), 227_911L),
            ModelFile("chatterbox_mix/language_model_q4.onnx_data",
                hf(CB_MIX, "onnx/language_model_q4.onnx_data"), 353_621_248L, Kind.ONNX_DATA),
            // FP32 — turns speech tokens into the waveform
            ModelFile("chatterbox_mix/conditional_decoder.onnx",
                hf(CB_MIX, "onnx/conditional_decoder.onnx"), 6_350_448L),
            ModelFile("chatterbox_mix/conditional_decoder.onnx_data",
                hf(CB_MIX, "onnx/conditional_decoder.onnx_data"), 533_970_816L, Kind.ONNX_DATA),
            ModelFile("chatterbox_mix/tokenizer.json",
                hf(CB_MIX, "tokenizer.json"), 25_470L, Kind.JSON),
            ModelFile("chatterbox_mix/tokenizer_config.json",
                hf(CB_MIX, "tokenizer_config.json"), 244L, Kind.JSON),
            ModelFile("chatterbox_mix/generation_config.json",
                hf(CB_MIX, "generation_config.json"), 93L, Kind.JSON),
            ModelFile("chatterbox_mix/Cangjie5_TC.json",
                hf(CB_MIX, "Cangjie5_TC.json"), 1_920_163L, Kind.JSON),
            ModelFile("chatterbox_mix/default_voice.wav",
                hf(CB_MIX, "default_voice.wav"), 714_320L, Kind.BIN)
        ),
        required = false,
        runtimeRamBytes = 1_600_000_000L
    )

    /**
     * Voice engine C — DhVaani 0.5, a ZipVoice flow-matching model for Indic
     * languages (ARTPARK-IISc), exported to ONNX.
     *
     * The interesting one for this project: 183 MB against Chatterbox's 830,
     * and it generates *faster than real time* (RTF 0.84 at 4 sampling steps
     * on two CPU cores) because there is no autoregressive loop — the whole
     * utterance is denoised in a fixed number of passes. Chatterbox spends a
     * minute a line emitting speech tokens one at a time.
     *
     * Three graphs, run once per line:
     *   text_encoder  tokens + reference tokens -> a (1, T, 100) mel plan
     *   fm_decoder    one Euler step of the flow; called `steps` times
     *   vocoder       mel -> waveform, via a Vocos ISTFT head
     *
     * Two files are numpy archives rather than graphs, which is why [Kind.NPZ]
     * exists: `vocos_head.npz` holds the final linear layer and ISTFT window,
     * and `mel_fb.npz` the exact torchaudio-HTK filterbank. The filterbank is
     * not optional — recomputing it from a formula gives subtly different mel
     * bins, and the reference clip then encodes to features the model reads as
     * a different voice.
     *
     * Cloning is zero-shot but conditions on the reference *text* as well as
     * its audio, so [TtsEngine.enrol] takes a transcript here.
     */
    val DHVAANI_TTS = Model(
        id = "dhvaani",
        name = "DhVaani 0.5 (Indic, fast)",
        stage = Stage.TTS,
        sizeBytes = 191_000_000L,
        description = "Zero-shot cloning for 13 Indic languages. Six times " +
            "smaller than Chatterbox and faster than real time.",
        license = "Apache-2.0 (ARTPARK-IISc / ZipVoice)",
        note = "Recommended for Hindi. ~600 MB RAM, no watermark.",
        required = false,
        files = listOf(
            ModelFile("dhvaani/text_encoder.onnx",
                hf(DHVAANI, "text_encoder_int8.onnx"), 6_131_125L),
            ModelFile("dhvaani/fm_decoder.onnx",
                hf(DHVAANI, "fm_decoder_int8.onnx"), 124_752_448L),
            ModelFile("dhvaani/vocoder_backbone.onnx",
                hf(DHVAANI, "vocoder_backbone.onnx"), 52_048_662L),
            ModelFile("dhvaani/vocos_head.npz",
                hf(DHVAANI, "vocos_head.npz"), 2_110_996L, Kind.NPZ),
            ModelFile("dhvaani/mel_fb.npz",
                hf(DHVAANI, "mel_fb.npz"), 210_546L, Kind.NPZ),
            ModelFile("dhvaani/tokens.txt",
                hf(DHVAANI, "tokens.txt"), 8_065L, Kind.TEXT)
        ),
        runtimeRamBytes = 600_000_000L
    )

    /**
     * Voice engine D — Indri 0.1 (11mlabs), a 124M GPT-2 emitting Mimi codec
     * tokens, decoded to audio by Kyutai's Mimi.
     *
     * Preset voices only. Indri conditions on a `[spkr_NN]` token, not on a
     * reference clip, so unlike the other three it cannot reproduce the
     * original actor — Step 5 says so before a three-hour run rather than
     * after. It is here because it is small, and its Hindi preset speakers are
     * natural.
     *
     * ### The decoder had to be repaired, not just downloaded
     *
     * Indri's own repo ships the language model only: Kyutai Mimi does not
     * export to ONNX from transformers, so upstream points at llama.cpp for
     * the waveform. There *is* a community ONNX export, but its decoder is
     * frozen at 32 codebooks while Indri emits 8.
     *
     * Padding the other 24 with index 0 runs and sounds wrong — measured at
     * 6.8 dB SNR against the 8-codebook decode upstream performs, in PyTorch,
     * so it is not a quantization artefact. Mimi's residual quantizer decodes
     * by summing one embedding per codebook, so those 24 add a constant
     * vector C = Σ codebook_q[0] to every frame, and index 0 is an ordinary
     * vector rather than silence.
     *
     * Index 0 is not required, though: any index is legal. Choosing the 24
     * indices whose embeddings cancel drops ‖C‖ from 1.383 to 0.045 and the
     * error to 34.4 dB — inaudible. Those indices are pinned in [MimiDecoder]
     * and covered by a test.
     *
     * fp16 is used for the decoder: 34.41 dB versus 15.60 for int8 (whose
     * quantization error alone exceeds the padding error), at half the size of
     * fp32 and a third of int8's runtime.
     */
    val INDRI_TTS = Model(
        id = "indri",
        name = "Indri 0.1 (preset voices)",
        stage = Stage.TTS,
        sizeBytes = 488_000_000L,
        description = "Small English/Hindi TTS with 13 preset speakers. Does " +
            "not clone the original voice.",
        license = "CC-BY-SA-4.0, research only (11mlabs) · Mimi CC-BY-4.0",
        note = "No cloning, and slow: the exported graph has no KV cache, so " +
            "every token re-runs the whole sequence.",
        required = false,
        files = listOf(
            ModelFile("indri/indri_lm.onnx",
                hf(INDRI, "indri_lm_int8.onnx"), 358_050_633L),
            ModelFile("indri/vocab.json",
                hf(INDRI, "vocab.json"), 798_156L, Kind.JSON),
            ModelFile("indri/merges.txt",
                hf(INDRI, "merges.txt"), 456_318L, Kind.TEXT),
            ModelFile("indri/added_tokens.json",
                hf(INDRI, "added_tokens.json"), 383_940L, Kind.JSON),
            // fp16, not int8: int8's own error (15.6 dB) is worse than the
            // padding it would be hiding.
            ModelFile("indri/mimi_decoder.onnx",
                hf(MIMI, "onnx/decoder_model_fp16.onnx"), 114_242_965L)
        ),
        runtimeRamBytes = 900_000_000L
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

    val ALL = listOf(
        CAMPPLUS, SENSEVOICE, EMOTION2VEC, NLLB,
        CHATTERBOX_Q4, CHATTERBOX_MIX, DHVAANI_TTS, INDRI_TTS, CHATTERBOX_HI
    )

    /**
     * Selectable voice engines, in the order shown in Settings.
     *
     * Order is deliberate: the two cloning Chatterbox packs first, since they
     * are what the pipeline was designed around, then DhVaani (also cloning,
     * far smaller), then Indri last because it cannot clone at all.
     */
    val VOICE_ENGINES = listOf(CHATTERBOX_Q4, CHATTERBOX_MIX, DHVAANI_TTS, INDRI_TTS)

    /** Engines that reproduce the original speaker rather than a preset. */
    val CLONING_ENGINES = VOICE_ENGINES.filter { it.id != INDRI_TTS.id }

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
