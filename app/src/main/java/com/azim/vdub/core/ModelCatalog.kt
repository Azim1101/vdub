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
    data class Model(
        val id: String,
        val name: String,
        val stage: Stage,
        val sizeBytes: Long,
        val description: String,
        val files: List<ModelFile>,
        val required: Boolean = true,
        val license: String = "",
        val note: String = ""
    ) {
        val sizeMb: Int get() = (sizeBytes / 1024 / 1024).toInt()
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

    val KOKORO = Model(
        id = "kokoro",
        name = "Kokoro-82M TTS",
        stage = Stage.TTS,
        sizeBytes = 92_000_000L,
        description = "Speaks the translated lines. Includes Hindi and Chinese " +
            "voices. Small enough to run comfortably on a phone.",
        license = "Apache-2.0",
        note = "Preset voices — not a clone of the original actor.",
        files = listOf(
            ModelFile(
                localName = "kokoro.onnx",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "onnx/model_q8f16.onnx"),
                approxBytes = 92_000_000L
            ),
            ModelFile(
                localName = "kokoro_tokenizer.json",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "tokenizer.json"),
                approxBytes = 500_000L,
                kind = Kind.JSON
            ),
            // Hindi voices
            ModelFile(
                localName = "voices/hf_alpha.bin",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/hf_alpha.bin"),
                approxBytes = 524_288L,
                kind = Kind.BIN
            ),
            ModelFile(
                localName = "voices/hm_omega.bin",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/hm_omega.bin"),
                approxBytes = 524_288L,
                kind = Kind.BIN
            ),
            ModelFile(
                localName = "voices/hf_beta.bin",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/hf_beta.bin"),
                approxBytes = 524_288L,
                kind = Kind.BIN
            ),
            ModelFile(
                localName = "voices/hm_psi.bin",
                urls = hf("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/hm_psi.bin"),
                approxBytes = 524_288L,
                kind = Kind.BIN
            )
        )
    )

    val ALL = listOf(CAMPPLUS, SENSEVOICE, EMOTION2VEC, NLLB, KOKORO)

    fun byId(id: String): Model? = ALL.firstOrNull { it.id == id }

    fun forStage(stage: Stage): List<Model> = ALL.filter { it.stage == stage }

    /** Total disk if the user downloads everything. */
    val totalBytes: Long get() = ALL.sumOf { it.sizeBytes }

    /** Peak RAM is the biggest single model, since stages run one at a time. */
    val peakRamBytes: Long get() = ALL.maxOf { it.sizeBytes }
}
