package com.azim.vdub.audio

import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import java.io.File

/**
 * Resolves the files for whichever voice engine is selected, and opens it.
 *
 * Four engines sit behind [TtsEngine], and they share nothing structurally:
 *
 *  - `chatterbox_q4` / `chatterbox_mix` — four graphs, autoregressive speech
 *    tokens, zero-shot cloning
 *  - `dhvaani` — three graphs plus two numpy archives, flow matching, cloning
 *  - `indri` — a GPT-2 graph plus a separate Mimi codec decoder, preset voices
 *
 * Filenames come from the catalog rather than being written out again here, so
 * a catalog change cannot silently desync from what the loader looks for. The
 * lookup is by leaf name, which also means the two Chatterbox packs can keep
 * colliding filenames in separate folders.
 */
object VoiceEngine {

    /** Chatterbox sampling settings. Both packs need them. */
    const val REPETITION_PENALTY = 1.2f   // 2.0 makes the quantized LLM loop
    const val TEMPERATURE = 0.8f
    const val TOP_P = 0.95f
    const val MIN_P = 0.05f

    /** How an engine's files are laid out and which class runs them. */
    enum class Kind { CHATTERBOX, DHVAANI, INDRI }

    fun kindOf(id: String): Kind = when (id) {
        ModelCatalog.DHVAANI_TTS.id -> Kind.DHVAANI
        ModelCatalog.INDRI_TTS.id -> Kind.INDRI
        else -> Kind.CHATTERBOX
    }

    /**
     * Resolved on-disk paths.
     *
     * Chatterbox's four graphs are named fields because they are the original
     * shape of this class; the other engines' files are read through accessors
     * over the same [files] map. Every engine reports [missing] the same way,
     * which is what the UI shows.
     */
    data class Paths(
        val model: ModelCatalog.Model,
        val kind: Kind,
        /** Every catalog file of this engine, keyed by leaf name. */
        val files: Map<String, File>,
        val embedTokens: File,
        val languageModel: File,
        val speechEncoder: File,
        val conditionalDecoder: File,
        val tokenizer: File,
        val cangjie: File,
        val defaultVoice: File
    ) {
        /** Files that must exist for this engine to open. */
        val required: List<File>
            get() = when (kind) {
                Kind.CHATTERBOX ->
                    listOf(embedTokens, languageModel, speechEncoder,
                        conditionalDecoder, tokenizer)
                Kind.DHVAANI ->
                    listOf(textEncoder(), fmDecoder(), vocoderBackbone(),
                        vocosHead(), melFilterbank(), tokensTxt())
                Kind.INDRI ->
                    listOf(languageModel, mimiDecoder(), vocabJson(),
                        mergesTxt(), addedTokensJson())
            }

        /** Graphs opened at run time, for the Chatterbox-shaped callers. */
        val graphs: List<File>
            get() = required.filter { it.name.endsWith(".onnx") }

        val missing: List<File>
            get() = required.filter { !it.exists() || it.length() == 0L }

        val installed: Boolean get() = missing.isEmpty()

        private fun of(leaf: String): File = files[leaf]
            ?: File(VdubPaths.modelsDir, "${model.id}/$leaf")

        // DhVaani
        fun textEncoder() = of("text_encoder.onnx")
        fun fmDecoder() = of("fm_decoder.onnx")
        fun vocoderBackbone() = of("vocoder_backbone.onnx")
        fun vocosHead() = of("vocos_head.npz")
        fun melFilterbank() = of("mel_fb.npz")
        fun tokensTxt() = of("tokens.txt")

        // Indri
        fun mimiDecoder() = of("mimi_decoder.onnx")
        fun vocabJson() = of("vocab.json")
        fun mergesTxt() = of("merges.txt")
        fun addedTokensJson() = of("added_tokens.json")
    }

    fun byId(id: String): ModelCatalog.Model =
        ModelCatalog.VOICE_ENGINES.firstOrNull { it.id == id }
            ?: ModelCatalog.VOICE_ENGINES.first()

    /**
     * Catalog-relative names for a pack, e.g. "chatterbox_mix/language_model_q4.onnx".
     *
     * Kept free of Android APIs so the naming rules — which is what actually
     * differs between the engines — are unit testable without a device.
     */
    data class Names(
        val embedTokens: String,
        val languageModel: String,
        val speechEncoder: String,
        val conditionalDecoder: String,
        val tokenizer: String,
        val cangjie: String,
        val defaultVoice: String
    ) {
        val graphs: List<String>
            get() = listOf(embedTokens, languageModel, speechEncoder, conditionalDecoder)
    }

    fun namesFor(id: String): Names {
        val model = byId(id)
        val all = model.files.map { it.localName }
        val folder = folderOf(model)
        fun leaf(exact: String) = all.firstOrNull { it.substringAfterLast('/') == exact }
            ?: "$folder/$exact"
        return Names(
            embedTokens = leaf("embed_tokens.onnx"),
            // the mix pack ships language_model_q4.onnx; its sidecar reference
            // is baked into the graph, so the upstream name must survive
            languageModel = all.firstOrNull {
                val n = it.substringAfterLast('/')
                n.startsWith("language_model") && n.endsWith(".onnx")
            }
            // Indri's is named for the model rather than its role
                ?: all.firstOrNull { it.substringAfterLast('/') == "indri_lm.onnx" }
                ?: "$folder/language_model.onnx",
            speechEncoder = leaf("speech_encoder.onnx"),
            conditionalDecoder = leaf("conditional_decoder.onnx"),
            tokenizer = leaf("tokenizer.json"),
            cangjie = leaf("Cangjie5_TC.json"),
            defaultVoice = leaf("default_voice.wav")
        )
    }

    /**
     * Resolve on-disk paths for [id].
     */
    fun pathsFor(id: String): Paths {
        val model = byId(id)
        val n = namesFor(id)
        val root = VdubPaths.modelsDir
        return Paths(
            model = model,
            kind = kindOf(id),
            files = model.files.associate {
                it.localName.substringAfterLast('/') to File(root, it.localName)
            },
            embedTokens = File(root, n.embedTokens),
            languageModel = File(root, n.languageModel),
            speechEncoder = File(root, n.speechEncoder),
            conditionalDecoder = File(root, n.conditionalDecoder),
            tokenizer = File(root, n.tokenizer),
            cangjie = File(root, n.cangjie),
            defaultVoice = File(root, n.defaultVoice)
        )
    }

    private fun folderOf(model: ModelCatalog.Model): String =
        model.files.first().localName.substringBeforeLast('/')

    /** True when every file of [id] is present on disk. */
    fun isInstalled(id: String): Boolean = pathsFor(id).installed

    /**
     * Open the selected engine.
     *
     * The single place that maps an id to an implementation — callers hold a
     * [TtsEngine] and never learn which one they got.
     */
    fun open(id: String, threads: Int = 4): TtsEngine {
        val paths = pathsFor(id)
        return when (paths.kind) {
            Kind.CHATTERBOX -> ChatterboxTts.open(paths, threads)
            Kind.DHVAANI -> DhVaaniTts.open(paths, threads)
            Kind.INDRI -> IndriTts.open(paths, threads)
        }
    }
}
