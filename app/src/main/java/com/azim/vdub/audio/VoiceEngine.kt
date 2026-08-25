package com.azim.vdub.audio

import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import java.io.File

/**
 * Resolves the files for whichever Chatterbox pack is selected.
 *
 * Both packs run the identical four-graph pipeline with the same session API
 * and the same input/output names:
 *
 *   embed_tokens -> language_model -> speech_encoder -> conditional_decoder
 *
 * Only two things differ, and both are handled here so the inference code
 * never branches on the pack:
 *
 *  - the language-model filename (`language_model.onnx` in the Q4 pack,
 *    `language_model_q4.onnx` in the mix pack — the mix graph references its
 *    own .onnx_data sidecar by that exact name, so it cannot be renamed)
 *  - the folder, since the two packs use colliding filenames for entirely
 *    different content
 */
object VoiceEngine {

    /** Sampling settings shared by both packs. */
    const val REPETITION_PENALTY = 1.2f   // 2.0 makes the quantized LLM loop
    const val TEMPERATURE = 0.8f
    const val TOP_P = 0.95f
    const val MIN_P = 0.05f

    data class Paths(
        val model: ModelCatalog.Model,
        val embedTokens: File,
        val languageModel: File,
        val speechEncoder: File,
        val conditionalDecoder: File,
        val tokenizer: File,
        val cangjie: File,
        val defaultVoice: File
    ) {
        /** Every graph the pipeline opens, in run order. */
        val graphs: List<File>
            get() = listOf(embedTokens, languageModel, speechEncoder, conditionalDecoder)

        val missing: List<File>
            get() = (graphs + tokenizer).filter { !it.exists() || it.length() == 0L }

        val installed: Boolean get() = missing.isEmpty()
    }

    fun byId(id: String): ModelCatalog.Model =
        ModelCatalog.VOICE_ENGINES.firstOrNull { it.id == id }
            ?: ModelCatalog.VOICE_ENGINES.first()

    /**
     * Catalog-relative names for a pack, e.g. "chatterbox_mix/language_model_q4.onnx".
     *
     * Kept free of Android APIs so the naming rules — which is what actually
     * differs between the packs — are unit testable without a device.
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
        fun leaf(exact: String) = all.firstOrNull { it.substringAfterLast('/') == exact }
            ?: "${folderOf(model)}/$exact"
        return Names(
            embedTokens = leaf("embed_tokens.onnx"),
            // the mix pack ships language_model_q4.onnx; its sidecar reference
            // is baked into the graph, so the upstream name must survive
            languageModel = all.first {
                val n = it.substringAfterLast('/')
                n.startsWith("language_model") && n.endsWith(".onnx")
            },
            speechEncoder = leaf("speech_encoder.onnx"),
            conditionalDecoder = leaf("conditional_decoder.onnx"),
            tokenizer = leaf("tokenizer.json"),
            cangjie = leaf("Cangjie5_TC.json"),
            defaultVoice = leaf("default_voice.wav")
        )
    }

    /**
     * Resolve on-disk paths for [id]. Filenames come from the catalog rather
     * than being hardcoded, so a catalog change cannot silently desync from
     * what the loader looks for.
     */
    fun pathsFor(id: String): Paths {
        val n = namesFor(id)
        val root = VdubPaths.modelsDir
        return Paths(
            model = byId(id),
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

    /** True when every graph of [id] is present on disk. */
    fun isInstalled(id: String): Boolean = pathsFor(id).installed
}
