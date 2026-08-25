package com.azim.vdub.audio

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * Byte-level BPE tokenizer for Chatterbox, reading HuggingFace `tokenizer.json`.
 *
 * Written by hand because the app has no `transformers` at run time. Only the
 * pieces Chatterbox actually uses are implemented:
 *
 *  - byte-level pre-tokenization (bytes -> printable proxy chars)
 *  - greedy BPE merges driven by the merge ranks in the file
 *  - the added/special tokens, matched before anything else so `[hi]` and the
 *    Cangjie `[cj_*]` codes stay single tokens
 *
 * Text is NFKD-normalised first, matching the export's preprocessing; skipping
 * that shifts Hindi matras onto the wrong tokens and the audio comes out
 * mispronounced rather than failing outright.
 */
class ChatterboxTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: Map<Pair<String, String>, Int>,
    private val addedTokens: Map<String, Int>,
    val bosId: Int,
    val eosId: Int
) {

    companion object {
        /** Speech-token range markers from the model card. */
        const val START_SPEECH_TOKEN = 6561
        const val STOP_SPEECH_TOKEN = 6562

        fun load(tokenizerJson: File): ChatterboxTokenizer {
            require(tokenizerJson.exists()) { "tokenizer.json not found" }
            val root = Json.parseToJsonElement(tokenizerJson.readText()).jsonObject

            val model = root["model"]?.jsonObject
                ?: error("tokenizer.json has no model section")

            val vocab = model["vocab"]?.jsonObject
                ?.mapValues { it.value.jsonPrimitive.int() }
                ?: error("tokenizer.json has no vocab")

            val merges = LinkedHashMap<Pair<String, String>, Int>()
            model["merges"]?.jsonArray?.forEachIndexed { rank, el ->
                // Either "a b" or ["a", "b"] depending on tokenizers version.
                val pair = if (el is kotlinx.serialization.json.JsonArray) {
                    el.jsonArray.map { it.jsonPrimitive.content }
                } else {
                    el.jsonPrimitive.content.split(' ', limit = 2)
                }
                if (pair.size == 2) merges[pair[0] to pair[1]] = rank
            }

            val added = LinkedHashMap<String, Int>()
            root["added_tokens"]?.jsonArray?.forEach { el ->
                val o = el.jsonObject
                val content = o["content"]?.jsonPrimitive?.content ?: return@forEach
                val id = o["id"]?.jsonPrimitive?.int() ?: return@forEach
                added[content] = id
            }

            return ChatterboxTokenizer(
                vocab = vocab,
                merges = merges,
                addedTokens = added,
                bosId = added["<s>"] ?: vocab["<s>"] ?: 1,
                eosId = added["</s>"] ?: vocab["</s>"] ?: 2
            )
        }

        private fun kotlinx.serialization.json.JsonPrimitive.int(): Int =
            content.toIntOrNull() ?: content.toDouble().toInt()

        /**
         * GPT-2 byte encoder: maps every byte to a printable character so BPE
         * can work on text while still round-tripping arbitrary UTF-8.
         */
        private val byteToUnicode: Map<Int, Char> = buildMap {
            val bs = mutableListOf<Int>()
            (33..126).forEach { bs.add(it) }
            (161..172).forEach { bs.add(it) }
            (174..255).forEach { bs.add(it) }
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            bs.indices.forEach { put(bs[it], cs[it].toChar()) }
        }
    }

    /** Longest-first so `[cj_ABC]` wins over any shorter added token. */
    private val addedSorted = addedTokens.keys.sortedByDescending { it.length }

    fun encode(text: String, addBos: Boolean = true, addEos: Boolean = true): IntArray {
        val normalised = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFKD)
        val out = ArrayList<Int>(normalised.length / 2 + 8)
        if (addBos) out.add(bosId)

        var i = 0
        val buffer = StringBuilder()

        fun flushBuffer() {
            if (buffer.isEmpty()) return
            out.addAll(encodeChunk(buffer.toString()))
            buffer.setLength(0)
        }

        while (i < normalised.length) {
            val hit = addedSorted.firstOrNull { normalised.startsWith(it, i) }
            if (hit != null) {
                flushBuffer()
                out.add(addedTokens.getValue(hit))
                i += hit.length
            } else {
                buffer.append(normalised[i])
                i++
            }
        }
        flushBuffer()

        if (addEos) out.add(eosId)
        return out.toIntArray()
    }

    /** Byte-level encode + BPE for a stretch with no special tokens. */
    private fun encodeChunk(chunk: String): List<Int> {
        if (chunk.isEmpty()) return emptyList()
        val proxy = buildString {
            chunk.toByteArray(Charsets.UTF_8).forEach { b ->
                append(byteToUnicode[b.toInt() and 0xFF])
            }
        }
        val ids = ArrayList<Int>()
        bpe(proxy).forEach { piece ->
            val id = vocab[piece]
            if (id != null) {
                ids.add(id)
            } else {
                // Unknown piece: fall back to single characters so one odd
                // glyph cannot drop an entire line.
                piece.forEach { ch -> vocab[ch.toString()]?.let(ids::add) }
            }
        }
        return ids
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)
        var parts = token.map { it.toString() }.toMutableList()

        while (parts.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (k in 0 until parts.size - 1) {
                val rank = merges[parts[k] to parts[k + 1]] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = k
                }
            }
            if (bestIdx < 0) break
            val merged = parts[bestIdx] + parts[bestIdx + 1]
            parts[bestIdx] = merged
            parts.removeAt(bestIdx + 1)
        }
        return parts
    }

    /**
     * Chatterbox expects a leading language tag, e.g. "[hi]नमस्ते".
     * Chinese additionally needs Cangjie codes, which is why the source-language
     * path is not used for synthesis — we always speak the translated text.
     */
    fun withLanguage(text: String, lang: String): String = "[${lang.lowercase()}]$text"
}
