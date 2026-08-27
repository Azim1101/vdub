package com.azim.vdub.audio

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * GPT-2 byte-level BPE for Indri, from `vocab.json` + `merges.txt`.
 *
 * Indri is GPT-2 with an enlarged vocabulary: the original 50257 text tokens,
 * then 16468 added tokens — the Mimi acoustic tokens and the control tokens
 * that frame a request. Those added tokens are matched literally, before BPE,
 * so `[spkr_69]` stays one id instead of being split into brackets and digits.
 *
 * The 6.6 MB `tokenizer.json` is deliberately not used: it duplicates
 * vocab+merges in a form that costs several times the memory to parse, and the
 * pieces needed here are the same either way.
 *
 * Two details taken from the real tokenizer's behaviour (verified against
 * `transformers` in `tools/probe_tts2.py`, and pinned in `IndriTokenizerTest`):
 *
 *  - GPT-2 has no normalizer, so text is fed through unchanged apart from the
 *    lowercasing Indri's own pipeline applies first.
 *  - the pre-tokenizer is ByteLevel with `add_prefix_space=false`, so a
 *    leading space is *not* inserted; " world" and "world" are different
 *    tokens and inserting one shifts every id.
 */
class IndriTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: Map<Pair<String, String>, Int>,
    private val added: Map<String, Int>
) {

    companion object {
        /** Audio tokens start here; everything below is text. */
        const val AUDIO_OFFSET = 50257

        fun load(vocabJson: File, mergesTxt: File, addedTokensJson: File): IndriTokenizer {
            require(vocabJson.exists()) { "vocab.json not found" }
            require(mergesTxt.exists()) { "merges.txt not found" }
            require(addedTokensJson.exists()) { "added_tokens.json not found" }

            val json = Json { ignoreUnknownKeys = true }

            val vocab = json.parseToJsonElement(vocabJson.readText()).jsonObject
                .mapValues { it.value.jsonPrimitive.content.toInt() }

            val merges = LinkedHashMap<Pair<String, String>, Int>()
            mergesTxt.useLines { lines ->
                var rank = 0
                lines.forEach { line ->
                    if (line.isEmpty() || line.startsWith("#version")) return@forEach
                    val sp = line.indexOf(' ')
                    if (sp <= 0) return@forEach
                    merges[line.substring(0, sp) to line.substring(sp + 1)] = rank++
                }
            }

            val added = json.parseToJsonElement(addedTokensJson.readText()).jsonObject
                .mapValues { it.value.jsonPrimitive.content.toInt() }

            check(vocab.isNotEmpty() && merges.isNotEmpty()) {
                "Indri tokenizer files look truncated " +
                    "(${vocab.size} vocab, ${merges.size} merges)"
            }
            return IndriTokenizer(vocab, merges, added)
        }

        /** GPT-2's byte -> printable-character map. */
        internal val byteToUnicode: Map<Int, Char> = buildMap {
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

    /** Longest first, so `[spkr_69]` is not shadowed by a shorter token. */
    private val addedSorted = added.keys.sortedByDescending { it.length }

    fun id(token: String): Int? = added[token] ?: vocab[token]

    /** Id for `[spkr_NN]`, or null when that speaker does not exist. */
    fun speakerId(speaker: String): Int? = added[speaker]

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length / 2 + 8)
        var i = 0
        val buffer = StringBuilder()

        fun flush() {
            if (buffer.isEmpty()) return
            out.addAll(encodeChunk(buffer.toString()))
            buffer.setLength(0)
        }

        while (i < text.length) {
            val hit = addedSorted.firstOrNull { text.startsWith(it, i) }
            if (hit != null) {
                flush()
                out.add(added.getValue(hit))
                i += hit.length
            } else {
                buffer.append(text[i])
                i++
            }
        }
        flush()
        return out.toIntArray()
    }

    private fun encodeChunk(chunk: String): List<Int> {
        if (chunk.isEmpty()) return emptyList()
        val ids = ArrayList<Int>()
        // GPT-2 splits on its own regex; the parts that matter here are that a
        // space attaches to the following word and punctuation stands alone.
        for (word in splitGpt2(chunk)) {
            val proxy = buildString {
                word.toByteArray(Charsets.UTF_8).forEach { b ->
                    append(byteToUnicode[b.toInt() and 0xFF])
                }
            }
            bpe(proxy).forEach { piece ->
                val id = vocab[piece]
                if (id != null) {
                    ids.add(id)
                } else {
                    // Fall back to single characters so one unknown glyph
                    // cannot drop a whole line.
                    piece.forEach { ch -> vocab[ch.toString()]?.let(ids::add) }
                }
            }
        }
        return ids
    }

    /**
     * GPT-2's pre-tokenizer pattern.
     *
     * Written out rather than applied as one regex because Kotlin's engine has
     * no `\p{L}`-with-negation form matching Python's, and getting this subtly
     * wrong shifts every id after the first mistake.
     */
    private fun splitGpt2(text: String): List<String> {
        val parts = ArrayList<String>()
        var i = 0
        while (i < text.length) {
            val start = i
            val c = text[i]

            // contractions: 's 't 're 've 'm 'll 'd
            if (c == '\'' && i + 1 < text.length) {
                val rest = text.substring(i)
                val match = CONTRACTIONS.firstOrNull { rest.startsWith(it) }
                if (match != null) {
                    parts.add(match)
                    i += match.length
                    continue
                }
            }

            // optional single leading space, then a run of one character class
            var j = i
            if (text[j] == ' ' && j + 1 < text.length && !text[j + 1].isWhitespace()) j++

            when {
                j < text.length && text[j].isLetter() -> {
                    while (j < text.length && text[j].isLetter()) j++
                }
                j < text.length && text[j].isDigit() -> {
                    while (j < text.length && text[j].isDigit()) j++
                }
                j < text.length && !text[j].isWhitespace() -> {
                    while (j < text.length && !text[j].isWhitespace() &&
                        !text[j].isLetter() && !text[j].isDigit()
                    ) j++
                }
                else -> {
                    // whitespace run; the last space belongs to the next word
                    while (j < text.length && text[j].isWhitespace()) j++
                    if (j < text.length && j - start > 1) j--
                }
            }
            if (j == start) j = start + 1
            parts.add(text.substring(start, j))
            i = j
        }
        return parts
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)
        val parts = token.map { it.toString() }.toMutableList()
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
            parts[bestIdx] = parts[bestIdx] + parts[bestIdx + 1]
            parts.removeAt(bestIdx + 1)
        }
        return parts
    }
}

private val CONTRACTIONS = listOf("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
