package com.azim.vdub.subtitle

import com.azim.vdub.data.model.ScriptLine
import com.azim.vdub.data.model.SrtCue
import java.io.File
import java.io.InputStream

/**
 * SRT parsing + cue merging.
 *
 * The spec's real numbers: 473 cues collapse to 190 dialogue lines. Raw cues
 * are display chunks (a sentence gets split across 2-3 cards); TTS needs whole
 * utterances, so adjacent cues are merged while the gap is small and the
 * sentence hasn't terminated.
 */
object SrtParser {

    private val TIME_RE = Regex(
        """(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"""
    )
    private val TAG_RE = Regex("""</?[a-zA-Z][^>]*>""")
    private val BRACKET_RE = Regex("""^\s*[\[(][^\])]*[\])]\s*:?\s*""")

    fun parse(file: File): List<SrtCue> = file.inputStream().use { parse(it) }

    fun parse(input: InputStream): List<SrtCue> {
        val text = input.readBytes().decodeToString().removePrefix("\uFEFF")
        val cues = mutableListOf<SrtCue>()
        var index = 0

        text.split(Regex("\\r?\\n\\s*\\r?\\n")).forEach { block ->
            val lines = block.trim().lines().filter { it.isNotBlank() }
            if (lines.isEmpty()) return@forEach
            val timeLineIdx = lines.indexOfFirst { TIME_RE.containsMatchIn(it) }
            if (timeLineIdx < 0) return@forEach
            val m = TIME_RE.find(lines[timeLineIdx]) ?: return@forEach
            val g = m.groupValues
            val start = toMs(g[1], g[2], g[3], g[4])
            val end = toMs(g[5], g[6], g[7], g[8])
            val body = lines.drop(timeLineIdx + 1)
                .joinToString(" ") { it.trim() }
                .let { TAG_RE.replace(it, "") }
                .let { BRACKET_RE.replace(it, "") }
                .replace(Regex("\\s+"), " ")
                .trim()
            if (body.isNotEmpty() && end > start) {
                cues += SrtCue(index++, start, end, body)
            }
        }
        return cues
    }

    /**
     * Merge cues into speakable lines.
     *
     * @param maxGapMs      join cues separated by less than this
     * @param maxDurationMs never build a line longer than this
     * @param maxChars      never build a line wordier than this
     */
    fun mergeToLines(
        cues: List<SrtCue>,
        maxGapMs: Long = 400,
        maxDurationMs: Long = 15_000,
        maxChars: Int = 220
    ): List<ScriptLine> {
        if (cues.isEmpty()) return emptyList()
        val sorted = cues.sortedBy { it.startMs }
        val out = mutableListOf<ScriptLine>()

        var startMs = sorted.first().startMs
        var endMs = sorted.first().endMs
        var buf = StringBuilder(sorted.first().text)

        fun flush() {
            val text = buf.toString().replace(Regex("\\s+"), " ").trim()
            if (text.isNotEmpty()) {
                val id = out.size
                out += ScriptLine(
                    id = id,
                    start = startMs / 1000.0,
                    end = endMs / 1000.0,
                    text = text,
                    clip = "clips/line_%04d.wav".format(id)
                )
            }
        }

        for (i in 1 until sorted.size) {
            val cue = sorted[i]
            val gap = cue.startMs - endMs
            val wouldBeDuration = cue.endMs - startMs
            val wouldBeChars = buf.length + 1 + cue.text.length
            val terminated = buf.endsWith(".") || buf.endsWith("?") || buf.endsWith("!") ||
                buf.endsWith("।") || buf.endsWith("。") || buf.endsWith("？") ||
                buf.endsWith("！")

            val canMerge = gap in 0..maxGapMs &&
                !terminated &&
                wouldBeDuration <= maxDurationMs &&
                wouldBeChars <= maxChars

            if (canMerge) {
                buf.append(' ').append(cue.text)
                endMs = maxOf(endMs, cue.endMs)
            } else {
                flush()
                startMs = cue.startMs
                endMs = cue.endMs
                buf = StringBuilder(cue.text)
            }
        }
        flush()
        return out
    }

    /** Render lines back to SRT — used by "Download for Manual" translation. */
    fun toSrt(lines: List<ScriptLine>, useTranslated: Boolean = false): String =
        buildString {
            lines.forEachIndexed { i, line ->
                append(i + 1).append('\n')
                append(fmt(line.start)).append(" --> ").append(fmt(line.end)).append('\n')
                val body = if (useTranslated) line.translated.orEmpty().ifBlank { line.text }
                else line.text
                append(body).append("\n\n")
            }
        }

    private fun fmt(sec: Double): String {
        val totalMs = (sec * 1000).toLong().coerceAtLeast(0)
        val h = totalMs / 3_600_000
        val m = (totalMs % 3_600_000) / 60_000
        val s = (totalMs % 60_000) / 1000
        val ms = totalMs % 1000
        return "%02d:%02d:%02d,%03d".format(h, m, s, ms)
    }

    private fun toMs(h: String, m: String, s: String, frac: String): Long {
        val msPart = frac.padEnd(3, '0').take(3).toLong()
        return h.toLong() * 3_600_000 + m.toLong() * 60_000 + s.toLong() * 1000 + msPart
    }
}
