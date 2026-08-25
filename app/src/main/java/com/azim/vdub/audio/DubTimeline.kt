package com.azim.vdub.audio

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Places spoken clips back on the original timeline.
 *
 * Hindi rarely takes the same time as the source line, so each clip is fitted
 * to its slot before being laid down. Two knobs, in order of preference:
 *
 *  1. speed — up to [MAX_SPEED]; a small tempo change is far less noticeable
 *     than a line that talks over the next one
 *  2. overflow — if it still does not fit, the line is allowed to run past its
 *     slot rather than being chopped mid-word, and the overrun is reported
 *
 * Pitch is preserved: resampling alone would chipmunk the voice, so speed is
 * applied by overlap-add (WSOLA-lite), which repeats or drops whole pitch
 * periods instead of rescaling the waveform.
 */
object DubTimeline {

    const val MAX_SPEED = 2.0f
    const val MIN_SPEED = 0.8f

    data class Placement(
        val lineId: Int,
        val startSec: Double,
        val naturalSec: Double,
        val slotSec: Double,
        val speed: Float,
        val overflowSec: Double
    ) {
        val fitted: Boolean get() = overflowSec <= 0.01
    }

    data class Result(
        val samples: FloatArray,
        val sampleRate: Int,
        val placements: List<Placement>
    ) {
        val overflowing: List<Placement> get() = placements.filter { !it.fitted }
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    data class Clip(
        val lineId: Int,
        val startSec: Double,
        val endSec: Double,
        val samples: FloatArray
    ) {
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    /**
     * Lay [clips] onto a silent track of [totalSec].
     *
     * Clips are placed in time order; where a fitted clip would still collide
     * with the next one, the later clip is nudged rather than overwritten, so
     * dialogue never doubles up.
     */
    fun assemble(
        clips: List<Clip>,
        totalSec: Double,
        sampleRate: Int
    ): Result {
        val total = (totalSec * sampleRate).toInt().coerceAtLeast(1)
        val track = FloatArray(total)
        val placements = ArrayList<Placement>(clips.size)

        val ordered = clips.sortedBy { it.startSec }
        var cursorSample = 0

        ordered.forEachIndexed { index, clip ->
            val slotSec = (clip.endSec - clip.startSec).coerceAtLeast(0.05)
            val naturalSec = clip.samples.size.toDouble() / sampleRate
            if (clip.samples.isEmpty()) return@forEachIndexed

            // Speed needed to fit the slot, clamped to what stays natural.
            val wanted = (naturalSec / slotSec).toFloat()
            val speed = wanted.coerceIn(MIN_SPEED, MAX_SPEED)
            val fitted = if (abs(speed - 1f) < 0.02f) clip.samples
            else timeStretch(clip.samples, speed)

            val fittedSec = fitted.size.toDouble() / sampleRate
            val overflow = (fittedSec - slotSec).coerceAtLeast(0.0)

            var start = (clip.startSec * sampleRate).roundToInt().coerceIn(0, total - 1)
            // Never let a long line stamp over the start of the next one.
            if (start < cursorSample) start = cursorSample.coerceAtMost(total - 1)

            val n = minOf(fitted.size, total - start)
            if (n > 0) {
                fitted.copyInto(track, start, 0, n)
                cursorSample = start + n
            }

            placements += Placement(
                lineId = clip.lineId,
                startSec = start.toDouble() / sampleRate,
                naturalSec = naturalSec,
                slotSec = slotSec,
                speed = speed,
                overflowSec = overflow
            )
        }

        return Result(track, sampleRate, placements)
    }

    /**
     * Change duration by [speed] without changing pitch.
     *
     * Overlap-add with a fixed frame and a cross-faded join: frames are read at
     * `speed` intervals and written at a constant hop, so periodicity — and
     * therefore pitch — is preserved.
     */
    fun timeStretch(input: FloatArray, speed: Float): FloatArray {
        if (input.isEmpty() || abs(speed - 1f) < 0.01f) return input
        val s = speed.coerceIn(0.25f, 4f)

        val frame = 1024
        val hop = frame / 2
        val analysisHop = (hop * s).toInt().coerceAtLeast(1)
        val outLen = (input.size / s).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        val window = FloatArray(frame) { i ->
            (0.5 - 0.5 * kotlin.math.cos(2.0 * Math.PI * i / (frame - 1))).toFloat()
        }
        val norm = FloatArray(outLen)

        var readPos = 0
        var writePos = 0
        while (writePos + frame <= outLen && readPos + frame <= input.size) {
            for (i in 0 until frame) {
                val w = window[i]
                out[writePos + i] += input[readPos + i] * w
                norm[writePos + i] += w
            }
            readPos += analysisHop
            writePos += hop
        }
        // Undo the window sum so steady passages keep their level.
        for (i in out.indices) {
            if (norm[i] > 1e-4f) out[i] /= norm[i]
        }
        return out
    }

    /** Mix the dubbed track over a background bed. */
    fun mix(
        dubbed: FloatArray,
        background: FloatArray,
        backgroundGain: Float = 0.35f
    ): FloatArray {
        val n = maxOf(dubbed.size, background.size)
        val out = FloatArray(n)
        for (i in 0 until n) {
            val a = if (i < dubbed.size) dubbed[i] else 0f
            val b = if (i < background.size) background[i] * backgroundGain else 0f
            out[i] = (a + b).coerceIn(-1f, 1f)
        }
        return out
    }
}
