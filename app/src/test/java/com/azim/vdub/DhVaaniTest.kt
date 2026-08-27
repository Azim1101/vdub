package com.azim.vdub

import com.azim.vdub.audio.DhVaaniTts
import com.azim.vdub.audio.forwardFft
import com.azim.vdub.audio.inverseFft
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * DhVaani's signal-processing constants and the FFT its vocoder rests on.
 *
 * The engine's audio path is hand-written, and everything in it fails quietly:
 * a wrong hop produces audio at the wrong speed, a missing hermitian mirror
 * produces a half-amplitude buzz, and a wrong feature scale produces noise —
 * none of them throw.
 *
 * Values verified against the reference implementation in
 * `tools/probe_tts6.py`, which read them from the shipped model files.
 */
class DhVaaniTest {

    /**
     * Read from `model.json` and `vocos_head.npz` upstream. The vocoder is
     * built for exactly these; changing one without the others silently
     * detunes every mel bin.
     */
    @Test
    fun `stft constants match the shipped model`() {
        assertEquals(24_000, DhVaaniTts.SAMPLE_RATE)
        assertEquals(1024, DhVaaniTts.N_FFT)
        assertEquals(256, DhVaaniTts.HOP)
        assertEquals(100, DhVaaniTts.N_MELS)
    }

    /**
     * Features are scaled by 0.1 on the way in and unscaled on the way out.
     * The flow was trained in that range; at 1.0 it starts far outside its
     * trained distribution and returns noise rather than failing.
     */
    @Test
    fun `feature scale is the trained one`() {
        assertEquals(0.1f, DhVaaniTts.FEAT_SCALE, 1e-9f)
        assertEquals(0.1f, DhVaaniTts.TARGET_RMS, 1e-9f)
    }

    /**
     * Step count is a direct time/quality trade: measured RTF 0.84 at 4 steps,
     * 1.64 at 8, 3.26 at 16 on two cores. Eight keeps a 190-line project
     * inside a sensible wall-clock while staying clearly better than four.
     */
    @Test
    fun `sampling steps stay in the useful range`() {
        assertTrue(DhVaaniTts.DEFAULT_STEPS in 4..16)
        assertEquals(0.5f, DhVaaniTts.T_SHIFT, 1e-9f)
    }

    /** 24 kHz at hop 256 is 93.75 frames a second — what timing fit assumes. */
    @Test
    fun `frame rate follows from hop and sample rate`() {
        val framesPerSecond = DhVaaniTts.SAMPLE_RATE.toDouble() / DhVaaniTts.HOP
        assertEquals(93.75, framesPerSecond, 1e-9)
    }

    // ------------------------------------------------------------------ FFT

    /** A pure tone must land in exactly its own bin, at the right magnitude. */
    @Test
    fun `forward fft places a tone in the correct bin`() {
        val n = 1024
        val bin = 64
        val re = FloatArray(n) { cos(2.0 * PI * bin * it / n).toFloat() }
        val im = FloatArray(n)
        forwardFft(re, im)

        val magnitudes = FloatArray(n / 2 + 1) {
            kotlin.math.sqrt(re[it] * re[it] + im[it] * im[it])
        }
        assertEquals(bin, magnitudes.indices.maxByOrNull { magnitudes[it] })
        // a real cosine splits its energy across +f and -f, so the half
        // spectrum holds n/2
        assertEquals(n / 2.0, magnitudes[bin].toDouble(), n / 200.0)
        // and nothing meaningful anywhere else
        magnitudes.forEachIndexed { i, m ->
            if (i != bin) assertTrue("leak into bin $i: $m", m < 1e-2f)
        }
    }

    /** Inverse must undo forward, including the 1/n scale. */
    @Test
    fun `inverse fft round trips`() {
        val n = 256
        val original = FloatArray(n) {
            (sin(2.0 * PI * 5 * it / n) + 0.3 * cos(2.0 * PI * 33 * it / n)).toFloat()
        }
        val re = original.copyOf()
        val im = FloatArray(n)

        forwardFft(re, im)
        inverseFft(re, im)

        for (i in 0 until n) {
            assertEquals("sample $i", original[i], re[i], 1e-4f)
            assertEquals("imag $i", 0f, im[i], 1e-4f)
        }
    }

    /**
     * The vocoder predicts only the half spectrum, so the ISTFT mirrors it
     * back. A correctly mirrored hermitian spectrum inverts to a purely real
     * signal — if the mirror is missing or mis-signed, the imaginary part
     * survives, which is the buzz that would otherwise be discovered by ear.
     */
    @Test
    fun `hermitian spectrum inverts to a real signal`() {
        val n = 512
        val bins = n / 2 + 1
        val magnitude = FloatArray(bins) { 1f / (it + 1) }
        val phase = FloatArray(bins) { 0.3f * it }

        val re = FloatArray(n)
        val im = FloatArray(n)
        for (k in 0 until bins) {
            val r = magnitude[k] * cos(phase[k].toDouble()).toFloat()
            val i = magnitude[k] * sin(phase[k].toDouble()).toFloat()
            re[k] = r
            im[k] = i
            if (k in 1 until n / 2) {
                re[n - k] = r
                im[n - k] = -i
            }
        }
        im[0] = 0f
        im[n / 2] = 0f

        inverseFft(re, im)

        val maxImag = im.maxOf { abs(it) }
        assertTrue("residual imaginary part $maxImag", maxImag < 1e-5f)
        assertTrue("output is silent", re.maxOf { abs(it) } > 1e-3f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `fft rejects a non power of two length`() {
        forwardFft(FloatArray(100), FloatArray(100))
    }

    /**
     * Overlap-add divided by the summed squared window reconstructs exactly.
     * This is the property the ISTFT relies on; without the division the
     * output has a periodic tremor at the hop rate.
     *
     * Verified numerically against the reference: with the analysis padding
     * aligned to the synthesis trim, a cosine round-trips to 1.8e-7.
     */
    @Test
    fun `hann window overlap adds to a constant`() {
        val n = DhVaaniTts.N_FFT
        val hop = DhVaaniTts.HOP
        val window = FloatArray(n) { (0.5 - 0.5 * cos(2.0 * PI * it / n)).toFloat() }

        val frames = 16
        val envelope = FloatArray((frames - 1) * hop + n)
        for (f in 0 until frames) {
            for (i in 0 until n) envelope[f * hop + i] += window[i] * window[i]
        }

        // Away from the ramp-in and ramp-out the envelope must be flat, or
        // the division would impose its own amplitude modulation.
        val from = n
        val to = envelope.size - n
        val reference = envelope[from]
        for (i in from until to) {
            assertEquals("envelope at $i", reference, envelope[i], 1e-4f)
        }
        assertTrue("envelope should be non-zero", reference > 0.1f)
    }
}
