package com.azim.vdub.audio

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * Kaldi-compatible 80-dim log-Mel filterbank.
 *
 * campplus does NOT take a waveform. Its ONNX graph starts at the features,
 * so the input is (batch, frames, 80) fbank computed exactly the way the model
 * was trained — 25 ms window, 10 ms shift, Povey window, HTK mel scale, then
 * mean normalisation (CMN) over the utterance. Getting any of that wrong
 * yields embeddings that look plausible but cluster into noise.
 *
 * Reference: kaldi feature-fbank.cc / mel-computations.cc, mirrored by
 * kaldi_native_fbank which the published campplus usage examples rely on.
 */
object Fbank {

    const val NUM_BINS = 80
    const val SAMPLE_RATE = 16_000
    private const val FRAME_LENGTH_MS = 25.0
    private const val FRAME_SHIFT_MS = 10.0
    private const val PREEMPH = 0.97f
    private const val EPS = 1.1920928955078125e-7   // FLT_EPSILON, kaldi's log floor

    val frameLength = (SAMPLE_RATE * FRAME_LENGTH_MS / 1000).toInt()   // 400
    val frameShift = (SAMPLE_RATE * FRAME_SHIFT_MS / 1000).toInt()     // 160
    private val fftSize = run {
        var n = 1
        while (n < frameLength) n = n shl 1
        n                                                              // 512
    }

    /** Povey window: hann^0.85, kaldi's default for fbank. */
    private val window: FloatArray = FloatArray(frameLength) { i ->
        val hann = 0.5 - 0.5 * cos(2.0 * Math.PI * i / (frameLength - 1))
        Math.pow(hann, 0.85).toFloat()
    }

    /** Triangular mel bins over the power spectrum, precomputed once. */
    private data class MelBin(val offset: Int, val weights: FloatArray)

    private val melBins: Array<MelBin> = buildMelBins()

    private fun hzToMel(hz: Double) = 1127.0 * ln(1.0 + hz / 700.0)
    private fun melToHz(mel: Double) = 700.0 * (exp(mel / 1127.0) - 1.0)

    private fun buildMelBins(): Array<MelBin> {
        val numFftBins = fftSize / 2
        val nyquist = SAMPLE_RATE / 2.0
        val lowFreq = 20.0
        val highFreq = nyquist                       // kaldi high_freq=0 -> nyquist
        val melLow = hzToMel(lowFreq)
        val melHigh = hzToMel(highFreq)
        val melDelta = (melHigh - melLow) / (NUM_BINS + 1)
        val fftBinWidth = SAMPLE_RATE.toDouble() / fftSize

        return Array(NUM_BINS) { bin ->
            val leftMel = melLow + bin * melDelta
            val centerMel = melLow + (bin + 1) * melDelta
            val rightMel = melLow + (bin + 2) * melDelta

            var firstIndex = -1
            val vals = ArrayList<Float>()
            for (i in 0 until numFftBins) {
                val mel = hzToMel(fftBinWidth * i)
                if (mel > leftMel && mel < rightMel) {
                    val w = if (mel <= centerMel) {
                        (mel - leftMel) / (centerMel - leftMel)
                    } else {
                        (rightMel - mel) / (rightMel - centerMel)
                    }
                    if (firstIndex < 0) firstIndex = i
                    vals.add(w.toFloat())
                }
            }
            MelBin(
                offset = if (firstIndex < 0) 0 else firstIndex,
                weights = vals.toFloatArray()
            )
        }
    }

    /** Number of frames kaldi produces for [numSamples] with snip_edges=true. */
    fun numFrames(numSamples: Int): Int =
        if (numSamples < frameLength) 0
        else 1 + (numSamples - frameLength) / frameShift

    /**
     * @param samples mono PCM in [-1, 1] scaled to int16 range (kaldi works on
     *                the raw ±32768 scale; the published campplus example does
     *                `audio * 32768`).
     * @param applyCmn subtract the per-utterance mean, as campplus expects.
     * @return [frames][80] log-mel energies.
     */
    fun compute(samples: FloatArray, applyCmn: Boolean = true): Array<FloatArray> {
        val frames = numFrames(samples.size)
        if (frames <= 0) return emptyArray()

        val out = Array(frames) { FloatArray(NUM_BINS) }
        val re = FloatArray(fftSize)
        val im = FloatArray(fftSize)
        val buf = FloatArray(frameLength)

        for (f in 0 until frames) {
            val start = f * frameShift
            // copy + remove DC offset
            var mean = 0f
            for (i in 0 until frameLength) {
                buf[i] = samples[start + i]
                mean += buf[i]
            }
            mean /= frameLength
            for (i in 0 until frameLength) buf[i] -= mean

            // pre-emphasis, applied in place from the tail so x[i-1] is original
            for (i in frameLength - 1 downTo 1) {
                buf[i] -= PREEMPH * buf[i - 1]
            }
            buf[0] -= PREEMPH * buf[0]

            // window + zero pad
            java.util.Arrays.fill(re, 0f)
            java.util.Arrays.fill(im, 0f)
            for (i in 0 until frameLength) re[i] = buf[i] * window[i]

            fft(re, im)

            // power spectrum, then triangular mel bins, then log
            val row = out[f]
            for (b in 0 until NUM_BINS) {
                val mb = melBins[b]
                var acc = 0f
                for (k in mb.weights.indices) {
                    val idx = mb.offset + k
                    val power = re[idx] * re[idx] + im[idx] * im[idx]
                    acc += mb.weights[k] * power
                }
                row[b] = ln(maxOf(acc.toDouble(), EPS)).toFloat()
            }
        }

        if (applyCmn) {
            val mean = FloatArray(NUM_BINS)
            for (row in out) for (b in 0 until NUM_BINS) mean[b] += row[b]
            for (b in 0 until NUM_BINS) mean[b] /= frames
            for (row in out) for (b in 0 until NUM_BINS) row[b] -= mean[b]
        }
        return out
    }

    /** In-place iterative radix-2 complex FFT. [re]/[im] length must be 2^k. */
    private fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        // bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * Math.PI / len
            val wRe = cos(ang).toFloat()
            val wIm = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var curRe = 1f
                var curIm = 0f
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
                    val vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len = len shl 1
        }
    }
}
