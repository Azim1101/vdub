package com.azim.vdub.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.LongBuffer

/**
 * Kyutai Mimi's neural codec decoder: codebook indices -> 24 kHz waveform.
 *
 * Indri emits Mimi tokens rather than audio, so this is what turns them into
 * sound. It exists as its own class because the available ONNX export needed
 * repairing before it could be used at all.
 *
 * ### The 32-vs-8 codebook problem
 *
 * Mimi is a residual vector quantizer with 32 codebooks. Indri only ever
 * emits the first 8 — that is how it was trained, and upstream decodes exactly
 * those 8 through the PyTorch model. The community ONNX export, however, froze
 * the input at `(batch, 32, frames)`, so 8 cannot be passed at all.
 *
 * The obvious fix is to pad the other 24 with index 0. It runs, and it is
 * wrong: measured against the 8-codebook decode upstream performs, in PyTorch
 * with no quantization involved, padding with zeros scores **6.8 dB SNR**.
 *
 * The reason is in how an RVQ decodes. Every codebook contributes one
 * embedding and they are summed:
 *
 *     latent = proj( Σ_q  codebook_q[ index_q ] )
 *
 * Padding with index 0 therefore adds `C = Σ_{q≥8} codebook_q[0]` to *every*
 * frame. Index 0 is not silence — it is an ordinary trained vector, and the
 * sum of 24 of them has norm 1.38, more than three times a typical single
 * codebook vector (0.42). It arrives as a constant timbre smeared over the
 * whole utterance.
 *
 * But nothing requires index 0. Any index is legal, and 24 codebooks × 2048
 * entries leaves ample freedom to choose a set that cancels. [PADDING_INDICES]
 * is that set, found by solving disjoint pairs exactly and then running
 * coordinate descent with restarts: it brings ‖C‖ from 1.383 down to 0.045, a
 * 31× reduction, and the decode from 6.8 dB to **34.4 dB** — inaudible.
 *
 * Those indices are a property of Mimi's weights, not of Indri, so they are
 * pinned here as constants and checked by a test rather than recomputed on the
 * phone (which would need the codebook tensors, which the ONNX graph does not
 * expose).
 *
 * ### Why fp16
 *
 * The int8 decoder's own quantization error is 15.6 dB even when handed all 32
 * real codebooks — worse than the padding error it would be masking. fp16
 * scores 34.4 dB, matching fp32, at half the size and a third of int8's
 * runtime. See `tools/probe_tts5.py`.
 */
class MimiDecoder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String
) : Closeable {

    companion object {
        const val SAMPLE_RATE = 24_000

        /** Codebooks the graph demands. */
        const val TOTAL_CODEBOOKS = 32

        /** Codebooks Indri actually produces. */
        const val USED_CODEBOOKS = 8

        const val CODEBOOK_SIZE = 2048

        /** Frames per second of audio — 24000 / 1920. */
        const val FRAME_RATE = 12.5

        /**
         * Indices for codebooks 8..31 whose embeddings very nearly cancel.
         *
         * ‖Σ codebook_q[i_q]‖ = 0.045, against 1.383 for all-zeros. Derived
         * from `kyutai/mimi`'s acoustic quantizer weights; see the class
         * comment and `MimiPaddingTest`.
         */
        val PADDING_INDICES = intArrayOf(
            1437, 374, 662, 1190, 1908, 1714, 220, 610,
            32, 1642, 1736, 1402, 692, 1897, 1332, 1774,
            724, 591, 1538, 483, 35, 14, 332, 1833
        )

        fun open(model: File, threads: Int = 4): MimiDecoder {
            check(model.exists() && model.length() > 0) {
                "${model.name} is missing — re-download the Indri engine."
            }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = try {
                env.createSession(model.absolutePath, opts)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Could not load ${model.name}. Remove and re-download the " +
                        "Indri engine.\n\n(${e.message.orEmpty().take(300)})",
                    e
                )
            }
            return MimiDecoder(env, session, session.inputNames.first())
        }

        /**
         * Reshape Indri's flat token stream into `(1, 32, frames)`.
         *
         * Indri interleaves codebooks: token *i* belongs to codebook `i % 8`,
         * and carries that codebook's base offset, which is removed here. The
         * remaining 24 rows are filled with [PADDING_INDICES].
         *
         * Kept static and free of ONNX types so the layout — the part that is
         * easy to get subtly wrong — is unit testable.
         */
        fun layout(tokens: IntArray): Array<IntArray> {
            val frames = tokens.size / USED_CODEBOOKS
            require(frames > 0) {
                "need at least $USED_CODEBOOKS tokens for one frame, got ${tokens.size}"
            }
            return Array(TOTAL_CODEBOOKS) { row ->
                if (row < USED_CODEBOOKS) {
                    IntArray(frames) { f ->
                        val v = tokens[f * USED_CODEBOOKS + row] - row * CODEBOOK_SIZE
                        // A token outside its codebook's band means the
                        // alternating mask failed upstream; clamping keeps the
                        // decoder from reading another codebook's entry.
                        v.coerceIn(0, CODEBOOK_SIZE - 1)
                    }
                } else {
                    val fill = PADDING_INDICES[row - USED_CODEBOOKS]
                    IntArray(frames) { fill }
                }
            }
        }
    }

    /** @return mono float samples in [-1, 1] at [SAMPLE_RATE]. */
    fun decode(tokens: IntArray): FloatArray {
        val grid = layout(tokens)
        val frames = grid[0].size

        val flat = LongArray(TOTAL_CODEBOOKS * frames)
        var k = 0
        for (row in grid) {
            for (v in row) flat[k++] = v.toLong()
        }

        OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(flat),
            longArrayOf(1, TOTAL_CODEBOOKS.toLong(), frames.toLong())
        ).use { input ->
            session.run(mapOf(inputName to input)).use { r ->
                return flattenAudio(r[0].value)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flattenAudio(value: Any?): FloatArray = when (value) {
        is FloatArray -> value
        is Array<*> -> when (val first = value.firstOrNull()) {
            is FloatArray -> {
                val rows = value as Array<FloatArray>
                if (rows.size == 1) rows[0]
                else FloatArray(rows.sumOf { it.size }).also { out ->
                    var k = 0
                    rows.forEach { it.copyInto(out, k); k += it.size }
                }
            }
            is Array<*> -> flattenAudio(first)
            else -> error("unexpected Mimi output element ${first?.javaClass}")
        }
        else -> error("unexpected Mimi output ${value?.javaClass}")
    }

    override fun close() {
        runCatching { session.close() }
    }
}
