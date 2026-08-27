package com.azim.vdub.audio

import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/**
 * Reads `.npz` archives — a zip of `.npy` arrays.
 *
 * DhVaani ships two of its pieces this way rather than as ONNX graphs: the
 * Vocos ISTFT head (`linear_weight`, `linear_bias`, `window`) and the exact
 * torchaudio mel filterbank. Both are plain weight matrices with no operations
 * attached, so there is nothing for ONNX Runtime to execute.
 *
 * Recomputing the filterbank from the usual HTK formula instead of reading it
 * is tempting and wrong: slightly different mel bin edges shift the reference
 * clip's features, and the model then clones a subtly different voice. The
 * shipped filterbank is the one it was trained against.
 *
 * Only what those two files use is supported — little-endian float32/float64
 * and int64, C-ordered, non-pickled. Anything else throws instead of returning
 * plausible garbage.
 */
object NpzReader {

    class NpArray(val shape: IntArray, val data: FloatArray) {
        val rows: Int get() = if (shape.isEmpty()) 1 else shape[0]
        val cols: Int get() = if (shape.size < 2) data.size / rows else shape[1]

        /** First element, for the rank-0 scalars npz uses for ints. */
        val scalar: Float get() = data.firstOrNull() ?: 0f

        /** Row-major view as [rows][cols]. */
        fun as2D(): Array<FloatArray> {
            require(shape.size == 2) { "expected rank 2, got ${shape.toList()}" }
            val (r, c) = shape[0] to shape[1]
            return Array(r) { i -> FloatArray(c) { j -> data[i * c + j] } }
        }
    }

    /** Every array in [file], keyed by name without the `.npy` suffix. */
    fun read(file: File): Map<String, NpArray> {
        require(file.exists() && file.length() > 0) { "${file.name} is missing" }
        val out = LinkedHashMap<String, NpArray>()
        ZipFile(file).use { zip ->
            for (entry in zip.entries()) {
                val name = entry.name.removeSuffix(".npy")
                zip.getInputStream(entry).use { stream ->
                    out[name] = readNpy(stream, file.name, name)
                }
            }
        }
        check(out.isNotEmpty()) { "${file.name} contains no arrays" }
        return out
    }

    private fun readNpy(raw: InputStream, container: String, entry: String): NpArray {
        val input = DataInputStream(raw.buffered())

        val magic = ByteArray(6)
        input.readFully(magic)
        require(magic[0] == 0x93.toByte() && String(magic, 1, 5) == "NUMPY") {
            "$container/$entry is not a .npy array"
        }

        val major = input.readUnsignedByte()
        input.readUnsignedByte()                        // minor, unused
        val headerLen = if (major >= 2) {
            val b = ByteArray(4)
            input.readFully(b)
            ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
        } else {
            val b = ByteArray(2)
            input.readFully(b)
            ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
        }

        val header = ByteArray(headerLen).also { input.readFully(it) }
            .toString(Charsets.US_ASCII)

        val descr = Regex("'descr'\\s*:\\s*'([^']+)'").find(header)?.groupValues?.get(1)
            ?: error("$container/$entry has no dtype")
        val fortran = Regex("'fortran_order'\\s*:\\s*(True|False)")
            .find(header)?.groupValues?.get(1) == "True"
        require(!fortran) { "$container/$entry is Fortran-ordered" }

        val shape = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)
            .orEmpty()
            .split(',')
            .mapNotNull { it.trim().toIntOrNull() }
            .toIntArray()

        val count = if (shape.isEmpty()) 1 else shape.fold(1) { a, b -> a * b }
        require(descr.startsWith("<") || descr.startsWith("|")) {
            "$container/$entry is big-endian ($descr)"
        }

        val elementBytes = when (descr.drop(1)) {
            "f4" -> 4
            "f8" -> 8
            "i8" -> 8
            "i4" -> 4
            else -> error("$container/$entry has unsupported dtype $descr")
        }

        val bytes = ByteArray(count * elementBytes).also { input.readFully(it) }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val data = FloatArray(count)
        when (descr.drop(1)) {
            "f4" -> for (i in 0 until count) data[i] = buf.float
            "f8" -> for (i in 0 until count) data[i] = buf.double.toFloat()
            "i8" -> for (i in 0 until count) data[i] = buf.long.toFloat()
            "i4" -> for (i in 0 until count) data[i] = buf.int.toFloat()
        }
        return NpArray(shape, data)
    }
}
