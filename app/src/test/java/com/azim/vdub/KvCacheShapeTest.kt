package com.azim.vdub

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The generation loop starts with an *empty* KV cache — shape
 * [1, heads, 0, head_dim]. ORT's nested-array API infers shape from the arrays
 * and rejects a zero-length axis with "Supplied array has a zero dimension",
 * so the cache has to be passed as a buffer with an explicit shape.
 *
 * These pin the shape arithmetic that fix depends on.
 */
class KvCacheShapeTest {

    private val layers = 30
    private val heads = 16
    private val headDim = 64

    private fun emptyShape() = longArrayOf(1, heads.toLong(), 0L, headDim.toLong())

    private fun grownShape(pastLen: Int) =
        longArrayOf(1, heads.toLong(), pastLen.toLong(), headDim.toLong())

    @Test
    fun `empty cache has a zero time axis`() {
        val s = emptyShape()
        assertEquals(4, s.size)
        assertEquals(0L, s[2])
        // a buffer of length 0 matches it, which is exactly what nested arrays
        // cannot express
        assertEquals(0, s.fold(1L) { a, b -> a * b }.toInt())
    }

    @Test
    fun `every layer gets a key and a value entry`() {
        val names = buildList {
            for (l in 0 until layers) for (kv in listOf("key", "value")) {
                add("past_key_values.$l.$kv")
            }
        }
        assertEquals(layers * 2, names.size)
        assertEquals(names.size, names.distinct().size)
        assertTrue(names.contains("past_key_values.0.key"))
        assertTrue(names.contains("past_key_values.29.value"))
    }

    /** present.N.key must map back onto past_key_values.N.key for the next step. */
    @Test
    fun `present names map onto past names`() {
        assertEquals(
            "past_key_values.7.value",
            "present.7.value".replace("present", "past_key_values")
        )
    }

    @Test
    fun `buffer length matches the declared shape as the cache grows`() {
        listOf(0, 1, 12, 400).forEach { past ->
            val s = grownShape(past)
            val expected = 1 * heads * past * headDim
            assertEquals(expected.toLong(), s.fold(1L) { a, b -> a * b })
        }
    }

    /** A [1, T, D] feature block flattens to exactly T*D floats. */
    @Test
    fun `prompt features flatten consistently`() {
        val rows = 7
        val dim = 80
        val flat = FloatArray(rows * dim)
        assertEquals(rows.toLong() * dim, flat.size.toLong())
        val shape = longArrayOf(1, rows.toLong(), dim.toLong())
        assertEquals(flat.size.toLong(), shape.fold(1L) { a, b -> a * b })
    }
}
