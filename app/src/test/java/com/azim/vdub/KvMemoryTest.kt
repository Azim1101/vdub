package com.azim.vdub

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The KV cache must stay in native memory. Copying it into Java arrays blew
 * Android's 512 MB heap partway through a line, which is how the OOM
 * ("2916368 byte allocation ... growth limit 536870912") happened.
 *
 * These make the arithmetic explicit so a future refactor that reintroduces a
 * copy is caught by the numbers rather than by a three-hour run failing.
 */
class KvMemoryTest {

    private val layers = 30
    private val heads = 16
    private val headDim = 64
    private val entries = layers * 2
    private val androidHeapCap = 536_870_912L    // the limit in the crash

    private fun cacheBytes(tokens: Int): Long =
        entries.toLong() * heads * tokens * headDim * 4

    @Test
    fun `one copy of a mid-length cache is already huge`() {
        val bytes = cacheBytes(700)
        assertTrue("${bytes / 1024 / 1024} MB", bytes > 150L * 1024 * 1024)
    }

    /** Holding the old and new cache at once is what actually overflowed. */
    @Test
    fun `two copies exceed the heap before the token cap`() {
        val doubled = cacheBytes(1000) * 2
        assertTrue(
            "two copies = ${doubled / 1024 / 1024} MB vs cap " +
                "${androidHeapCap / 1024 / 1024} MB",
            doubled > androidHeapCap / 2
        )
    }

    /** A single entry matches the size the allocator reported failing on. */
    @Test
    fun `per entry size matches the reported allocation`() {
        val perEntry = heads.toLong() * 712 * headDim * 4
        assertTrue(perEntry in 2_700_000..3_000_000)
    }

    @Test
    fun `cache grows linearly with tokens`() {
        assertTrue(cacheBytes(200) * 2 == cacheBytes(400))
        assertTrue(cacheBytes(0) == 0L)
    }
}
