package com.azim.vdub

import com.azim.vdub.audio.SpeakerCluster
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt
import kotlin.random.Random

class SpeakerClusterTest {

    private val dim = 192

    private fun unit(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x.toDouble() * x
        val n = sqrt(s).toFloat().coerceAtLeast(1e-9f)
        return FloatArray(v.size) { v[it] / n }
    }

    /**
     * Factor model matching real campplus statistics: within-speaker cosine
     * ~0.70, cross-speaker ~0.09.
     */
    private fun synth(counts: List<Int>, seed: Int = 7): Pair<List<FloatArray>, List<Int>> {
        val rnd = Random(seed)
        fun gauss() = rnd.nextDouble().let {
            sqrt(-2.0 * kotlin.math.ln(rnd.nextDouble() + 1e-12)) *
                kotlin.math.cos(2 * Math.PI * it)
        }.toFloat()

        val shared = unit(FloatArray(dim) { gauss() })
        val a = sqrt(0.29f)
        val b = sqrt(1 - 0.29f)
        val centers = counts.map { unit(FloatArray(dim) { i -> a * shared[i] + b * gauss() }) }

        val ra = sqrt(0.70f)
        val rb = sqrt(1 - 0.70f)
        val embs = ArrayList<FloatArray>()
        val truth = ArrayList<Int>()
        centers.forEachIndexed { s, c ->
            repeat(counts[s]) {
                val nz = unit(FloatArray(dim) { gauss() })
                embs += unit(FloatArray(dim) { i -> ra * c[i] + rb * nz[i] })
                truth += s
            }
        }
        return embs to truth
    }

    private fun purity(labels: IntArray, truth: List<Int>): Double {
        val groups = labels.indices.groupBy { labels[it] }
        val correct = groups.values.sumOf { idx ->
            idx.groupingBy { truth[it] }.eachCount().values.maxOrNull() ?: 0
        }
        return correct.toDouble() / truth.size
    }

    @Test
    fun `recovers three speakers from 190 clips`() {
        val (embs, truth) = synth(listOf(120, 50, 20))
        val r = SpeakerCluster.cluster(embs, threshold = 0.45f)
        assertEquals(3, r.speakerCount)
        assertTrue("purity ${purity(r.labels, truth)}", purity(r.labels, truth) > 0.95)
        assertEquals(listOf(120, 50, 20), r.sizes())
    }

    /**
     * Guards the direction of the threshold. Raising it must never *reduce*
     * the speaker count — the opposite of what the original spec assumed.
     */
    @Test
    fun `higher threshold yields at least as many speakers`() {
        val (embs, _) = synth(listOf(120, 50, 20))
        var previous = 0
        listOf(0.20f, 0.30f, 0.45f, 0.60f, 0.75f).forEach { thr ->
            val k = SpeakerCluster.cluster(embs, thr).speakerCount
            assertTrue(
                "thr=$thr gave $k, previous was $previous — must be monotonic",
                k >= previous
            )
            previous = k
        }
    }

    @Test
    fun `target k overrides the threshold`() {
        val (embs, truth) = synth(listOf(120, 50, 20))
        val r = SpeakerCluster.clusterToK(embs, 3)
        assertEquals(3, r.speakerCount)
        assertTrue(purity(r.labels, truth) > 0.95)
    }

    @Test
    fun `largest cluster is speaker one`() {
        val (embs, _) = synth(listOf(20, 120, 50))
        val r = SpeakerCluster.clusterToK(embs, 3)
        val sizes = r.sizes()
        assertEquals(sizes.sortedDescending(), sizes)
    }

    @Test
    fun `single linkage chaining is avoided`() {
        // Two tight groups joined by one ambiguous bridge clip. Single-link
        // would merge everything; average link must keep them apart.
        val rnd = Random(3)
        fun g() = (rnd.nextDouble() - 0.5).toFloat()
        val a = unit(FloatArray(dim) { g() })
        val b = unit(FloatArray(dim) { g() })
        val embs = ArrayList<FloatArray>()
        repeat(20) { embs += unit(FloatArray(dim) { i -> a[i] + 0.10f * g() }) }
        repeat(20) { embs += unit(FloatArray(dim) { i -> b[i] + 0.10f * g() }) }
        embs += unit(FloatArray(dim) { i -> 0.5f * a[i] + 0.5f * b[i] })  // bridge

        val r = SpeakerCluster.cluster(embs, threshold = 0.55f)
        assertTrue("expected >=2 clusters, got ${r.speakerCount}", r.speakerCount >= 2)
    }

    @Test
    fun `handles empty and single input`() {
        assertEquals(0, SpeakerCluster.cluster(emptyList()).speakerCount)
        assertEquals(1, SpeakerCluster.cluster(listOf(unit(FloatArray(dim) { 1f }))).speakerCount)
    }

    @Test
    fun `cosine of identical vectors is one`() {
        val v = unit(FloatArray(dim) { it.toFloat() })
        assertEquals(1.0f, SpeakerCluster.cosine(v, v), 1e-5f)
    }
}
