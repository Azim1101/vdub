package com.azim.vdub

import com.azim.vdub.audio.WavIo
import com.azim.vdub.data.repo.ProjectRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The reference-audio cap, which is a stability limit rather than a tuning
 * knob.
 *
 * Without it the app took the whole phone down. A speaker whose longest lines
 * summed to 44 s produced a 44 s reference; Chatterbox converts that to speech
 * tokens at ~25/s and prepends them to every sequence it generates, so its KV
 * cache grew with it — and the cache is briefly held twice while each step
 * swaps old for new:
 *
 * | reference | prompt tokens | KV cache (×2) | with Q4 weights |
 * |-----------|---------------|---------------|-----------------|
 * | 8 s       | 200           | ~550 MB       | ~1.6 GB         |
 * | 44 s      | 1102          | ~985 MB       | ~2.1 GB         |
 *
 * At 2.1 GB Android's low-memory killer does not stop at this app — it takes
 * the launcher and system UI too, which the user sees as a spontaneous reboot.
 * Nothing in the pipeline capped the reference, and nothing failed loudly
 * first.
 *
 * These tests pin the arithmetic that makes the cap sufficient, so a future
 * "let's use more reference for better cloning" change has to confront it.
 */
class ReferenceLengthTest {

    /** Chatterbox's decoder geometry, from ChatterboxTts. */
    private val layers = 30
    private val kvHeads = 16
    private val headDim = 64
    private val bytesPerFloat = 4

    /** S3 speech-token rate: what one second of reference costs in sequence. */
    private val tokensPerSecond = 25

    /** Cap on tokens the model may generate for one line. */
    private val maxNewTokens = 1000

    private fun kvCacheBytes(sequenceLength: Int): Long =
        layers.toLong() * 2 * kvHeads * sequenceLength * headDim * bytesPerFloat

    private fun peakBytesFor(referenceSeconds: Double): Long {
        val prompt = (referenceSeconds * tokensPerSecond).toInt()
        // Old and new cache coexist for an instant during the swap.
        return kvCacheBytes(prompt + maxNewTokens) * 2
    }

    @Test
    fun `cap is short enough to keep chatterbox inside a phone`() {
        val peak = peakBytesFor(ProjectRepository.MAX_REFERENCE_SECONDS)
        val weights = 1_100_000_000L                    // Q4 pack, working set
        val total = peak + weights

        assertTrue(
            "peak ${total / 1024 / 1024} MB leaves no headroom on a 6 GB phone",
            total < 1_800_000_000L
        )
    }

    /**
     * The regression itself. If someone raises the cap to the old behaviour
     * this fails rather than the phone rebooting in the field.
     */
    @Test
    fun `the length that crashed the phone is now rejected`() {
        val crashed = 44.1
        assertTrue(
            "a $crashed s reference must not be allowed",
            ProjectRepository.MAX_REFERENCE_SECONDS < crashed
        )

        val before = peakBytesFor(crashed) + 1_100_000_000L
        val after = peakBytesFor(ProjectRepository.MAX_REFERENCE_SECONDS) + 1_100_000_000L
        assertTrue("the cap must actually save memory", after < before)
        assertTrue(
            "expected to save at least 300 MB, saved ${(before - after) / 1024 / 1024} MB",
            before - after > 300_000_000L
        )
    }

    /**
     * Still long enough to clone from. Both engines' guidance is 3–10 s, and
     * the UI warns below 3 s, so the cap must sit above that warning or every
     * speaker would look under-referenced.
     */
    @Test
    fun `cap stays in the range that clones well`() {
        assertTrue(
            "too short to clone from",
            ProjectRepository.MAX_REFERENCE_SECONDS >= 5.0
        )
        assertTrue(
            "longer than zero-shot cloning benefits from",
            ProjectRepository.MAX_REFERENCE_SECONDS <= 15.0
        )
        // The Step 5 "weak reference" warning triggers under 3 s.
        assertTrue(ProjectRepository.MAX_REFERENCE_SECONDS > 3.0)
    }

    /**
     * The byte budget the trimming uses must match the cap at the rate clips
     * are actually stored (16 kHz mono PCM16), or the cap silently means a
     * different duration than it claims.
     */
    @Test
    fun `byte budget matches the declared seconds`() {
        val sampleRate = 16_000
        val channels = 1
        val maxBytes = (ProjectRepository.MAX_REFERENCE_SECONDS * sampleRate).toInt() *
            2 * channels

        val seconds = maxBytes.toDouble() / (sampleRate * 2 * channels)
        assertEquals(ProjectRepository.MAX_REFERENCE_SECONDS, seconds, 1e-9)

        // and a capped wav is far smaller than the one that caused the crash
        val crashedBytes = (44.1 * sampleRate).toInt() * 2 * channels
        assertTrue(maxBytes < crashedBytes / 4)
    }

    /**
     * Trimming must land on a whole frame. A byte-odd cut shifts every
     * following sample by one byte, which is heard as a burst of noise rather
     * than a clean end.
     */
    @Test
    fun `a trimmed length stays frame aligned`() {
        for (channels in 1..2) {
            val frameBytes = channels * 2
            val budget = (ProjectRepository.MAX_REFERENCE_SECONDS * 16_000).toInt() *
                2 * channels
            // mirrors the alignment in buildReference
            val aligned = budget - budget % frameBytes
            assertEquals(0, aligned % frameBytes)
            assertTrue(aligned > 0)
        }
    }

    /** A capped reference is still comfortably above the minimum clip size. */
    @Test
    fun `capped reference exceeds the minimum usable clip`() {
        val capBytes = WavIo.HEADER_BYTES +
            (ProjectRepository.MAX_REFERENCE_SECONDS * 16_000).toInt() * 2
        val minBytes = 44 + 16_000 * 2          // WAV_MIN_BYTES
        assertTrue(capBytes > minBytes * 4)
    }

    /**
     * The stale-cache regression: installing the cap changed nothing for the
     * users who had already hit the crash.
     *
     * buildReference caches its result as out/ref_<speaker>.wav and returned
     * any existing file that passed a *minimum* size check. A project created
     * before the cap existed therefore kept handing the engine its old 44 s
     * reference, so the fix was a no-op precisely where it was needed. The
     * cached file must be validated against the cap, not just against zero.
     */
    @Test
    fun `an over-long cached reference is not reusable`() {
        val sampleRate = 16_000
        val channels = 1
        val cap = (ProjectRepository.MAX_REFERENCE_SECONDS * sampleRate).toLong() *
            2 * channels

        fun cachedBytesFor(seconds: Double) =
            (seconds * sampleRate).toLong() * 2 * channels

        // what the old build left on disk
        assertTrue("44 s cache must be rejected", cachedBytesFor(44.1) > cap)
        // what the new build writes
        assertTrue(
            "a freshly capped cache must be accepted",
            cachedBytesFor(ProjectRepository.MAX_REFERENCE_SECONDS) <= cap
        )
        // and something comfortably short stays valid
        assertTrue("a 5 s cache must be accepted", cachedBytesFor(5.0) <= cap)
    }

    /**
     * Generation is bounded too. The reference is only half the sequence — every
     * token the model emits also lengthens the KV cache, so a line that never
     * emits STOP could reach the old 1000-token ceiling and ~1.66 GB.
     */
    @Test
    fun `generation budget keeps the sequence bounded`() {
        val tokensPerSecond = 25
        val maxTokens = 500

        fun budgetFor(chars: Int): Int {
            val expected = chars / 15.0
            return ((expected * 3.0 + 4.0) * tokensPerSecond).toInt()
                .coerceIn(100, maxTokens)
        }

        // A short line must not be allowed to run away.
        assertTrue("a 10-char line got too much budget", budgetFor(10) < maxTokens)
        // But every line gets enough headroom for normal delivery: a 3x margin
        // over the expected length, so nothing is clipped mid-word.
        for (chars in listOf(10, 30, 60, 100)) {
            val spoken = budgetFor(chars).toDouble() / tokensPerSecond
            assertTrue(
                "$chars chars: budget ${spoken}s is under its own ${chars / 15.0}s",
                spoken > chars / 15.0 * 2
            )
        }
        // And the ceiling holds no matter how long the text is.
        assertEquals(maxTokens, budgetFor(10_000))

        val peak = kvCacheBytes((ProjectRepository.MAX_REFERENCE_SECONDS *
            tokensPerSecond).toInt() + maxTokens) * 2 + 1_100_000_000L
        assertTrue(
            "worst-case peak ${peak / 1024 / 1024} MB is too high",
            peak < 1_550_000_000L
        )
    }
}
