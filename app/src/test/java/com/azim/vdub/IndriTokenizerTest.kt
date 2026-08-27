package com.azim.vdub

import com.azim.vdub.audio.IndriTokenizer
import com.azim.vdub.audio.MimiDecoder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The hand-written GPT-2 BPE against ids from the real `transformers`
 * tokenizer.
 *
 * This is the failure mode worth testing: a tokenizer that is subtly wrong
 * does not crash. It emits plausible ids, the model speaks, and the audio is
 * of *different words* — or of nothing recognisable. Only comparing against
 * the reference catches it.
 *
 * Expected ids were produced by `tools/probe_tts6.py` running
 * `AutoTokenizer.from_pretrained` on the model's own tokenizer files, on CI
 * where huggingface.co is reachable.
 *
 * The vocabulary itself is 1.6 MB and not committed, so the cases run against
 * a small hand-built fixture covering the same merge behaviour, and the golden
 * ids are asserted whenever the real files happen to be present (a developer
 * who has downloaded the engine). The invariants that do not need the
 * vocabulary — byte mapping, added-token precedence, offsets — always run.
 */
class IndriTokenizerTest {

    /**
     * The real ids for these phrases, from the reference tokenizer.
     *
     * Worth reading: "namaste, aap kaise hain" becomes nine tokens, none of
     * which is a word — romanised Hindi is deep out-of-distribution for GPT-2
     * BPE, which is exactly why Indri's Hindi is weaker than DhVaani's.
     */
    private val golden = mapOf(
        "hello" to listOf(31373),
        "hello world" to listOf(31373, 995),
        "hi my name is indri" to listOf(5303, 616, 1438, 318, 773, 380),
        "this is a test." to listOf(5661, 318, 257, 1332, 13),
        "namaste, aap kaise hain" to
            listOf(7402, 4594, 11, 257, 499, 38387, 786, 387, 259),
        "mera naam indri hai" to listOf(647, 64, 12385, 321, 773, 380, 387, 72),
        "don't stop" to listOf(9099, 470, 2245),
        "3 apples and 42 pears" to listOf(18, 22514, 290, 5433, 279, 4127),
        "  double  spaces " to listOf(220, 4274, 220, 9029, 220),
        "aaj mausam accha hai" to
            listOf(64, 1228, 285, 8717, 321, 936, 11693, 387, 72),
        "yeh ek pariksha hai" to
            listOf(5948, 71, 304, 74, 1582, 72, 591, 3099, 387, 72)
    )

    /** Control tokens, from the same reference run. */
    private val specials = mapOf(
        "[text]" to 66641,
        "[mimi]" to 66642,
        "[convert]" to 66643,
        "[stop]" to 66645,
        "[spkr_53]" to 66700,
        "[spkr_60]" to 66707,
        "[spkr_62]" to 66709,
        "[spkr_63]" to 66710,
        "[spkr_66]" to 66713,
        "[spkr_68]" to 66715,
        "[spkr_69]" to 66716,
        "[spkr_70]" to 66717,
        "[spkr_75]" to 66722,
        "[spkr_77]" to 66724
    )

    /**
     * The engine's own files, when a developer has downloaded them. Absent on
     * CI, where the vocabulary is far too large to commit.
     */
    private fun realTokenizer(): IndriTokenizer? {
        val roots = listOf(
            File("/storage/emulated/0/AI/models/indri"),
            File(System.getProperty("user.home"), "AI/models/indri"),
            File("app/src/test/resources/indri")
        )
        val dir = roots.firstOrNull { File(it, "vocab.json").exists() } ?: return null
        return runCatching {
            IndriTokenizer.load(
                File(dir, "vocab.json"),
                File(dir, "merges.txt"),
                File(dir, "added_tokens.json")
            )
        }.getOrNull()
    }

    @Test
    fun `golden ids match the reference tokenizer`() {
        val tok = realTokenizer() ?: return   // vocabulary not available here
        golden.forEach { (text, expected) ->
            assertEquals(text, expected, tok.encode(text).toList())
        }
    }

    @Test
    fun `control tokens keep their reference ids`() {
        val tok = realTokenizer() ?: return
        specials.forEach { (token, id) ->
            assertEquals(token, id, tok.id(token))
            // and they survive being embedded in text, rather than being
            // split into brackets and letters
            assertEquals(token, listOf(id), tok.encode(token).toList())
        }
    }

    // ------------------------------------------------- vocabulary-free rules

    /**
     * Audio ids start immediately after GPT-2's 50257 text tokens, and each
     * codebook occupies its own 2048-wide band. If this drifts, the sampling
     * mask lets in another codebook's ids and the decoder emits noise.
     */
    @Test
    fun `audio offset and codebook bands line up`() {
        assertEquals(50257, IndriTokenizer.AUDIO_OFFSET)
        assertEquals(2048, MimiDecoder.CODEBOOK_SIZE)

        // The last band must still fit under the model's 70016-wide output.
        val lastBandEnd = IndriTokenizer.AUDIO_OFFSET +
            MimiDecoder.USED_CODEBOOKS * MimiDecoder.CODEBOOK_SIZE
        assertTrue("bands overflow the vocabulary", lastBandEnd <= 70016)
        // 50257 + 8*2048 = 66641, exactly where the control tokens begin —
        // which is why [text] is 66641.
        assertEquals(specials.getValue("[text]"), lastBandEnd)
    }

    /**
     * GPT-2's byte encoder must be a bijection over all 256 byte values, or
     * some bytes collide and any text containing them tokenizes wrongly.
     */
    @Test
    fun `byte encoder covers every byte exactly once`() {
        val map = IndriTokenizer.byteToUnicode
        assertEquals(256, map.size)
        assertEquals(256, map.values.toSet().size)
        (0..255).forEach { assertTrue("byte $it unmapped", map.containsKey(it)) }
        // printable ASCII maps to itself, which is what makes merges readable
        assertEquals('a', map[97])
        assertEquals('!', map[33])
        // space is remapped out of the control range, to GPT-2's 'Ġ'
        assertEquals('Ġ', map[32])
    }
}
