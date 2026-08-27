package com.azim.vdub.audio

import java.io.Closeable
import java.io.File

/**
 * What Step 5 needs from a voice engine, and nothing more.
 *
 * Four engines now sit behind this: two Chatterbox packs, DhVaani and Indri.
 * They disagree about almost everything — Chatterbox is an autoregressive LLM
 * over speech tokens, DhVaani is flow matching over mel frames, Indri is GPT-2
 * emitting Mimi codec tokens — but the stage only ever does three things:
 * enrol a speaker, speak a line, close. Keeping the surface that small is what
 * lets `speakAll` stay engine-agnostic.
 *
 * Two rules the implementations must honour, because the caller cannot check
 * them:
 *
 *  - [speak] returns mono float samples in [-1, 1] at [sampleRate]. Engines
 *    differ (24 kHz here, but that is not promised), so the rate is read from
 *    the instance rather than assumed by the caller.
 *  - [speak] is cancellable. Lines take tens of seconds and a user who taps
 *    Cancel must not wait for the current one to finish, so implementations
 *    check `coroutineContext.ensureActive()` inside their generation loop.
 */
interface TtsEngine : Closeable {

    /** Output rate of [speak], in Hz. */
    val sampleRate: Int

    /**
     * Whether this engine reproduces the reference speaker's voice.
     *
     * False for engines that only offer fixed preset voices. Step 5 shows that
     * up front rather than letting a user run a three-hour job and discover
     * every speaker sounds identical.
     */
    val clonesVoice: Boolean get() = true

    /**
     * Read a reference clip into whatever conditioning this engine uses.
     *
     * Called once per speaker and reused for all their lines: for most engines
     * this costs about as much as generating a line, so doing it per line
     * would roughly double the stage.
     *
     * @param referenceWav the speaker's own audio, or null when there is none
     *        to offer. Cloning engines reject null with a message naming the
     *        speaker; preset-voice engines ignore it entirely, which is why it
     *        is nullable rather than the caller inventing a dummy file.
     * @param transcript what is actually said in [referenceWav], when known.
     *        DhVaani needs it — it conditions on the reference *text* as well
     *        as the audio. Engines that do not care ignore it.
     */
    fun enrol(referenceWav: File?, transcript: String = ""): Voice

    /**
     * Speak [text] in [voice].
     *
     * @param exaggeration delivery strength from Step 3's emotion label, 1.0
     *        being neutral. Engines without such a control ignore it rather
     *        than approximating it with something that sounds wrong.
     * @param onToken progress within the line, for the UI. Units are
     *        engine-specific; only monotonicity is promised.
     */
    suspend fun speak(
        text: String,
        voice: Voice,
        language: String = "hi",
        exaggeration: Float = 1.0f,
        onToken: (Int) -> Unit = {}
    ): FloatArray

    /**
     * Opaque per-speaker conditioning.
     *
     * Deliberately empty: the shapes differ completely between engines and no
     * caller has any business reading them. Passing one engine's voice to
     * another is a programming error, and each implementation rejects it
     * rather than producing noise.
     */
    interface Voice
}
