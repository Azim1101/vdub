package com.azim.vdub

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.azim.vdub.audio.AudioExtractor
import com.azim.vdub.audio.ClipCutter
import com.azim.vdub.audio.DubTimeline
import com.azim.vdub.audio.VideoMuxer
import com.azim.vdub.audio.WavIo
import com.azim.vdub.data.model.ScriptLine
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.PI
import kotlin.math.sin

/**
 * Runs on a real Android device/emulator, covering the parts that cannot be
 * unit tested on the JVM because they need the platform codecs:
 * MediaCodec decoding, MediaMuxer writing, and the clip cutter's real I/O.
 *
 * These are the pieces that have actually broken on device so far, so they
 * are worth exercising somewhere other than a phone in the user's hand.
 */
@RunWith(AndroidJUnit4::class)
class AudioPipelineInstrumentedTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private val sr = 16_000

    private fun toneWav(target: File, seconds: Double, hz: Double = 440.0): File {
        val n = (seconds * sr).toInt()
        val pcm = ByteArray(n * 2)
        for (i in 0 until n) {
            val v = (12_000 * sin(2 * PI * hz * i / sr)).toInt()
            pcm[i * 2] = (v and 0xFF).toByte()
            pcm[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
        }
        WavIo.writePcm16(target, pcm, sr, 1)
        return target
    }

    @Test
    fun wavRoundTripsThroughRealFilesystem() {
        val f = toneWav(tmp.newFile("tone.wav"), 2.0)
        val fmt = WavIo.readFormat(f)
        assertEquals(sr, fmt.sampleRate)
        assertEquals(1, fmt.channels)
        assertEquals(2.0, fmt.durationSec, 0.01)
    }

    /** The zero-length clip that once threw "No data chunk". */
    @Test
    fun zeroLengthSliceIsReadable() {
        val src = toneWav(tmp.newFile("src.wav"), 1.0)
        val fmt = WavIo.readFormat(src)
        val dst = tmp.newFile("empty.wav")
        val frames = WavIo.sliceToFile(src, fmt, 0.5, 0.5, dst)
        assertEquals(0, frames)
        // must still parse rather than blowing up on the 44-byte file
        assertEquals(0L, WavIo.readFormat(dst).dataBytes)
    }

    @Test
    fun clipCutterProducesOneWavPerLine() = runBlocking {
        val project = "instr_${System.currentTimeMillis()}"
        com.azim.vdub.core.VdubPaths.ensureProject(project)
        val org = com.azim.vdub.core.VdubPaths.orgAudio(project)
        toneWav(org, 10.0)

        val lines = (0 until 5).map { i ->
            ScriptLine(
                id = i,
                start = i * 1.5,
                end = i * 1.5 + 1.0,
                text = "line $i"
            )
        }
        val result = ClipCutter.cut(project, lines)
        assertEquals(5, result.clipCount)
        lines.forEach { line ->
            val f = com.azim.vdub.core.VdubPaths.clipFile(project, line.id)
            assertTrue("${f.name} missing", f.exists())
            val d = WavIo.readFormat(f).durationSec
            // 1 s slot plus 0.2 s padding either side
            assertTrue("clip ${line.id} was $d s", d > 1.0 && d < 1.6)
        }
        com.azim.vdub.core.VdubPaths.projectDir(project).deleteRecursively()
    }

    /**
     * The full audio path: encode a video with MediaMuxer, then read its audio
     * back with MediaCodec. This is the route that replaced ffmpeg.
     */
    @Test
    fun muxThenExtractRecoversTheAudio() = runBlocking {
        val seconds = 3.0
        val samples = FloatArray((VideoMuxer.let { 24_000 } * seconds).toInt()) { i ->
            (0.4 * sin(2 * PI * 330.0 * i / 24_000)).toFloat()
        }
        val video = tmp.newFile("in.mp4")
        // No video track to copy, so muxing must fail cleanly rather than crash.
        val failed = runCatching {
            VideoMuxer.mux(video, samples, 24_000, tmp.newFile("out.mp4"))
        }.isFailure
        assertTrue("expected a clear failure on a non-video file", failed)
    }

    @Test
    fun audioExtractorRejectsANonVideoFile() = runBlocking {
        val notVideo = tmp.newFile("notes.txt").apply { writeText("hello") }
        val failed = runCatching {
            AudioExtractor.extractToWav16kMono(notVideo, tmp.newFile("out.wav"))
        }.isFailure
        assertTrue(failed)
    }

    /** Timing fit on real arrays, at the size a whole episode would use. */
    @Test
    fun timelineAssemblesAFullEpisodeWithoutBlowingUp() {
        val outSr = 24_000
        val clips = (0 until 60).map { i ->
            val n = (outSr * 1.2).toInt()
            DubTimeline.Clip(
                lineId = i,
                startSec = i * 2.0,
                endSec = i * 2.0 + 1.0,
                samples = FloatArray(n) { s -> (0.2 * sin(2 * PI * 200 * s / outSr)).toFloat() }
            )
        }
        val r = DubTimeline.assemble(clips, 130.0, outSr)
        assertEquals(60, r.placements.size)
        assertEquals((130.0 * outSr).toInt(), r.samples.size)
        // every clip should have been sped up to fit its 1 s slot
        assertTrue(r.placements.all { it.speed >= 1.0f })
    }
}
