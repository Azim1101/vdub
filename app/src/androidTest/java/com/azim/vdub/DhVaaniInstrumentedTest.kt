package com.azim.vdub

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.azim.vdub.audio.DhVaaniTts
import com.azim.vdub.audio.VoiceEngine
import com.azim.vdub.audio.WavIo
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.net.ModelDownloader
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Runs the real DhVaani pipeline on the ONNX Runtime that ships in the APK.
 *
 * The unit tests pin the hand-written DSP (FFT, feature scaling, npz layout)
 * and the probes verified the graphs — but on desktop onnxruntime on x86
 * Linux. ORT *Mobile* is a different build: different op coverage, different
 * int8 kernels, different memory behaviour, and none of that is reachable
 * from a JVM test. Every on-device failure so far came from exactly that gap,
 * so this test runs the full path once — download, open all three graphs,
 * enrol, speak — on one short line, and leaves the result on disk where the
 * CI workflow pulls it for a human to listen to.
 *
 * The reference is a tone, not speech. Cloning fidelity is not the target;
 * the target is "the graphs execute on the phone stack and produce audio".
 * A tone is deterministic and asset-free, and it still surfaces every
 * realistic failure: an unsupported op, an int8 kernel that NaNs, an OOM,
 * a shape mismatch deep in the vocoder.
 */
@RunWith(AndroidJUnit4::class)
class DhVaaniInstrumentedTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun dhvaaniSpeaksOneLineOnDevice() = runBlocking {
        val model = ModelCatalog.DHVAANI_TTS
        val downloader = ModelDownloader()
        if (!downloader.isInstalled(model)) {
            downloader.download(model)
        }
        assertTrue(
            "DhVaani files incomplete after download: " +
                VoiceEngine.pathsFor(model.id).missing.joinToString { it.name },
            VoiceEngine.isInstalled(model.id)
        )

        // Two threads: the runner vCPUs are shared and the probe's RTF
        // numbers were taken at exactly this setting.
        VoiceEngine.open(model.id, threads = 2).use { engine ->
            assertEquals(DhVaaniTts.SAMPLE_RATE, engine.sampleRate)

            // 0.5 s of tone at 16 kHz — the app's analysis rate — which also
            // exercises the resample to 24 kHz inside enrol.
            val refRate = 16_000
            val ref = File(context.filesDir, "dhvaani_smoke_ref.wav")
            val pcm = ByteArray((0.5 * refRate).toInt() * 2)
            for (i in pcm.indices step 2) {
                val v = (9_000 * sin(2 * PI * 440.0 * i / refRate)).toInt()
                pcm[i] = (v and 0xFF).toByte()
                pcm[i + 1] = ((v shr 8) and 0xFF).toByte()
            }
            WavIo.writePcm16(ref, pcm, refRate, 1)

            val voice = engine.enrol(ref, "नमस्ते।")
            val wav = engine.speak("नमस्ते, मैं ठीक हूँ।", voice, language = "hi")

            val seconds = wav.size / engine.sampleRate.toDouble()
            assertTrue("output is empty", wav.isNotEmpty())
            assertTrue("duration $seconds s is implausible for one short line",
                seconds in 0.3..60.0)
            assertTrue("output contains non-finite samples",
                wav.all { it.isFinite() })
            val rms = sqrt(wav.sumOf { (it * it).toDouble() } / wav.size)
            assertTrue("output is silent (rms $rms)", rms > 1e-4)

            // Leave the result where the workflow can pull it: the probes
            // proved the model on Linux, and this file is what the phone
            // stack actually produced.
            val out = File(VdubPaths.aiRoot, "dhvaani_smoke_line.wav")
            out.delete()
            out.parentFile?.mkdirs()
            WavIo.writePcm16(out, toPcm16(wav), engine.sampleRate, 1)
            assertTrue("could not write ${out.absolutePath}", out.exists())
        }
    }

    private fun toPcm16(samples: FloatArray): ByteArray {
        val out = ByteArray(samples.size * 2)
        for (i in samples.indices) {
            val v = (samples[i].coerceIn(-1f, 1f) * 32_767f).toInt()
            out[i * 2] = (v and 0xFF).toByte()
            out[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
        }
        return out
    }
}
