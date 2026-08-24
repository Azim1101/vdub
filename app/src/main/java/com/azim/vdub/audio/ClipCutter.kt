package com.azim.vdub.audio

import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.model.ScriptLine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/**
 * Cuts org_audio.wav into one wav per script line.
 *
 * Per spec this is pure sample slicing (numpy-style), NOT ffmpeg:
 *   s = (start - 0.2) * sr ; e = (end + 0.2) * sr ; clip = wav[s:e]
 * ffmpeg-based cutting failed on-device, and 190 subprocess spawns would be
 * slow anyway. Here each clip is one seek + one read of the exact byte range.
 */
object ClipCutter {

    const val PAD_SEC = 0.2

    data class Result(
        val clipCount: Int,
        val totalSeconds: Double,
        val outputDir: File
    )

    suspend fun cut(
        project: String,
        lines: List<ScriptLine>,
        padSec: Double = PAD_SEC,
        onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }
    ): Result = withContext(Dispatchers.IO) {
        val source = VdubPaths.orgAudio(project)
        require(source.exists()) { "org_audio.wav missing — extract audio first" }

        val fmt = WavIo.readFormat(source)
        val clipsDir = VdubPaths.clipsDir(project).apply { mkdirs() }

        // Fresh cut: drop stale clips so a re-run can't leave orphans behind.
        clipsDir.listFiles { f -> f.name.startsWith("line_") && f.extension == "wav" }
            ?.forEach { it.delete() }

        var totalSec = 0.0
        lines.forEachIndexed { i, line ->
            coroutineContext.ensureActive()
            val start = (line.start - padSec).coerceAtLeast(0.0)
            val end = (line.end + padSec).coerceAtMost(fmt.durationSec)
            val target = VdubPaths.clipFile(project, line.id)
            val frames = WavIo.sliceToFile(source, fmt, start, end, target)
            totalSec += frames.toDouble() / fmt.sampleRate
            onProgress(i + 1, lines.size)
        }

        Result(
            clipCount = lines.size,
            totalSeconds = totalSec,
            outputDir = clipsDir
        )
    }
}
