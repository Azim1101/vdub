package com.azim.vdub.core

import android.os.Environment
import java.io.File

/**
 * Canonical on-device layout (the "ALU /AI/ folder" from the spec).
 *
 * /storage/emulated/0/AI/
 * ├── libs/arm64-v8a/         libMNN.so, libonnxruntime.so
 * ├── models/                 campplus.onnx, emotion2vec_plus_base.onnx, ...
 * └── vdub_projects/{project}/
 *     ├── input_video.mp4
 *     ├── org_audio.wav        16 kHz mono PCM16
 *     ├── subs/original.srt
 *     ├── subs/translated.srt
 *     ├── clips/line_0000.wav  ... 190 clips
 *     ├── out/script_raw.json
 *     └── S01.done             step marker (resume-safe)
 */
object VdubPaths {

    private val sdRoot: File get() = Environment.getExternalStorageDirectory()

    val aiRoot: File get() = File(sdRoot, "AI")
    val libsDir: File get() = File(aiRoot, "libs/arm64-v8a")
    val modelsDir: File get() = File(aiRoot, "models")
    val projectsRoot: File get() = File(aiRoot, "vdub_projects")

    fun projectDir(project: String): File = File(projectsRoot, project.sanitized())

    fun inputVideo(project: String) = File(projectDir(project), "input_video.mp4")
    fun orgAudio(project: String) = File(projectDir(project), "org_audio.wav")

    fun subsDir(project: String) = File(projectDir(project), "subs")
    fun originalSrt(project: String) = File(subsDir(project), "original.srt")
    fun translatedSrt(project: String) = File(subsDir(project), "translated.srt")

    fun clipsDir(project: String) = File(projectDir(project), "clips")
    fun clipFile(project: String, index: Int) =
        File(clipsDir(project), "line_%04d.wav".format(index))

    fun outDir(project: String) = File(projectDir(project), "out")
    fun scriptRaw(project: String) = File(outDir(project), "script_raw.json")
    fun scriptSpeakers(project: String) = File(outDir(project), "script_speakers.json")
    fun speakerEmbeds(project: String) = File(outDir(project), "speaker_embeds.bin")

    /** Resume markers: S01.done ... S05.done */
    fun stepMarker(project: String, step: Int) =
        File(projectDir(project), "S%02d.done".format(step))

    fun isStepDone(project: String, step: Int) = stepMarker(project, step).exists()

    fun markStepDone(project: String, step: Int) {
        ensureProject(project)
        stepMarker(project, step).writeText(System.currentTimeMillis().toString())
    }

    fun clearStep(project: String, step: Int) {
        stepMarker(project, step).delete()
    }

    fun ensureProject(project: String) {
        listOf(projectDir(project), subsDir(project), clipsDir(project), outDir(project))
            .forEach { it.mkdirs() }
    }

    fun ensureRoots() {
        listOf(aiRoot, libsDir, modelsDir, projectsRoot).forEach { it.mkdirs() }
    }

    fun listProjects(): List<String> =
        projectsRoot.listFiles { f -> f.isDirectory }?.map { it.name }?.sorted() ?: emptyList()

    fun clipCount(project: String): Int =
        clipsDir(project).listFiles { f -> f.name.endsWith(".wav") }?.size ?: 0

    /** Directory size in bytes (cheap recursive walk). */
    fun sizeOf(file: File): Long =
        if (!file.exists()) 0L
        else if (file.isFile) file.length()
        else file.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    private fun String.sanitized(): String =
        trim().replace(Regex("[^A-Za-z0-9._-]"), "_").ifBlank { "vdub_step" }
}
