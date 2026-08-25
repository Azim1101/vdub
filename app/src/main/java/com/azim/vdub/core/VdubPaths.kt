package com.azim.vdub.core

import android.content.Context
import android.os.Build
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
 *
 * ## Storage fallback
 *
 * Writing to /storage/emulated/0/AI needs All-files access (MANAGE_EXTERNAL_
 * STORAGE), which the user must grant by hand in Settings. Until they do,
 * every write there fails with EPERM. Rather than dead-ending, the app falls
 * back to its own external directory, which needs no permission at all:
 *
 *   /Android/data/com.azim.vdub/files/AI/
 *
 * Everything works there; the only cost is that other apps (and adb push
 * without the full path) cannot see it. [usingSharedStorage] tells the UI
 * which root is live so it can offer the upgrade.
 */
object VdubPaths {

    /** Set once from the Application so a Context is always available. */
    @Volatile
    private var fallbackRoot: File? = null

    fun init(context: Context) {
        fallbackRoot = File(context.getExternalFilesDir(null), "AI")
    }

    /** True when we can use the shared /storage/emulated/0/AI folder. */
    fun hasAllFilesAccess(): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            // Pre-R the legacy WRITE_EXTERNAL_STORAGE grant covers it.
            Environment.getExternalStorageState() == Environment.MEDIA_MOUNTED
        }

    private val sharedRoot: File get() = File(Environment.getExternalStorageDirectory(), "AI")

    val usingSharedStorage: Boolean get() = hasAllFilesAccess() || fallbackRoot == null

    /**
     * Active root. Prefers the shared /AI folder, but only when we can
     * actually write there — otherwise the app-private fallback.
     */
    val aiRoot: File
        get() {
            if (hasAllFilesAccess()) return sharedRoot
            return fallbackRoot ?: sharedRoot
        }

    /** The shared path, for display even when it is not currently usable. */
    val sharedRootPath: String get() = sharedRoot.absolutePath

    val libsDir: File get() = File(aiRoot, "libs/arm64-v8a")
    val modelsDir: File get() = File(aiRoot, "models")
    val projectsRoot: File get() = File(aiRoot, "vdub_projects")

    /**
     * Models may sit in either root — a user who ran `adb push` before
     * granting access would otherwise appear to have no model.
     */
    fun findModel(name: String): File? {
        val candidates = listOfNotNull(
            File(modelsDir, name),
            File(File(sharedRoot, "models"), name),
            fallbackRoot?.let { File(File(it, "models"), name) }
        )
        return candidates.firstOrNull { it.exists() && it.length() > 1_000 }
    }

    fun projectDir(project: String): File = File(projectsRoot, project.sanitized())

    fun inputVideo(project: String) = File(projectDir(project), "input_video.mp4")
    fun orgAudio(project: String) = File(projectDir(project), "org_audio.wav")

    fun subsDir(project: String) = File(projectDir(project), "subs")
    fun originalSrt(project: String) = File(subsDir(project), "original.srt")
    fun translatedSrt(project: String) = File(subsDir(project), "translated.srt")

    fun clipsDir(project: String) = File(projectDir(project), "clips")
    fun clipFile(project: String, index: Int) =
        File(clipsDir(project), "line_%04d.wav".format(index))

    /** Step 5 output: one spoken wav per line. */
    fun hiClipsDir(project: String) = File(projectDir(project), "hi_clips")
    fun hiClipFile(project: String, index: Int) =
        File(hiClipsDir(project), "line_%04d.wav".format(index))

    /** Final muxed video. */
    fun dubbedVideo(project: String) = File(projectDir(project), "dubbed_video.mp4")

    fun outDir(project: String) = File(projectDir(project), "out")
    fun scriptRaw(project: String) = File(outDir(project), "script_raw.json")
    fun scriptSpeakers(project: String) = File(outDir(project), "script_speakers.json")
    fun speakerEmbeds(project: String) = File(outDir(project), "speaker_embeds.bin")
    fun scriptEmotion(project: String) = File(outDir(project), "script_emotion.json")
    fun scriptTranslated(project: String) = File(outDir(project), "script_translated.json")

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

    /**
     * Create the project tree and prove it is writable.
     * @throws IllegalStateException with an actionable message, instead of
     *         letting a raw EPERM surface later from deep inside a copy.
     */
    fun ensureProject(project: String) {
        val dir = projectDir(project)
        listOf(dir, subsDir(project), clipsDir(project), outDir(project))
            .forEach { it.mkdirs() }

        if (!dir.isDirectory) {
            error(
                "Cannot create ${dir.absolutePath}\n\n" +
                    "Grant \"All files access\" (folder icon, top right), " +
                    "or the app will use its private folder instead."
            )
        }
        val probe = File(dir, ".write_test")
        try {
            probe.writeText("ok")
            probe.delete()
        } catch (e: Exception) {
            error(
                "Storage is not writable:\n${dir.absolutePath}\n\n" +
                    "Tap the folder icon (top right) and enable " +
                    "\"All files access\", then press Open / Resume."
            )
        }
    }

    fun ensureRoots() {
        runCatching { listOf(aiRoot, libsDir, modelsDir, projectsRoot).forEach { it.mkdirs() } }
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
