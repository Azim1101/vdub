package com.azim.vdub.data.repo

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import com.azim.vdub.audio.AudioExtractor
import com.azim.vdub.audio.ClipCutter
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.local.ClipDao
import com.azim.vdub.data.local.ClipEntity
import com.azim.vdub.data.local.ProjectDao
import com.azim.vdub.data.local.ProjectEntity
import com.azim.vdub.data.model.ScriptLine
import com.azim.vdub.data.model.ScriptRaw
import com.azim.vdub.data.model.SrtCue
import com.azim.vdub.data.model.VideoSource
import com.azim.vdub.net.DownloadClient
import com.azim.vdub.subtitle.SrtParser
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.coroutineContext

@Singleton
class ProjectRepository @Inject constructor(
    @ApplicationContext private val context: Context,
    private val projectDao: ProjectDao,
    private val clipDao: ClipDao,
    private val downloadClient: DownloadClient
) {
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        explicitNulls = false
    }

    fun observeProject(name: String): Flow<ProjectEntity?> = projectDao.observe(name)
    fun observeAllProjects(): Flow<List<ProjectEntity>> = projectDao.observeAll()

    suspend fun getProject(name: String): ProjectEntity? = projectDao.get(name)

    suspend fun ensureProject(name: String): ProjectEntity = withContext(Dispatchers.IO) {
        VdubPaths.ensureRoots()
        VdubPaths.ensureProject(name)
        projectDao.get(name) ?: ProjectEntity(name = name).also { projectDao.upsert(it) }
    }

    suspend fun update(entity: ProjectEntity) =
        projectDao.upsert(entity.copy(updatedAt = System.currentTimeMillis()))

    // ---------------------------------------------------------------- video

    /** Copy a picked gallery/SAF video into the project folder. */
    suspend fun importVideoFromUri(
        project: String,
        uri: Uri,
        onProgress: (Long, Long) -> Unit = { _, _ -> }
    ): File = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val target = VdubPaths.inputVideo(project)
        val total = querySize(uri)

        context.contentResolver.openInputStream(uri)
            ?.use { input ->
                target.outputStream().use { out ->
                    val buf = ByteArray(512 * 1024)
                    var copied = 0L
                    var lastTick = 0L
                    while (true) {
                        coroutineContext.ensureActive()
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        copied += n
                        val now = System.currentTimeMillis()
                        if (now - lastTick > 150) {
                            lastTick = now
                            onProgress(copied, total)
                        }
                    }
                    onProgress(copied, total)
                }
            } ?: error("Cannot open $uri")

        registerVideo(project, target, VideoSource.GALLERY, null)
        target
    }

    suspend fun downloadVideoFromUrl(
        project: String,
        url: String,
        serverBase: String?,
        onProgress: (Long, Long) -> Unit = { _, _ -> }
    ): File = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val target = VdubPaths.inputVideo(project)
        if (!serverBase.isNullOrBlank()) downloadClient.serverBase = serverBase

        val direct = url.substringBefore('?').endsWith(".mp4", ignoreCase = true)
        if (direct) {
            downloadClient.downloadDirect(url, target, onProgress)
        } else {
            downloadClient.downloadFromUrl(url, project, target, onProgress = onProgress)
        }
        registerVideo(project, target, VideoSource.URL, url)
        target
    }

    private suspend fun registerVideo(
        project: String,
        file: File,
        source: VideoSource,
        url: String?
    ) {
        val duration = probeDurationMs(file)
        val existing = projectDao.get(project) ?: ProjectEntity(name = project)
        projectDao.upsert(
            existing.copy(
                videoPath = file.absolutePath,
                videoSource = source.name,
                sourceUrl = url,
                durationMs = duration,
                step1Done = false,
                updatedAt = System.currentTimeMillis()
            )
        )
        VdubPaths.clearStep(project, 1)
    }

    /** MediaMetadataRetriever is only AutoCloseable from API 29, so release by hand. */
    fun probeDurationMs(file: File): Long {
        if (!file.exists()) return 0L
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(file.absolutePath)
            mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        } catch (_: Exception) {
            0L
        } finally {
            runCatching { mmr.release() }
        }
    }

    private fun querySize(uri: Uri): Long = runCatching {
        context.contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.SIZE)
            if (idx >= 0 && c.moveToFirst()) c.getLong(idx) else -1L
        } ?: -1L
    }.getOrDefault(-1L)

    // ------------------------------------------------------------- subtitles

    /** Import an SRT, merge cues into lines, persist script_raw.json. */
    suspend fun importSrt(
        project: String,
        uri: Uri,
        maxGapMs: Long = 400
    ): Pair<List<SrtCue>, List<ScriptLine>> = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val target = VdubPaths.originalSrt(project)
        context.contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { input.copyTo(it) }
        } ?: error("Cannot open subtitle $uri")

        val cues = SrtParser.parse(target)
        val lines = SrtParser.mergeToLines(cues, maxGapMs = maxGapMs)
        writeScriptRaw(project, lines, cues.size)

        val existing = projectDao.get(project) ?: ProjectEntity(name = project)
        projectDao.upsert(
            existing.copy(
                srtPath = target.absolutePath,
                cueCount = cues.size,
                lineCount = lines.size,
                updatedAt = System.currentTimeMillis()
            )
        )
        cues to lines
    }

    /**
     * Re-merge the already-imported SRT with a different gap threshold.
     * Lets the user dial the cue->line ratio in (e.g. 473 cues -> 190 lines)
     * without re-picking the file. Any existing clips become stale, so S01 is
     * cleared and the clip table is dropped.
     */
    suspend fun remergeLines(project: String, maxGapMs: Long): List<ScriptLine> =
        withContext(Dispatchers.IO) {
            val srt = VdubPaths.originalSrt(project)
            require(srt.exists()) { "No subtitles imported yet" }
            val cues = SrtParser.parse(srt)
            var lines = SrtParser.mergeToLines(cues, maxGapMs = maxGapMs)

            // Re-attach translations from the translated SRT — new line
            // boundaries mean the old per-line mapping no longer applies.
            val translatedFile = VdubPaths.translatedSrt(project)
            if (translatedFile.exists()) {
                val tCues = SrtParser.parse(translatedFile)
                lines = lines.map { line ->
                    val overlapping = tCues.filter { cue ->
                        cue.endMs / 1000.0 > line.start && cue.startMs / 1000.0 < line.end
                    }
                    if (overlapping.isEmpty()) line
                    else line.copy(translated = overlapping.joinToString(" ") { it.text })
                }
            }
            writeScriptRaw(project, lines, cues.size)

            clipDao.clear(project)
            VdubPaths.clearStep(project, 1)
            val existing = projectDao.get(project) ?: ProjectEntity(name = project)
            projectDao.upsert(
                existing.copy(
                    cueCount = cues.size,
                    lineCount = lines.size,
                    clipCount = 0,
                    step1Done = false,
                    updatedAt = System.currentTimeMillis()
                )
            )
            lines
        }

    /** Import a translated SRT and attach it line-by-line to the script. */
    suspend fun importTranslatedSrt(project: String, uri: Uri): Int =
        withContext(Dispatchers.IO) {
            VdubPaths.ensureProject(project)
            val target = VdubPaths.translatedSrt(project)
            context.contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { input.copyTo(it) }
            } ?: error("Cannot open subtitle $uri")

            val translatedCues = SrtParser.parse(target)
            val script = readScriptRaw(project) ?: error("Import the original SRT first")

            val merged = script.lines.map { line ->
                val overlapping = translatedCues.filter { cue ->
                    val s = cue.startMs / 1000.0
                    val e = cue.endMs / 1000.0
                    e > line.start && s < line.end
                }
                if (overlapping.isEmpty()) line
                else line.copy(translated = overlapping.joinToString(" ") { it.text })
            }
            writeScriptRaw(project, merged, script.cueCount)

            val existing = projectDao.get(project) ?: ProjectEntity(name = project)
            projectDao.upsert(
                existing.copy(
                    translatedSrtPath = target.absolutePath,
                    updatedAt = System.currentTimeMillis()
                )
            )
            merged.count { !it.translated.isNullOrBlank() }
        }

    /** Export the merged lines as an SRT the user can translate by hand. */
    suspend fun exportScriptAsSrt(project: String): File = withContext(Dispatchers.IO) {
        val script = readScriptRaw(project) ?: error("No script yet")
        val out = File(VdubPaths.outDir(project), "${project}_for_translation.srt")
        out.parentFile?.mkdirs()
        out.writeText(SrtParser.toSrt(script.lines, useTranslated = false))
        out
    }

    // ---------------------------------------------------------------- script

    suspend fun writeScriptRaw(project: String, lines: List<ScriptLine>, cueCount: Int) =
        withContext(Dispatchers.IO) {
            VdubPaths.ensureProject(project)
            val payload = ScriptRaw(
                project = project,
                sourceVideo = VdubPaths.inputVideo(project).takeIf { it.exists() }?.name,
                sampleRate = AudioExtractor.TARGET_SAMPLE_RATE,
                padSec = ClipCutter.PAD_SEC,
                cueCount = cueCount,
                lines = lines
            )
            VdubPaths.scriptRaw(project)
                .writeText(json.encodeToString(ScriptRaw.serializer(), payload))
        }

    suspend fun readScriptRaw(project: String): ScriptRaw? = withContext(Dispatchers.IO) {
        val f = VdubPaths.scriptRaw(project)
        if (!f.exists()) return@withContext null
        runCatching { json.decodeFromString(ScriptRaw.serializer(), f.readText()) }.getOrNull()
    }

    // ------------------------------------------------------------ extraction

    suspend fun extractAudio(project: String, onProgress: (Float) -> Unit): File {
        val video = VdubPaths.inputVideo(project)
        return AudioExtractor.extractToWav16kMono(
            video = video,
            target = VdubPaths.orgAudio(project),
            onProgress = onProgress
        )
    }

    /** Full trim: (extract audio if needed) -> slice clips -> persist + mark S01. */
    suspend fun trimIntoClips(
        project: String,
        onExtractProgress: (Float) -> Unit,
        onClipProgress: (Int, Int) -> Unit
    ): ClipCutter.Result = withContext(Dispatchers.IO) {
        val script = readScriptRaw(project) ?: error("No subtitles imported yet")
        require(script.lines.isNotEmpty()) { "Script has no lines" }

        if (!VdubPaths.orgAudio(project).exists()) {
            extractAudio(project, onExtractProgress)
        } else {
            onExtractProgress(1f)
        }

        val result = ClipCutter.cut(project, script.lines, onProgress = onClipProgress)

        val withClips = script.lines.map {
            it.copy(clip = "clips/line_%04d.wav".format(it.id))
        }
        writeScriptRaw(project, withClips, script.cueCount)

        clipDao.clear(project)
        clipDao.insertAll(
            withClips.map { line ->
                ClipEntity(
                    id = "$project#${line.id}",
                    project = project,
                    lineId = line.id,
                    startSec = line.start,
                    endSec = line.end,
                    text = line.text,
                    wavPath = VdubPaths.clipFile(project, line.id).absolutePath,
                    speaker = line.speaker,
                    emotion = line.emotion
                )
            }
        )

        val existing = projectDao.get(project) ?: ProjectEntity(name = project)
        projectDao.upsert(
            existing.copy(
                clipCount = result.clipCount,
                lineCount = withClips.size,
                step1Done = true,
                updatedAt = System.currentTimeMillis()
            )
        )
        VdubPaths.markStepDone(project, 1)
        result
    }

    /** Reconcile DB with what's actually on disk (resume after reinstall). */
    suspend fun rehydrate(project: String): ProjectEntity = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val video = VdubPaths.inputVideo(project)
        val srt = VdubPaths.originalSrt(project)
        val translated = VdubPaths.translatedSrt(project)
        val script = readScriptRaw(project)
        val clipCount = VdubPaths.clipCount(project)
        val base = projectDao.get(project) ?: ProjectEntity(name = project)

        val entity = base.copy(
            videoPath = video.takeIf { it.exists() }?.absolutePath ?: base.videoPath,
            durationMs = if (video.exists() && base.durationMs == 0L) probeDurationMs(video)
            else base.durationMs,
            srtPath = srt.takeIf { it.exists() }?.absolutePath ?: base.srtPath,
            translatedSrtPath = translated.takeIf { it.exists() }?.absolutePath
                ?: base.translatedSrtPath,
            cueCount = script?.cueCount ?: base.cueCount,
            lineCount = script?.lines?.size ?: base.lineCount,
            clipCount = clipCount,
            step1Done = VdubPaths.isStepDone(project, 1) || clipCount > 0,
            updatedAt = System.currentTimeMillis()
        )
        projectDao.upsert(entity)
        entity
    }

    suspend fun pingServer(base: String?): Boolean {
        if (!base.isNullOrBlank()) downloadClient.serverBase = base
        return downloadClient.ping()
    }
}
