package com.azim.vdub.data.repo

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import com.azim.vdub.audio.AudioExtractor
import com.azim.vdub.audio.ClipCutter
import com.azim.vdub.audio.ChatterboxTts
import com.azim.vdub.audio.DubTimeline
import com.azim.vdub.audio.EmotionClassifier
import com.azim.vdub.audio.VideoMuxer
import com.azim.vdub.audio.VoiceEngine
import com.azim.vdub.audio.WavIo
import com.azim.vdub.audio.resampleLinear
import com.azim.vdub.audio.SpeakerCluster
import com.azim.vdub.audio.SpeakerEmbedder
import com.azim.vdub.core.VdubPaths
import com.azim.vdub.data.local.ClipDao
import com.azim.vdub.data.local.ClipEntity
import com.azim.vdub.data.local.ProjectDao
import com.azim.vdub.data.local.ProjectEntity
import com.azim.vdub.data.model.ScriptLine
import com.azim.vdub.data.model.ScriptRaw
import com.azim.vdub.data.model.EmotionScript
import com.azim.vdub.data.model.EmotionStyle
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.model.TranslatedScript
import com.azim.vdub.data.model.TranslationSource
import com.azim.vdub.data.model.EmotionStyle as Style
import com.azim.vdub.data.model.SpeakerScript
import com.azim.vdub.data.model.SrtCue
import com.azim.vdub.data.model.VideoSource
import com.azim.vdub.net.DownloadClient
import com.azim.vdub.net.VideoResolver
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

private const val WAV_MIN_BYTES = 44L + 16_000 * 2 * 1   // header + ~1 s at 16 kHz

@Singleton
class ProjectRepository @Inject constructor(
    @ApplicationContext private val context: Context,
    private val projectDao: ProjectDao,
    private val clipDao: ClipDao,
    private val downloadClient: DownloadClient,
    private val videoResolver: VideoResolver
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

    /**
     * Resolve a page/URL to a media link on-device, then stream it down.
     * Throws a human-readable message for sites that need a JS engine or DRM.
     */
    suspend fun downloadVideoFromUrl(
        project: String,
        url: String,
        onProgress: (Long, Long) -> Unit = { _, _ -> }
    ): File = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val target = VdubPaths.inputVideo(project)

        when (val res = videoResolver.resolve(url)) {
            is VideoResolver.Resolution.Direct -> {
                downloadClient.downloadDirect(res.url, target, onProgress)
            }
            is VideoResolver.Resolution.Unsupported -> error(
                "Cannot download from ${res.site}: ${res.reason}.\n\n" +
                    "Download it on a PC (yt-dlp works there), copy the file to " +
                    "your phone, then use Gallery."
            )
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

    /**
     * Reconcile DB with what's actually on disk (resume after reinstall).
     *
     * Disk is the source of truth in both directions: if a file has been
     * deleted, the stale DB path is cleared rather than kept. Falling back to
     * the old value made a removed video look present, so the player pointed
     * at a file that was not there.
     */
    suspend fun rehydrate(project: String): ProjectEntity = withContext(Dispatchers.IO) {
        VdubPaths.ensureProject(project)
        val video = VdubPaths.inputVideo(project)
        val srt = VdubPaths.originalSrt(project)
        val translated = VdubPaths.translatedSrt(project)
        val script = readScriptRaw(project)
        val clipCount = VdubPaths.clipCount(project)
        val base = projectDao.get(project) ?: ProjectEntity(name = project)

        val hasVideo = video.exists() && video.length() > 0
        val entity = base.copy(
            videoPath = if (hasVideo) video.absolutePath else null,
            videoSource = if (hasVideo) base.videoSource else VideoSource.NONE.name,
            sourceUrl = if (hasVideo) base.sourceUrl else null,
            durationMs = if (hasVideo) {
                if (base.durationMs == 0L) probeDurationMs(video) else base.durationMs
            } else 0L,
            srtPath = srt.takeIf { it.exists() }?.absolutePath,
            translatedSrtPath = translated.takeIf { it.exists() }?.absolutePath,
            cueCount = script?.cueCount ?: 0,
            lineCount = script?.lines?.size ?: 0,
            clipCount = clipCount,
            step1Done = VdubPaths.isStepDone(project, 1) && clipCount > 0,
            updatedAt = System.currentTimeMillis()
        )
        projectDao.upsert(entity)
        entity
    }

    /** Remove the video and anything derived from it, keeping subtitles. */
    suspend fun clearVideo(project: String) = withContext(Dispatchers.IO) {
        VdubPaths.inputVideo(project).delete()
        VdubPaths.orgAudio(project).delete()
        VdubPaths.clipsDir(project).listFiles()?.forEach { it.delete() }
        VdubPaths.clearStep(project, 1)
        clipDao.clear(project)
        projectDao.get(project)?.let {
            projectDao.upsert(
                it.copy(
                    videoPath = null,
                    videoSource = VideoSource.NONE.name,
                    sourceUrl = null,
                    durationMs = 0L,
                    clipCount = 0,
                    step1Done = false,
                    updatedAt = System.currentTimeMillis()
                )
            )
        }
    }

    /**
     * Delete everything belonging to a project — files, clips, markers and
     * DB rows — so a stuck or half-finished run can be started over.
     */
    suspend fun resetProject(project: String) = withContext(Dispatchers.IO) {
        val dir = VdubPaths.projectDir(project)
        if (dir.exists()) dir.deleteRecursively()
        clipDao.clear(project)
        projectDao.delete(project)
        VdubPaths.ensureProject(project)
    }

    // ------------------------------------------------------- Step 2: speakers

    /**
     * Run campplus over every clip and cache the embeddings.
     *
     * Embeddings are the expensive part (190 ONNX inferences); clustering is
     * milliseconds. Caching to speaker_embeds.bin means re-tuning the
     * threshold is instant instead of a full re-run.
     */
    suspend fun embedSpeakers(
        project: String,
        onProgress: (Int, Int) -> Unit
    ): LinkedHashMap<String, FloatArray> = withContext(Dispatchers.Default) {
        val script = readScriptRaw(project) ?: error("Run Step 1 first")
        require(script.lines.isNotEmpty()) { "No lines to embed" }

        val clips = script.lines.map { VdubPaths.clipFile(project, it.id) }
        val missing = clips.count { !it.exists() }
        check(missing == 0) {
            "$missing of ${clips.size} clips are missing — re-run the Step 1 trim."
        }

        val embeds = SpeakerEmbedder.open().use { it.embedAll(clips, onProgress) }
        check(embeds.isNotEmpty()) { "campplus produced no embeddings" }
        writeEmbeds(project, embeds)
        embeds
    }

    /** Cluster cached (or freshly computed) embeddings into speakers. */
    suspend fun clusterSpeakers(
        project: String,
        threshold: Float,
        targetK: Int?,
        embeds: LinkedHashMap<String, FloatArray>? = null
    ): SpeakerScript = withContext(Dispatchers.Default) {
        val script = readScriptRaw(project) ?: error("Run Step 1 first")
        val vectors = embeds ?: readEmbeds(project)
            ?: error("No embeddings cached — run Extract first")

        val order = script.lines.filter { vectors.containsKey(clipName(it.id)) }
        val list = order.map { vectors.getValue(clipName(it.id)) }
        check(list.isNotEmpty()) { "No embeddings match the script" }

        val result = if (targetK != null) SpeakerCluster.clusterToK(list, targetK)
        else SpeakerCluster.cluster(list, threshold)

        val previous = readSpeakerScript(project)
        val lines = order.mapIndexed { i, line ->
            SpeakerLine(
                utt = clipName(line.id),
                start = line.start,
                end = line.end,
                text = line.text,
                spk = "Speaker ${result.labels[i] + 1}",
                emotion = line.emotion ?: "NEUTRAL",
                hi = line.translated.orEmpty()
            )
        }

        val speakerScript = SpeakerScript(
            project = project,
            speakerCount = result.speakerCount,
            threshold = threshold,
            // keep any names the user already typed
            names = previous?.names.orEmpty(),
            lines = lines
        )
        writeSpeakerScript(project, speakerScript)

        // mirror the speaker back into script_raw.json so later steps see it
        val byUtt = lines.associateBy { it.utt }
        writeScriptRaw(
            project,
            script.lines.map { l -> l.copy(speaker = byUtt[clipName(l.id)]?.spk) },
            script.cueCount
        )
        VdubPaths.markStepDone(project, 2)
        speakerScript
    }

    suspend fun renameSpeaker(project: String, speakerId: String, name: String): SpeakerScript? =
        withContext(Dispatchers.IO) {
            val current = readSpeakerScript(project) ?: return@withContext null
            val names = current.names.toMutableMap()
            if (name.isBlank()) names.remove(speakerId) else names[speakerId] = name.trim()
            val updated = current.copy(names = names)
            writeSpeakerScript(project, updated)
            updated
        }

    // ------------------------------------------------------- Step 3: emotion

    /**
     * Classify every clip and merge the label into the speaker script.
     * Runs after diarization so one file carries both, which is what the
     * TTS stage reads.
     */
    suspend fun detectEmotions(
        project: String,
        onProgress: (Int, Int) -> Unit
    ): EmotionScript = withContext(Dispatchers.Default) {
        val speakers = readSpeakerScript(project)
            ?: error("Run Step 2 (speakers) first")
        require(speakers.lines.isNotEmpty()) { "No lines to classify" }

        val clips = speakers.lines.map { File(VdubPaths.clipsDir(project), it.utt + ".wav") }
        val missing = clips.count { !it.exists() }
        check(missing == 0) {
            "$missing of ${clips.size} clips are missing — re-run the Step 1 trim."
        }

        val results = EmotionClassifier.open().use { it.classifyAll(clips, onProgress) }
        check(results.isNotEmpty()) { "emotion2vec produced no results" }

        val lines = speakers.lines.map { line ->
            val r = results[line.utt]
            if (r == null) line
            else line.copy(
                emotion = EmotionStyle.normalise(r.label),
                emotionScore = r.confidence
            )
        }

        val counts = lines.groupingBy { it.emotion }.eachCount()
        val script = EmotionScript(project = project, counts = counts, lines = lines)
        withContext(Dispatchers.IO) {
            VdubPaths.ensureProject(project)
            VdubPaths.scriptEmotion(project)
                .writeText(json.encodeToString(EmotionScript.serializer(), script))
        }

        // keep script_speakers.json authoritative too
        writeSpeakerScript(project, speakers.copy(lines = lines))

        // and mirror into script_raw.json for later steps
        readScriptRaw(project)?.let { raw ->
            val byUtt = lines.associateBy { it.utt }
            writeScriptRaw(
                project,
                raw.lines.map { l -> l.copy(emotion = byUtt[clipName(l.id)]?.emotion) },
                raw.cueCount
            )
        }

        VdubPaths.markStepDone(project, 3)
        script
    }

    /** Manual override — the model is a guess, the user is not. */
    suspend fun setLineEmotion(project: String, utt: String, emotion: String): EmotionScript? =
        withContext(Dispatchers.IO) {
            val current = readEmotionScript(project) ?: return@withContext null
            val lines = current.lines.map {
                if (it.utt == utt) it.copy(
                    emotion = EmotionStyle.normalise(emotion),
                    emotionScore = 1f
                ) else it
            }
            val updated = current.copy(
                lines = lines,
                counts = lines.groupingBy { it.emotion }.eachCount()
            )
            VdubPaths.scriptEmotion(project)
                .writeText(json.encodeToString(EmotionScript.serializer(), updated))
            readSpeakerScript(project)?.let { writeSpeakerScript(project, it.copy(lines = lines)) }
            updated
        }

    suspend fun readEmotionScript(project: String): EmotionScript? = withContext(Dispatchers.IO) {
        val f = VdubPaths.scriptEmotion(project)
        if (!f.exists()) return@withContext null
        runCatching { json.decodeFromString(EmotionScript.serializer(), f.readText()) }.getOrNull()
    }

    // --------------------------------------------------- Step 4: translation

    /** Lines carrying speaker + emotion, ready to be translated. */
    private suspend fun linesForTranslation(project: String): List<SpeakerLine> =
        readTranslatedScript(project)?.lines
            ?: readEmotionScript(project)?.lines
            ?: readSpeakerScript(project)?.lines
            ?: error("Run the earlier steps first")

    /**
     * Import an already-translated SRT and skip machine translation entirely.
     * Cues are matched to lines by time overlap, so the file does not have to
     * share our line boundaries — a translator working from the exported SRT
     * may well have merged or split a few.
     */
    suspend fun importTranslationSrt(project: String, uri: Uri): TranslatedScript =
        withContext(Dispatchers.IO) {
            val target = VdubPaths.translatedSrt(project)
            target.parentFile?.mkdirs()
            context.contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { input.copyTo(it) }
            } ?: error("Cannot open $uri")

            val cues = SrtParser.parse(target)
            check(cues.isNotEmpty()) { "No subtitles found in that file" }

            val lines = linesForTranslation(project).map { line ->
                val hit = cues.filter { c ->
                    c.endMs / 1000.0 > line.start && c.startMs / 1000.0 < line.end
                }
                if (hit.isEmpty()) line
                else line.copy(hi = hit.joinToString(" ") { it.text }.trim())
            }
            saveTranslated(project, lines, TranslationSource.UPLOADED_SRT)
        }

    /**
     * Import a translated JSON — the file produced by Export, with `hi` filled
     * in. Matched by `utt`, so timing edits cannot misalign it.
     */
    suspend fun importTranslationJson(project: String, uri: Uri): TranslatedScript =
        withContext(Dispatchers.IO) {
            val text = context.contentResolver.openInputStream(uri)?.use {
                it.readBytes().decodeToString()
            } ?: error("Cannot open $uri")

            val incoming = runCatching {
                json.decodeFromString(TranslatedScript.serializer(), text).lines
            }.getOrElse {
                runCatching {
                    json.decodeFromString(
                        kotlinx.serialization.builtins.ListSerializer(SpeakerLine.serializer()),
                        text
                    )
                }.getOrElse {
                    error("Not a vdub translation JSON — expected the exported file")
                }
            }
            check(incoming.isNotEmpty()) { "That JSON has no lines" }

            val byUtt = incoming.associateBy { it.utt }
            val lines = linesForTranslation(project).map { line ->
                val hi = byUtt[line.utt]?.hi.orEmpty()
                if (hi.isBlank()) line else line.copy(hi = hi.trim())
            }
            saveTranslated(project, lines, TranslationSource.UPLOADED_JSON)
        }

    /** Export lines as JSON for hand translation — `hi` left blank to fill in. */
    suspend fun exportTranslationJson(project: String): File = withContext(Dispatchers.IO) {
        val lines = linesForTranslation(project)
        val out = File(VdubPaths.outDir(project), "${project}_to_translate.json")
        out.parentFile?.mkdirs()
        out.writeText(
            json.encodeToString(
                TranslatedScript.serializer(),
                TranslatedScript(
                    project = project,
                    source = TranslationSource.NONE.name,
                    translatedCount = 0,
                    lines = lines
                )
            )
        )
        out
    }

    /** Export as SRT for translators who prefer subtitle tools. */
    suspend fun exportTranslationSrt(project: String): File = withContext(Dispatchers.IO) {
        val lines = linesForTranslation(project)
        val out = File(VdubPaths.outDir(project), "${project}_to_translate.srt")
        out.parentFile?.mkdirs()
        out.writeText(
            buildString {
                lines.forEachIndexed { i, l ->
                    append(i + 1).append('\n')
                    append(srtTime(l.start)).append(" --> ").append(srtTime(l.end)).append('\n')
                    append(l.text).append("\n\n")
                }
            }
        )
        out
    }

    suspend fun setLineTranslation(project: String, utt: String, hi: String): TranslatedScript? =
        withContext(Dispatchers.IO) {
            val current = readTranslatedScript(project) ?: return@withContext null
            val lines = current.lines.map { if (it.utt == utt) it.copy(hi = hi) else it }
            saveTranslated(project, lines, TranslationSource.MANUAL_EDIT)
        }

    private suspend fun saveTranslated(
        project: String,
        lines: List<SpeakerLine>,
        source: TranslationSource
    ): TranslatedScript = withContext(Dispatchers.IO) {
        val done = lines.count { it.hi.isNotBlank() }
        val script = TranslatedScript(
            project = project,
            source = source.name,
            translatedCount = done,
            lines = lines
        )
        VdubPaths.ensureProject(project)
        VdubPaths.scriptTranslated(project)
            .writeText(json.encodeToString(TranslatedScript.serializer(), script))

        // Only complete once every line has text; a partial upload must not
        // let the pipeline move on and silently drop lines.
        if (done == lines.size && done > 0) VdubPaths.markStepDone(project, 4)
        else VdubPaths.clearStep(project, 4)
        script
    }

    suspend fun readTranslatedScript(project: String): TranslatedScript? =
        withContext(Dispatchers.IO) {
            val f = VdubPaths.scriptTranslated(project)
            if (!f.exists()) return@withContext null
            runCatching {
                json.decodeFromString(TranslatedScript.serializer(), f.readText())
            }.getOrNull()
        }

    /**
     * Reference clips for a speaker, longest first.
     *
     * Zero-shot cloning quality depends heavily on the reference: a two-word
     * clip carries almost no timbre. Step 5 should enrol from the longest few
     * rather than whichever line happens to be first.
     */
    suspend fun referenceClipsFor(
        project: String,
        speaker: String,
        limit: Int = 3
    ): List<File> = withContext(Dispatchers.IO) {
        val lines = readSpeakerScript(project)?.lines.orEmpty()
            .filter { it.spk == speaker }
        lines.sortedByDescending { it.end - it.start }
            .asSequence()
            .map { File(VdubPaths.clipsDir(project), it.utt + ".wav") }
            .filter { it.exists() && it.length() > WAV_MIN_BYTES }
            .take(limit)
            .toList()
    }

    /** Distinct speakers in the order they first appear. */
    suspend fun speakersOf(project: String): List<String> = withContext(Dispatchers.IO) {
        readSpeakerScript(project)?.lines.orEmpty().map { it.spk }.distinct()
    }

    // ------------------------------------------------------ Step 5: speaking

    data class SpeakProgress(
        val done: Int,
        val total: Int,
        val line: String,
        val speaker: String
    )

    /**
     * Speak every line in its speaker's cloned voice.
     *
     * Resumable: a line whose wav already exists is skipped, so a run
     * interrupted after two hours continues instead of restarting. Speakers
     * are enrolled once and reused, since encoding a reference is as expensive
     * as generating a short line.
     */
    suspend fun speakAll(
        project: String,
        engineId: String,
        onProgress: (SpeakProgress) -> Unit
    ): Int = withContext(Dispatchers.Default) {
        val script = readTranslatedScript(project)
            ?: error("Run Step 4 (translation) first")
        val lines = script.lines.filter { it.hi.isNotBlank() }
        check(lines.isNotEmpty()) { "No translated lines to speak" }

        VdubPaths.hiClipsDir(project).mkdirs()
        val paths = VoiceEngine.pathsFor(engineId)

        var spoken = 0
        ChatterboxTts.open(paths).use { tts ->
            val voices = HashMap<String, ChatterboxTts.SpeakerVoice>()

            lines.forEachIndexed { index, line ->
                coroutineContext.ensureActive()
                val id = line.utt.removePrefix("line_").toIntOrNull() ?: index
                val target = VdubPaths.hiClipFile(project, id)

                onProgress(SpeakProgress(index, lines.size, line.hi.take(40), line.spk))

                if (target.exists() && target.length() > WAV_MIN_BYTES) {
                    spoken++
                    return@forEachIndexed          // already done, resume past it
                }

                val voice = voices.getOrPut(line.spk) {
                    // Enrol from several of the speaker's clips joined together,
                    // not just the longest one: a single short line carries far
                    // less timbre, and this is the only thing the clone is built
                    // from. Matches the reference seconds shown in the UI.
                    val refs = referenceClipsFor(project, line.spk, limit = 3)
                    val ref = when {
                        refs.isNotEmpty() -> buildReference(project, line.spk, refs)
                        paths.defaultVoice.exists() -> paths.defaultVoice
                        else -> error(
                            "No usable reference audio for ${line.spk} — its clips " +
                                "are missing or too short to clone from."
                        )
                    }
                    tts.enrol(ref)
                }

                val wav = tts.speak(
                    text = line.hi,
                    voice = voice,
                    language = "hi",
                    exaggeration = Style.exaggeration(line.emotion)
                )
                WavIo.writePcm16(
                    target,
                    floatsToPcm16(wav),
                    ChatterboxTts.SAMPLE_RATE,
                    1
                )
                spoken++
            }
        }
        onProgress(SpeakProgress(lines.size, lines.size, "", ""))
        if (spoken == lines.size) VdubPaths.markStepDone(project, 5)
        spoken
    }

    /**
     * Fit the spoken clips to the original timing and mux them onto the video.
     *
     * @param keepBackground mixes the original audio underneath, so music and
     *        effects survive. Without separation this also leaves the original
     *        dialogue faintly audible, which is why it defaults off.
     */
    suspend fun buildDubbedVideo(
        project: String,
        keepBackground: Boolean = false,
        onProgress: (String, Float) -> Unit
    ): File = withContext(Dispatchers.Default) {
        val script = readTranslatedScript(project)
            ?: error("Run Step 4 (translation) first")
        val video = VdubPaths.inputVideo(project)
        check(video.exists()) { "input_video.mp4 is missing" }

        val sr = ChatterboxTts.SAMPLE_RATE
        onProgress("Loading clips", 0f)

        val clips = script.lines.mapNotNull { line ->
            val id = line.utt.removePrefix("line_").toIntOrNull() ?: return@mapNotNull null
            val f = VdubPaths.hiClipFile(project, id)
            if (!f.exists() || f.length() <= WAV_MIN_BYTES) return@mapNotNull null
            DubTimeline.Clip(id, line.start, line.end, readWavFloat(f))
        }
        check(clips.isNotEmpty()) { "No spoken clips yet — run Speak first" }

        onProgress("Fitting timing", 0.2f)
        val totalSec = script.lines.maxOf { it.end } + 2.0
        val assembled = DubTimeline.assemble(clips, totalSec, sr)

        var track = assembled.samples
        if (keepBackground) {
            onProgress("Mixing background", 0.35f)
            val org = VdubPaths.orgAudio(project)
            if (org.exists()) {
                val bed = resampleLinear(readWavFloat(org), 16_000, sr)
                track = DubTimeline.mix(track, bed)
            }
        }

        onProgress("Writing video", 0.5f)
        val out = VdubPaths.dubbedVideo(project)
        VideoMuxer.mux(video, track, sr, out) { p ->
            onProgress("Writing video", 0.5f + 0.5f * p)
        }

        VdubPaths.markStepDone(project, 6)
        out
    }

    /**
     * Concatenate a speaker's reference clips into one wav for enrolment.
     *
     * A short silence is inserted between them so two unrelated lines do not
     * run into each other as one impossible utterance. Cached per speaker, so
     * the joins happen once rather than per line.
     */
    private fun buildReference(project: String, speaker: String, refs: List<File>): File {
        val safe = speaker.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val target = File(VdubPaths.outDir(project), "ref_$safe.wav")
        if (target.exists() && target.length() > WAV_MIN_BYTES) return target
        if (refs.size == 1) return refs.first()

        val fmt = WavIo.readFormat(refs.first())
        val gap = ByteArray(fmt.sampleRate / 5 * 2)      // 200 ms of silence
        val out = java.io.ByteArrayOutputStream()
        refs.forEachIndexed { i, f ->
            runCatching {
                val info = WavIo.readFormat(f)
                if (info.sampleRate != fmt.sampleRate) return@runCatching
                val pcm = ByteArray(info.dataBytes.toInt())
                java.io.RandomAccessFile(f, "r").use { raf ->
                    raf.seek(info.dataOffset)
                    raf.readFully(pcm)
                }
                if (i > 0) out.write(gap)
                out.write(pcm)
            }
        }
        val bytes = out.toByteArray()
        if (bytes.isEmpty()) return refs.first()
        WavIo.writePcm16(target, bytes, fmt.sampleRate, fmt.channels)
        return target
    }

    fun spokenClipCount(project: String): Int =
        VdubPaths.hiClipsDir(project)
            .listFiles(java.io.FileFilter { f ->
                f.extension == "wav" && f.length() > WAV_MIN_BYTES
            })?.size ?: 0

    private fun readWavFloat(f: File): FloatArray {
        val fmt = WavIo.readFormat(f)
        val bytesPerFrame = fmt.channels * 2
        val frames = (fmt.dataBytes / bytesPerFrame).toInt()
        if (frames <= 0) return FloatArray(0)
        val pcm = ByteArray(fmt.dataBytes.toInt())
        java.io.RandomAccessFile(f, "r").use { raf ->
            raf.seek(fmt.dataOffset)
            raf.readFully(pcm)
        }
        val out = FloatArray(frames)
        var p = 0
        for (i in 0 until frames) {
            var acc = 0
            for (c in 0 until fmt.channels) {
                val lo = pcm[p].toInt() and 0xFF
                val hi = pcm[p + 1].toInt()
                acc += (hi shl 8) or lo
                p += 2
            }
            out[i] = (acc / fmt.channels) / 32768f
        }
        return out
    }

    private fun floatsToPcm16(samples: FloatArray): ByteArray {
        val out = ByteArray(samples.size * 2)
        var j = 0
        for (s in samples) {
            val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
            out[j++] = (v and 0xFF).toByte()
            out[j++] = ((v shr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun srtTime(sec: Double): String {
        val ms = (sec * 1000).toLong().coerceAtLeast(0)
        return "%02d:%02d:%02d,%03d".format(
            ms / 3_600_000, (ms % 3_600_000) / 60_000, (ms % 60_000) / 1000, ms % 1000
        )
    }

    suspend fun readSpeakerScript(project: String): SpeakerScript? = withContext(Dispatchers.IO) {
        val f = VdubPaths.scriptSpeakers(project)
        if (!f.exists()) return@withContext null
        runCatching { json.decodeFromString(SpeakerScript.serializer(), f.readText()) }.getOrNull()
    }

    private suspend fun writeSpeakerScript(project: String, script: SpeakerScript) =
        withContext(Dispatchers.IO) {
            VdubPaths.ensureProject(project)
            VdubPaths.scriptSpeakers(project)
                .writeText(json.encodeToString(SpeakerScript.serializer(), script))
        }

    private fun clipName(id: Int) = "line_%04d".format(id)

    /** Compact binary cache: [count][dim] then name-length/name/floats. */
    private fun writeEmbeds(project: String, embeds: Map<String, FloatArray>) {
        val f = VdubPaths.speakerEmbeds(project)
        f.parentFile?.mkdirs()
        java.io.DataOutputStream(f.outputStream().buffered()).use { out ->
            out.writeInt(embeds.size)
            out.writeInt(embeds.values.firstOrNull()?.size ?: 0)
            embeds.forEach { (name, vec) ->
                out.writeUTF(name)
                vec.forEach(out::writeFloat)
            }
        }
    }

    fun readEmbeds(project: String): LinkedHashMap<String, FloatArray>? {
        val f = VdubPaths.speakerEmbeds(project)
        if (!f.exists()) return null
        return runCatching {
            java.io.DataInputStream(f.inputStream().buffered()).use { input ->
                val count = input.readInt()
                val dim = input.readInt()
                val map = LinkedHashMap<String, FloatArray>(count)
                repeat(count) {
                    val name = input.readUTF()
                    val vec = FloatArray(dim) { input.readFloat() }
                    map[name] = vec
                }
                map
            }
        }.getOrNull()
    }

    fun hasEmbeds(project: String) = VdubPaths.speakerEmbeds(project).exists()

}
