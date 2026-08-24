package com.azim.vdub.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.ContentCut
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.azim.vdub.data.model.JobState

/* ------------------------------------------------------------------ shared */

@Composable
fun SectionCard(
    number: String,
    title: String,
    subtitle: String? = null,
    done: Boolean = false,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .size(26.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(
                            if (done) MaterialTheme.colorScheme.secondary
                            else MaterialTheme.colorScheme.primary
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    if (done) {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSecondary,
                            modifier = Modifier.size(16.dp)
                        )
                    } else {
                        Text(
                            number,
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                    }
                }
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        title,
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    if (!subtitle.isNullOrBlank()) {
                        Text(
                            subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            content()
        }
    }
}

@Composable
private fun ActionChip(
    icon: ImageVector,
    label: String,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    primary: Boolean = false,
    onClick: () -> Unit
) {
    if (primary) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier.height(44.dp),
            shape = RoundedCornerShape(12.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp)
        ) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier.height(44.dp),
            shape = RoundedCornerShape(12.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = MaterialTheme.colorScheme.onSurface
            )
        ) {
            Icon(
                icon,
                contentDescription = null,
                modifier = Modifier.size(18.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(Modifier.width(6.dp))
            Text(
                label,
                style = MaterialTheme.typography.labelLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

/* ------------------------------------------------ 1. Video Upload group */

@Composable
fun VideoUploadSection(
    hasVideo: Boolean,
    url: String,
    serverBase: String,
    serverOnline: Boolean?,
    busy: Boolean,
    onPickGallery: () -> Unit,
    onPickDrive: () -> Unit,
    onPasteUrl: () -> Unit,
    onUrlChange: (String) -> Unit,
    onServerChange: (String) -> Unit,
    onPingServer: () -> Unit,
    onDownload: () -> Unit
) {
    SectionCard(
        number = "1",
        title = "Video Upload",
        subtitle = if (hasVideo) "input_video.mp4 ready" else "Gallery, URL or Drive",
        done = hasVideo
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ActionChip(
                Icons.Default.Folder, "Gallery",
                Modifier.weight(1f), enabled = !busy, onClick = onPickGallery
            )
            ActionChip(
                Icons.Default.Link, "URL",
                Modifier.weight(1f), enabled = !busy,
                onClick = onPasteUrl
            )
            ActionChip(
                Icons.Default.Storage, "Drive",
                Modifier.weight(1f), enabled = !busy, onClick = onPickDrive
            )
        }

        Spacer(Modifier.height(10.dp))

        OutlinedTextField(
            value = url,
            onValueChange = onUrlChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Video URL (iq.com, youtube, direct .mp4)") },
            placeholder = { Text("https://www.iq.com/play/...") },
            singleLine = true,
            shape = RoundedCornerShape(12.dp),
            enabled = !busy
        )

        Spacer(Modifier.height(8.dp))

        OutlinedTextField(
            value = serverBase,
            onValueChange = onServerChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("yt-dlp server (PhantomJS box)") },
            placeholder = { Text("https://xxxx.trycloudflare.com") },
            singleLine = true,
            shape = RoundedCornerShape(12.dp),
            enabled = !busy,
            supportingText = {
                Text(
                    when (serverOnline) {
                        true -> "Server online ✓"
                        false -> "Server unreachable — check the tunnel URL"
                        null -> "Needed for iq.com / DRM-free stream sites"
                    },
                    style = MaterialTheme.typography.bodySmall
                )
            }
        )

        Spacer(Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = onPingServer,
                enabled = !busy,
                modifier = Modifier.height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) { Text("Test") }
            ActionChip(
                Icons.Default.CloudDownload, "Download",
                Modifier.weight(1f),
                enabled = !busy && url.isNotBlank(),
                primary = true,
                onClick = onDownload
            )
        }
    }
}

/* -------------------------------------------- 2. Subtitles Upload group */

@Composable
fun SubtitlesSection(
    cueCount: Int,
    lineCount: Int,
    mergeGapMs: Long,
    busy: Boolean,
    onUploadSrt: () -> Unit,
    onAutoAsr: () -> Unit,
    onMergeGapChange: (Long) -> Unit
) {
    SectionCard(
        number = "2",
        title = "Subtitles Upload",
        subtitle = if (lineCount > 0) "$cueCount cues → $lineCount lines"
        else "SRT file, or auto-transcribe",
        done = lineCount > 0
    ) {
        if (lineCount > 0) {
            Surface(
                shape = RoundedCornerShape(10.dp),
                color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    "$cueCount cues merged into $lineCount speakable lines " +
                        "→ out/script_raw.json",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.padding(10.dp)
                )
            }
            Spacer(Modifier.height(10.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ActionChip(
                Icons.Default.UploadFile, "Upload SRT",
                Modifier.weight(1f), enabled = !busy, primary = true, onClick = onUploadSrt
            )
            ActionChip(
                Icons.Default.Mic, "Auto ASR",
                Modifier.weight(1f), enabled = !busy, onClick = onAutoAsr
            )
        }

        if (cueCount > 0) {
            Spacer(Modifier.height(12.dp))
            Text(
                "Merge gap: ${mergeGapMs} ms — raise it to fold more cues into " +
                    "each spoken line, lower it to keep them separate.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(6.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(200L, 400L, 800L, 1200L, 2000L).forEach { gap ->
                    OutlinedButton(
                        onClick = { onMergeGapChange(gap) },
                        enabled = !busy,
                        modifier = Modifier
                            .weight(1f)
                            .height(38.dp),
                        shape = RoundedCornerShape(10.dp),
                        contentPadding = androidx.compose.foundation.layout
                            .PaddingValues(horizontal = 2.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = if (gap == mergeGapMs)
                                MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    ) {
                        Text("$gap", style = MaterialTheme.typography.bodySmall, maxLines = 1)
                    }
                }
            }
        }
    }
}

/* ---------------------------------------- 3. Translation Subtitles group */

@Composable
fun TranslationSection(
    lineCount: Int,
    translatedCount: Int,
    busy: Boolean,
    onManualUpload: () -> Unit,
    onAutoNllb: () -> Unit,
    onDownloadForManual: () -> Unit
) {
    SectionCard(
        number = "3",
        title = "Translation Subtitles",
        subtitle = if (translatedCount > 0) "$translatedCount / $lineCount lines translated"
        else "Manual, auto NLLB, or export to translate",
        done = translatedCount > 0 && translatedCount >= lineCount
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ActionChip(
                Icons.Default.UploadFile, "Manual",
                Modifier.weight(1f), enabled = !busy, onClick = onManualUpload
            )
            ActionChip(
                Icons.Default.SmartToy, "Auto NLLB",
                Modifier.weight(1f), enabled = !busy, onClick = onAutoNllb
            )
            ActionChip(
                Icons.Default.Download, "Export",
                Modifier.weight(1f),
                enabled = !busy && lineCount > 0,
                onClick = onDownloadForManual
            )
        }
        if (lineCount > 0 && translatedCount in 1 until lineCount) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { translatedCount.toFloat() / lineCount },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp))
            )
        }
        Spacer(Modifier.height(6.dp))
        Text(
            "Translation is optional for Step 1 — clips can be cut from the " +
                "original timing and translated later in Step 4.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/* --------------------------------------------------------- Trim section */

@Composable
fun TrimSection(
    lineCount: Int,
    clipCount: Int,
    clipsSizeLabel: String,
    canTrim: Boolean,
    busy: Boolean,
    job: JobState,
    onTrim: () -> Unit,
    onCancel: () -> Unit
) {
    SectionCard(
        number = "✂",
        title = "Audio Trim — Choti Clips",
        subtitle = if (clipCount > 0) "$clipCount audio clips · $clipsSizeLabel"
        else "Cut org_audio.wav into per-line clips",
        done = clipCount > 0
    ) {
        Text(
            "Audio is decoded with MediaCodec (no ffmpeg) into 16 kHz mono, " +
                "then sliced sample-by-sample with ±0.2 s padding — " +
                "the numpy-slicing approach, no per-clip subprocess.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(6.dp))
        Text(
            "Note: the video itself is never cut. Diarization, emotion and TTS " +
                "only read audio, and Step 6 muxes the dubbed track onto the " +
                "original full-length mp4 — so input_video.mp4 stays intact.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.secondary
        )
        Spacer(Modifier.height(10.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ActionChip(
                Icons.Default.ContentCut,
                if (lineCount > 0) "Trim Karo → $lineCount Audio Clips" else "Trim Karo",
                Modifier.weight(1f),
                enabled = canTrim,
                primary = true,
                onClick = onTrim
            )
            if (busy) {
                TextButton(onClick = onCancel) { Text("Cancel") }
            }
        }

        when (job) {
            is JobState.Running -> {
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        strokeWidth = 2.dp
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        buildString {
                            append(job.label)
                            if (job.detail.isNotBlank()) append(" · ").append(job.detail)
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
                Spacer(Modifier.height(8.dp))
                if (job.progress >= 0f) {
                    LinearProgressIndicator(
                        progress = { job.progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp))
                    )
                } else {
                    LinearProgressIndicator(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .clip(RoundedCornerShape(4.dp))
                    )
                }
            }
            is JobState.Done -> {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✓ ${job.label}${if (job.detail.isNotBlank()) " · ${job.detail}" else ""}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            is JobState.Failed -> {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✗ ${job.label}: ${job.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
            JobState.Idle -> Unit
        }
    }
}

@Composable
fun NextStepButton(
    enabled: Boolean,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp),
        shape = RoundedCornerShape(14.dp)
    ) {
        Icon(Icons.Default.Translate, contentDescription = null)
        Spacer(Modifier.width(8.dp))
        Text("Next → Step 2: Diarization", style = MaterialTheme.typography.titleMedium)
    }
}
