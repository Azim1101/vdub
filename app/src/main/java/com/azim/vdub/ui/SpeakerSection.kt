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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine

val SpeakerColors = listOf(
    Color(0xFF5B9CFF),  // blue
    Color(0xFF35D6A4),  // green
    Color(0xFFFFA33F),  // orange
    Color(0xFFC77DFF),  // violet
    Color(0xFFFF6B8A),  // pink
    Color(0xFF4DD0E1),  // cyan
    Color(0xFFFFD54F),  // amber
    Color(0xFF9CCC65)   // lime
)

fun speakerColor(index: Int): Color = SpeakerColors[index % SpeakerColors.size]

/** Model status + one-tap download. No PC, no adb. */
@Composable
fun ModelStatusCard(
    present: Boolean,
    path: String,
    sizeBytes: Long,
    busy: Boolean,
    job: JobState,
    onDownload: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = if (present) MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)
        else MaterialTheme.colorScheme.error.copy(alpha = 0.14f)
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                if (present) "campplus.onnx ready ✓" else "campplus.onnx missing",
                style = MaterialTheme.typography.titleMedium,
                color = if (present) MaterialTheme.colorScheme.secondary
                else MaterialTheme.colorScheme.error
            )
            Spacer(Modifier.height(4.dp))
            Text(
                if (present) {
                    "%.1f MB · %s".format(sizeBytes / 1024.0 / 1024.0, path)
                } else {
                    "The 28 MB CAM++ model is needed to tell voices apart. " +
                        "Download it in Settings — no PC needed."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Only this card's own job should drive its progress bar —
            // clustering runs in the section below and has its own.
            val running = job as? JobState.Running
            val downloading = running != null && (
                running.label.contains("campplus", ignoreCase = true) ||
                    running.label.startsWith("Verifying")
                )

            if (!present || downloading) {
                Spacer(Modifier.height(10.dp))
                Button(
                    onClick = onDownload,
                    enabled = !busy,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(44.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.CloudDownload, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(if (present) "Re-download model" else "Download model (28 MB)")
                }
            }

            if (downloading && running != null) {
                Spacer(Modifier.height(10.dp))
                Text(
                    "${running.label}" +
                        if (running.detail.isNotBlank()) " · ${running.detail}" else "",
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(6.dp))
                if (running.progress >= 0f) {
                    LinearProgressIndicator(
                        progress = { running.progress },
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

            if (!present && !downloading) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "Already have it? Copy it to:\n$path",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun SpeakerSection(
    speakers: List<SpeakerInfo>,
    clipCount: Int,
    embedCount: Int,
    threshold: Float,
    useTargetK: Boolean,
    targetK: Int,
    modelPresent: Boolean,
    hasEmbeds: Boolean,
    busy: Boolean,
    job: JobState,
    onThresholdChange: (Float) -> Unit,
    onUseTargetK: (Boolean) -> Unit,
    onTargetKChange: (Int) -> Unit,
    onExtract: () -> Unit,
    onRecluster: () -> Unit,
    onCancel: () -> Unit,
    onRename: (String, String) -> Unit
) {
    SectionCard(
        number = "1",
        title = "Speaker Diarization",
        subtitle = if (speakers.isNotEmpty())
            "${speakers.size} speakers detected across $clipCount clips"
        else "campplus embeddings → clustering",
        done = speakers.isNotEmpty()
    ) {
        Text(
            "Each clip becomes a 192-dim voice fingerprint, then clips with " +
                "similar fingerprints are grouped. Embeddings are cached, so " +
                "re-tuning below is instant.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(12.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onExtract,
                enabled = !busy && modelPresent && clipCount > 0,
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.GraphicEq, null, Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(if (hasEmbeds) "Re-extract" else "Extract & Cluster", maxLines = 1)
            }
            OutlinedButton(
                onClick = onRecluster,
                enabled = !busy && hasEmbeds,
                modifier = Modifier.height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Refresh, null, Modifier.size(18.dp))
                Spacer(Modifier.width(4.dp))
                Text("Re-cluster", maxLines = 1)
            }
        }

        if (hasEmbeds) {
            Spacer(Modifier.height(6.dp))
            Text(
                "$embedCount embeddings cached",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary
            )
        }

        // ---- tuning ----
        Spacer(Modifier.height(14.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            FilterChip(
                selected = !useTargetK,
                onClick = { onUseTargetK(false) },
                label = { Text("Auto (threshold)") },
                enabled = !busy
            )
            Spacer(Modifier.width(8.dp))
            FilterChip(
                selected = useTargetK,
                onClick = { onUseTargetK(true) },
                label = { Text("I know the count") },
                enabled = !busy
            )
        }

        Spacer(Modifier.height(10.dp))

        if (useTargetK) {
            Text(
                "Exactly $targetK speakers",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            Spacer(Modifier.height(4.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                (2..6).forEach { k ->
                    OutlinedButton(
                        onClick = { onTargetKChange(k) },
                        enabled = !busy,
                        modifier = Modifier
                            .weight(1f)
                            .height(38.dp),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text(
                            "$k",
                            color = if (k == targetK) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
            Spacer(Modifier.height(6.dp))
            Text(
                "Most reliable when you already know who is in the episode.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            Text(
                "Merge threshold: %.2f".format(threshold),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            Slider(
                value = threshold,
                onValueChange = onThresholdChange,
                valueRange = 0.10f..0.85f,
                enabled = !busy
            )
            Text(
                "Lower = merges more = fewer speakers.  Higher = stricter = more " +
                    "speakers. If you get 29 speakers instead of 3, move it DOWN.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // ---- progress ----
        when (job) {
            is JobState.Running -> {
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "${job.label}${if (job.detail.isNotBlank()) " · ${job.detail}" else ""}",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
                Spacer(Modifier.height(6.dp))
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
            is JobState.Failed -> {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✗ ${job.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
            is JobState.Done -> {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✓ ${job.label} · ${job.detail}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            JobState.Idle -> Unit
        }

        // ---- speaker list + rename ----
        if (speakers.isNotEmpty()) {
            Spacer(Modifier.height(16.dp))
            Text(
                "Edit names",
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(Modifier.height(8.dp))
            speakers.forEach { spk ->
                SpeakerRow(spk, busy, onRename)
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun SpeakerRow(
    speaker: SpeakerInfo,
    busy: Boolean,
    onRename: (String, String) -> Unit
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier
                .size(14.dp)
                .clip(CircleShape)
                .background(speakerColor(speaker.colorIndex))
        )
        Spacer(Modifier.width(10.dp))
        Column(Modifier.width(96.dp)) {
            Text(
                speaker.id,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                maxLines = 1
            )
            Text(
                "${speaker.clipCount} clips",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(Modifier.width(8.dp))
        OutlinedTextField(
            value = if (speaker.displayName == speaker.id) "" else speaker.displayName,
            onValueChange = { onRename(speaker.id, it) },
            modifier = Modifier.weight(1f),
            placeholder = { Text("name…", style = MaterialTheme.typography.bodySmall) },
            singleLine = true,
            enabled = !busy,
            shape = RoundedCornerShape(10.dp)
        )
    }
}

/** Preview of the tagged script. */
@Composable
fun ClipsPreviewSection(
    lines: List<SpeakerLine>,
    speakers: List<SpeakerInfo>,
    nameFor: (String) -> String,
    limit: Int = 40
) {
    if (lines.isEmpty()) return
    val colorOf = speakers.associate { it.id to it.colorIndex }

    SectionCard(
        number = "2",
        title = "Clips",
        subtitle = "${lines.size} lines tagged",
        done = true
    ) {
        lines.take(limit).forEach { line ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(vertical = 3.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(speakerColor(colorOf[line.spk] ?: 0))
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    line.utt.removePrefix("line_"),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    nameFor(line.spk),
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Medium,
                    color = speakerColor(colorOf[line.spk] ?: 0),
                    maxLines = 1,
                    modifier = Modifier.width(76.dp)
                )
                Text(
                    line.text,
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        if (lines.size > limit) {
            Spacer(Modifier.height(6.dp))
            Text(
                "… and ${lines.size - limit} more (full list in out/script_speakers.json)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun NextStep3Button(enabled: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp),
        shape = RoundedCornerShape(14.dp)
    ) {
        Icon(Icons.Default.AutoAwesome, null)
        Spacer(Modifier.width(8.dp))
        Text("Next → Step 3: Emotion", style = MaterialTheme.typography.titleMedium)
    }
}
