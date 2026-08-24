package com.azim.vdub.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Mood
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.azim.vdub.data.model.EmotionStyle
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine

private val EmotionColors = mapOf(
    "angry" to Color(0xFFFF6B6B),
    "happy" to Color(0xFFFFD54F),
    "sad" to Color(0xFF5B9CFF),
    "surprised" to Color(0xFFC77DFF),
    "fearful" to Color(0xFF9CCC65),
    "disgusted" to Color(0xFF4DD0E1),
    "neutral" to Color(0xFF8A8FA3),
    "other" to Color(0xFF8A8FA3)
)

fun emotionColor(e: String): Color =
    EmotionColors[e.lowercase()] ?: Color(0xFF8A8FA3)

@Composable
fun EmotionModelCard(
    present: Boolean,
    path: String,
    sizeBytes: Long,
    job: JobState,
    busy: Boolean,
    onDownload: () -> Unit
) {
    val running = job as? JobState.Running
    val downloading = running != null &&
        (running.label.contains("emotion", true) || running.label.startsWith("Verifying"))

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = if (present) MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)
        else MaterialTheme.colorScheme.error.copy(alpha = 0.14f)
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                if (present) "emotion2vec ready ✓" else "emotion2vec missing",
                style = MaterialTheme.typography.titleMedium,
                color = if (present) MaterialTheme.colorScheme.secondary
                else MaterialTheme.colorScheme.error
            )
            Spacer(Modifier.height(4.dp))
            Text(
                if (present) "%.0f MB · %s".format(sizeBytes / 1024.0 / 1024.0, path)
                else "The 355 MB model tags each line angry / happy / sad …, " +
                    "which drives how expressively it is spoken.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (!present) {
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
                    Text("Download model (355 MB)")
                }
            }
            if (downloading && running != null) {
                Spacer(Modifier.height(10.dp))
                Text(
                    running.label + if (running.detail.isNotBlank()) " · ${running.detail}" else "",
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { running.progress.coerceAtLeast(0f) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                )
            }
        }
    }
}

@Composable
fun EmotionSection(
    clipCount: Int,
    counts: Map<String, Int>,
    modelPresent: Boolean,
    busy: Boolean,
    job: JobState,
    onDetect: () -> Unit,
    onCancel: () -> Unit
) {
    SectionCard(
        number = "1",
        title = "Emotion Detection",
        subtitle = if (counts.isNotEmpty())
            "${counts.values.sum()} lines tagged"
        else "emotion2vec+ over each clip",
        done = counts.isNotEmpty()
    ) {
        Text(
            "Each clip is classified into one of nine emotions. The label sets " +
                "the delivery strength used later by the voice stage — " +
                "angry ${EmotionStyle.exaggeration("angry")}×, " +
                "happy ${EmotionStyle.exaggeration("happy")}×, " +
                "sad ${EmotionStyle.exaggeration("sad")}×.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(12.dp))

        Button(
            onClick = onDetect,
            enabled = !busy && modelPresent && clipCount > 0,
            modifier = Modifier
                .fillMaxWidth()
                .height(44.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.Mood, null, Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
            Text(if (counts.isEmpty()) "Detect Emotions" else "Re-detect")
        }

        if (counts.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            Row(
                Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                counts.entries.sortedByDescending { it.value }.forEach { (emotion, n) ->
                    AssistChip(
                        onClick = {},
                        label = { Text("$emotion  $n") },
                        leadingIcon = {
                            Box(
                                Modifier
                                    .size(10.dp)
                                    .clip(CircleShape)
                                    .background(emotionColor(emotion))
                            )
                        },
                        colors = AssistChipDefaults.assistChipColors(
                            labelColor = MaterialTheme.colorScheme.onSurface
                        )
                    )
                }
            }
        }

        when (job) {
            is JobState.Running -> if (job.label.startsWith("Detecting")) {
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "${job.label} · ${job.detail}",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f)
                    )
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { job.progress.coerceAtLeast(0f) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                )
            }
            is JobState.Failed -> {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✗ ${job.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
            is JobState.Done -> if (job.label == "Emotion tagged") {
                Spacer(Modifier.height(10.dp))
                Text(
                    "✓ ${job.detail}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            JobState.Idle -> Unit
        }
    }
}

/** Per-line list with a tap-to-override emotion picker. */
@Composable
fun EmotionLinesSection(
    lines: List<SpeakerLine>,
    speakerName: (String) -> String,
    onSetEmotion: (String, String) -> Unit,
    limit: Int = 60
) {
    if (lines.isEmpty()) return
    var editing by remember { mutableStateOf<String?>(null) }

    SectionCard(
        number = "2",
        title = "Lines",
        subtitle = "Tap an emotion to correct it",
        done = true
    ) {
        lines.take(limit).forEach { line ->
            Column(Modifier.fillMaxWidth()) {
                Row(
                    Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        line.utt.removePrefix("line_"),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.width(34.dp)
                    )
                    Text(
                        speakerName(line.spk),
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.width(64.dp)
                    )
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = emotionColor(line.emotion).copy(alpha = 0.20f),
                        modifier = Modifier.width(88.dp)
                    ) {
                        TextButton(
                            onClick = {
                                editing = if (editing == line.utt) null else line.utt
                            },
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                                horizontal = 6.dp, vertical = 2.dp
                            )
                        ) {
                            Text(
                                line.emotion,
                                style = MaterialTheme.typography.bodySmall,
                                color = emotionColor(line.emotion),
                                maxLines = 1
                            )
                        }
                    }
                    Spacer(Modifier.width(6.dp))
                    Text(
                        line.text,
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                if (editing == line.utt) {
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                            .padding(bottom = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        EmotionStyle.KNOWN.filter { it != "unk" }.forEach { e ->
                            AssistChip(
                                onClick = {
                                    onSetEmotion(line.utt, e)
                                    editing = null
                                },
                                label = { Text(e, style = MaterialTheme.typography.bodySmall) },
                                colors = AssistChipDefaults.assistChipColors(
                                    labelColor = emotionColor(e)
                                )
                            )
                        }
                    }
                }
            }
        }
        if (lines.size > limit) {
            Spacer(Modifier.height(6.dp))
            Text(
                "… and ${lines.size - limit} more (full list in out/script_emotion.json)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun NextStep4Button(enabled: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp),
        shape = RoundedCornerShape(14.dp)
    ) {
        Icon(Icons.Default.Movie, null)
        Spacer(Modifier.width(8.dp))
        Text("Next → Step 4: Translation", style = MaterialTheme.typography.titleMedium)
    }
}
