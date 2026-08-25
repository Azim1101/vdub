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
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Movie
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.azim.vdub.data.model.JobState

/** Which engine is selected, and whether it is actually on disk. */
@Composable
fun VoiceEngineCard(
    engineName: String,
    installed: Boolean,
    sizeMb: Int,
    missingFiles: List<String>,
    busy: Boolean,
    job: JobState,
    onDownload: () -> Unit,
    onOpenSettings: () -> Unit
) {
    val running = job as? JobState.Running
    val downloading = running != null &&
        (running.label.startsWith("Downloading voice") || running.label.startsWith("Verifying"))

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        color = if (installed) MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f)
        else MaterialTheme.colorScheme.error.copy(alpha = 0.14f)
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                if (installed) "$engineName ready ✓" else "$engineName not downloaded",
                style = MaterialTheme.typography.titleMedium,
                color = if (installed) MaterialTheme.colorScheme.secondary
                else MaterialTheme.colorScheme.error
            )
            Spacer(Modifier.height(4.dp))
            Text(
                if (installed) {
                    "Selected in Settings → Voice engine."
                } else {
                    // Never quietly fall back to the other pack: the user chose
                    // this one and the two do not sound the same.
                    "This is the engine selected in Settings. Download it, or " +
                        "switch to the other pack there — the app will not " +
                        "substitute one for the other."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            if (!installed && missingFiles.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "Missing: " + missingFiles.take(4).joinToString(", ") +
                        if (missingFiles.size > 4) " +${missingFiles.size - 4} more" else "",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (!installed) {
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = onDownload,
                        enabled = !busy,
                        modifier = Modifier
                            .weight(1f)
                            .height(44.dp),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.CloudDownload, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("Download ($sizeMb MB)")
                    }
                    TextButton(onClick = onOpenSettings, enabled = !busy) {
                        Text("Change")
                    }
                }
            }

            if (downloading && running != null) {
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        running.label +
                            if (running.detail.isNotBlank()) " · ${running.detail}" else "",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { running.progress.coerceAtLeast(0f) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                )
            }

            if (job is JobState.Failed) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "✗ ${job.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}

/** Per-speaker reference audio — the thing clone quality actually depends on. */
@Composable
fun SpeakerVoiceSection(speakers: List<SpeakerPlan>) {
    if (speakers.isEmpty()) return
    SectionCard(
        number = "1",
        title = "Voices to clone",
        subtitle = "${speakers.size} speakers · cloned from their own clips",
        done = speakers.none { it.referenceWeak }
    ) {
        Text(
            "Each speaker is cloned zero-shot from their longest clips. More " +
                "reference audio means a closer match.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(10.dp))

        speakers.forEach { s ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(vertical = 5.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    Modifier
                        .size(12.dp)
                        .clip(CircleShape)
                        .background(speakerColor(s.colorIndex))
                )
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        s.displayName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        "${s.lineCount} lines · %.1f s reference".format(s.referenceSeconds),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (s.referenceWeak) {
                    Icon(
                        Icons.Default.Warning,
                        contentDescription = "short reference",
                        tint = MaterialTheme.colorScheme.tertiary,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }

        if (speakers.any { it.referenceWeak }) {
            Spacer(Modifier.height(8.dp))
            Text(
                "⚠ Some speakers have under 3 seconds of reference. Cloning will " +
                    "be rough for those — merging their lines in Step 2, or " +
                    "checking the speaker split, usually helps.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary
            )
        }
    }
}

/** The generate button and what it is waiting on. */
@Composable
fun GenerateVoiceSection(
    total: Int,
    untranslated: Int,
    spokenCount: Int,
    estimateMinutes: Int,
    engineInstalled: Boolean,
    ready: Boolean,
    busy: Boolean,
    job: JobState,
    onSpeak: () -> Unit,
    onCancel: () -> Unit
) {
    SectionCard(
        number = "2",
        title = "Generate speech",
        subtitle = if (spokenCount > 0) "$spokenCount / $total clips spoken"
        else "$total lines to speak",
        done = total > 0 && spokenCount >= total
    ) {
        Text(
            "Each line is spoken in its speaker's cloned voice, with the Step 3 " +
                "emotion setting how strongly it is delivered. Output goes to " +
                "hi_clips/.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        // Say what is blocking, one reason at a time.
        val blocker = when {
            total == 0 -> "No lines yet — finish the earlier steps first."
            untranslated > 0 -> "$untranslated lines still have no Hindi text (Step 4)."
            !engineInstalled -> "The selected voice engine is not downloaded."
            else -> null
        }
        if (blocker != null) {
            Spacer(Modifier.height(8.dp))
            Text(
                blocker,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary
            )
        }

        Spacer(Modifier.height(12.dp))
        Button(
            onClick = onSpeak,
            enabled = ready,
            modifier = Modifier
                .fillMaxWidth()
                .height(46.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.RecordVoiceOver, null, Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
            Text(if (total > 0) "Speak $total lines" else "Speak")
        }

        if (ready) {
            Spacer(Modifier.height(6.dp))
            Text(
                "Roughly $estimateMinutes minutes on a phone CPU — about a minute " +
                    "per line. Keep the app open.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        when (job) {
            is JobState.Running -> if (job.label.startsWith("Speaking")) {
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
            else -> Unit
        }
    }
}

/** Final step: fit timing, mux, preview. */
@Composable
fun FinalVideoSection(
    allSpoken: Boolean,
    canMux: Boolean,
    keepBackground: Boolean,
    dubbedVideoPath: String?,
    busy: Boolean,
    job: JobState,
    onKeepBackground: (Boolean) -> Unit,
    onBuild: () -> Unit
) {
    SectionCard(
        number = "3",
        title = "Dubbed video",
        subtitle = dubbedVideoPath?.let { "dubbed_video.mp4 ready" }
            ?: "Fit timing and write the final file",
        done = dubbedVideoPath != null
    ) {
        Text(
            "Each clip is fitted to its original slot — sped up to 2× if the " +
                "Hindi runs long, without changing pitch. The video track is " +
                "copied, not re-encoded, so nothing is lost and it takes seconds.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(
                checked = keepBackground,
                onCheckedChange = onKeepBackground,
                enabled = !busy
            )
            Spacer(Modifier.width(10.dp))
            Column {
                Text("Keep background audio", style = MaterialTheme.typography.bodyMedium)
                Text(
                    "Music and effects survive — but the original dialogue stays " +
                        "faintly audible too, since there is no separation step yet.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        if (!allSpoken) {
            Spacer(Modifier.height(8.dp))
            Text(
                "Speak all the lines first.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary
            )
        }

        Spacer(Modifier.height(12.dp))
        Button(
            onClick = onBuild,
            enabled = canMux,
            modifier = Modifier
                .fillMaxWidth()
                .height(46.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.Movie, null, Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
            Text(if (dubbedVideoPath != null) "Rebuild video" else "Build dubbed video")
        }

        dubbedVideoPath?.let {
            Spacer(Modifier.height(10.dp))
            Surface(
                shape = RoundedCornerShape(10.dp),
                color = MaterialTheme.colorScheme.secondary.copy(alpha = 0.12f),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    "✓ $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.padding(10.dp)
                )
            }
        }

        when (job) {
            is JobState.Running -> if (
                job.label.startsWith("Building") || job.label.startsWith("Writing") ||
                job.label.startsWith("Fitting") || job.label.startsWith("Loading") ||
                job.label.startsWith("Mixing")
            ) {
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(job.label, style = MaterialTheme.typography.bodySmall)
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
                Spacer(Modifier.height(8.dp))
                Text(
                    "✗ ${job.error}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
            else -> Unit
        }
    }
}
