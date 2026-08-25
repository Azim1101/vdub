package com.azim.vdub.ui

import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.Button
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.azim.vdub.core.ModelCatalog
import com.azim.vdub.data.model.JobState
import com.azim.vdub.data.model.SpeakerLine
import com.azim.vdub.data.model.TranslationSource

/**
 * Step 4. Uploading a finished translation is the primary path — it skips
 * machine translation and the 591 MB model download entirely.
 */
@Composable
fun TranslationUploadSection(
    source: TranslationSource,
    translatedCount: Int,
    total: Int,
    missing: Int,
    progress: Float,
    busy: Boolean,
    job: JobState,
    exportedPath: String?,
    onUploadSrt: () -> Unit,
    onUploadJson: () -> Unit,
    onExportSrt: () -> Unit,
    onExportJson: () -> Unit
) {
    SectionCard(
        number = "1",
        title = "Already translated?",
        subtitle = if (translatedCount > 0) "$translatedCount / $total lines"
        else "Upload your own translation and skip the model",
        done = total > 0 && missing == 0 && translatedCount > 0
    ) {
        Text(
            "If you already have a translated SRT — or the JSON exported below " +
                "with the Hindi filled in — upload it here. Nothing is " +
                "downloaded and no model runs.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(12.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onUploadSrt,
                enabled = !busy,
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.UploadFile, null, Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Upload SRT", maxLines = 1)
            }
            OutlinedButton(
                onClick = onUploadJson,
                enabled = !busy,
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.UploadFile, null, Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Upload JSON", maxLines = 1)
            }
        }

        Spacer(Modifier.height(10.dp))
        Text(
            "Or export the script, translate it anywhere, then upload it back:",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = onExportSrt,
                enabled = !busy,
                modifier = Modifier
                    .weight(1f)
                    .height(42.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Download, null, Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Export SRT", maxLines = 1)
            }
            OutlinedButton(
                onClick = onExportJson,
                enabled = !busy,
                modifier = Modifier
                    .weight(1f)
                    .height(42.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Download, null, Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Export JSON", maxLines = 1)
            }
        }

        exportedPath?.let {
            Spacer(Modifier.height(8.dp))
            Text(
                "Saved: $it",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary
            )
        }

        if (total > 0 && translatedCount > 0) {
            Spacer(Modifier.height(12.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp))
            )
            Spacer(Modifier.height(6.dp))
            Text(
                buildString {
                    append("$translatedCount of $total translated")
                    if (missing > 0) append(" · $missing still empty")
                    if (source != TranslationSource.NONE) {
                        append("  ·  from ")
                        append(
                            when (source) {
                                TranslationSource.UPLOADED_SRT -> "uploaded SRT"
                                TranslationSource.UPLOADED_JSON -> "uploaded JSON"
                                TranslationSource.AUTO_NLLB -> "NLLB"
                                TranslationSource.MANUAL_EDIT -> "manual edits"
                                TranslationSource.NONE -> ""
                            }
                        )
                    }
                },
                style = MaterialTheme.typography.bodySmall,
                color = if (missing > 0) MaterialTheme.colorScheme.tertiary
                else MaterialTheme.colorScheme.secondary
            )
        }

        JobLine(job, "Reading", "Exporting")
    }
}

/** Auto translation — secondary, and honest that it is not wired yet. */
@Composable
fun AutoTranslateSection(
    nllbInstalled: Boolean,
    busy: Boolean,
    job: JobState,
    onDownload: () -> Unit,
    onTranslate: () -> Unit
) {
    SectionCard(
        number = "2",
        title = "Or translate on-device",
        subtitle = "NLLB-200 · ${ModelCatalog.NLLB.sizeMb} MB",
        done = false
    ) {
        Text(
            "Only needed if you have no translation of your own. " +
                "${ModelCatalog.NLLB.license}.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(10.dp))

        if (!nllbInstalled) {
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
                Text("Download NLLB (${ModelCatalog.NLLB.sizeMb} MB)")
            }
        } else {
            OutlinedButton(
                onClick = onTranslate,
                enabled = !busy,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.SmartToy, null, Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Auto-translate to Hindi")
            }
            Spacer(Modifier.height(6.dp))
            Text(
                "Model installed, but the translation stage is not built yet — " +
                    "uploading your own file is the working path today.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.tertiary
            )
        }

        JobLine(job, "Downloading NLLB", "Verifying NLLB")
    }
}

/** Per-line Hindi, editable. */
@Composable
fun TranslationLinesSection(
    lines: List<SpeakerLine>,
    speakerName: (String) -> String,
    onSetLine: (String, String) -> Unit,
    limit: Int = 60
) {
    if (lines.isEmpty()) return
    var editing by remember { mutableStateOf<String?>(null) }
    var draft by remember { mutableStateOf("") }

    SectionCard(
        number = "3",
        title = "Lines",
        subtitle = "Tap any line to edit its Hindi",
        done = lines.all { it.hi.isNotBlank() }
    ) {
        lines.take(limit).forEach { line ->
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
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
                        modifier = Modifier.width(60.dp)
                    )
                    Text(
                        line.text,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                }

                if (editing == line.utt) {
                    Spacer(Modifier.height(4.dp))
                    OutlinedTextField(
                        value = draft,
                        onValueChange = { draft = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Hindi") },
                        shape = RoundedCornerShape(10.dp)
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = { editing = null }) { Text("Cancel") }
                        TextButton(onClick = {
                            onSetLine(line.utt, draft.trim())
                            editing = null
                        }) { Text("Save") }
                    }
                } else {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (line.hi.isBlank())
                            MaterialTheme.colorScheme.error.copy(alpha = 0.12f)
                        else MaterialTheme.colorScheme.secondary.copy(alpha = 0.10f),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        TextButton(
                            onClick = {
                                editing = line.utt
                                draft = line.hi
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                line.hi.ifBlank { "— not translated —" },
                                style = MaterialTheme.typography.bodySmall,
                                color = if (line.hi.isBlank())
                                    MaterialTheme.colorScheme.error
                                else MaterialTheme.colorScheme.onSurface,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }
                }
            }
        }
        if (lines.size > limit) {
            Spacer(Modifier.height(6.dp))
            Text(
                "… and ${lines.size - limit} more (full list in " +
                    "out/script_translated.json)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun JobLine(job: JobState, vararg prefixes: String) {
    when (job) {
        is JobState.Running -> if (prefixes.any { job.label.startsWith(it) }) {
            Spacer(Modifier.height(10.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text(
                    job.label + if (job.detail.isNotBlank()) " · ${job.detail}" else "",
                    style = MaterialTheme.typography.bodySmall
                )
            }
            if (job.progress >= 0f) {
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { job.progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                )
            }
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

@Composable
fun NextStep5Button(enabled: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .height(52.dp),
        shape = RoundedCornerShape(14.dp)
    ) {
        Icon(Icons.Default.RecordVoiceOver, null)
        Spacer(Modifier.width(8.dp))
        Text("Next → Step 5: Voice", style = MaterialTheme.typography.titleMedium)
    }
}
