package com.azim.vdub.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.foundation.clickable
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.azim.vdub.core.ModelCatalog

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onGrantStorage: () -> Unit,
    vm: SettingsViewModel = hiltViewModel()
) {
    val state by vm.state.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { vm.refresh() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings · Models") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                    navigationIconContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text(
                            "Everything runs on this phone",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "No server, no account. Stages run one at a time, so " +
                                "only one model is in memory at once — about " +
                                "${mb(ModelCatalog.peakAnalysisRamBytes)} for the " +
                                "analysis steps, not the " +
                                "${mb(ModelCatalog.totalBytes)} total on disk. " +
                                "Voice cloning is the heavy one " +
                                "(~${mb(ModelCatalog.VOICE_ENGINES.minOf { e -> e.ramBytes })}" +
                                "–${mb(ModelCatalog.VOICE_ENGINES.maxOf { e -> e.ramBytes })}).",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "${state.installedCount} of ${state.rows.size} installed · " +
                                "${mb(state.totalInstalledBytes)} used · " +
                                "${mb(state.freeBytes)} free",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.secondary
                        )
                        Spacer(Modifier.height(2.dp))
                        Text(
                            state.modelsPath,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            if (!state.storageShared) {
                item {
                    StorageWarningCard(
                        shared = false,
                        path = state.modelsPath,
                        onGrant = onGrantStorage
                    )
                }
            }

            ModelCatalog.Stage.entries.forEach { stage ->
                val rows = state.rows.filter { it.model.stage == stage }
                if (rows.isEmpty()) return@forEach
                item {
                    Text(
                        "Step ${stage.step} · ${stage.label}",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
                if (stage == ModelCatalog.Stage.TTS) {
                    item {
                        VoiceEnginePicker(
                            selectedId = state.voiceEngineId,
                            rows = state.rows,
                            onSelect = vm::selectVoiceEngine
                        )
                    }
                }
                rows.forEach { row ->
                    item(key = row.model.id) {
                        ModelCard(
                            row = row,
                            anyBusy = state.busyId != null,
                            onDownload = { vm.download(row.model) },
                            onDelete = { vm.delete(row.model) },
                            onCancel = vm::cancel
                        )
                    }
                }
            }

            item { Spacer(Modifier.height(24.dp)) }
        }
    }
}

/**
 * Choose which engine the voice stage uses. Only the selected one is ever
 * downloaded and loaded, so the RAM ceiling is one engine, never the sum.
 */
@Composable
private fun VoiceEnginePicker(
    selectedId: String,
    rows: List<ModelRow>,
    onSelect: (String) -> Unit
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(14.dp)) {
            Text("Voice engine", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(2.dp))
            Text(
                "Only the selected engine is downloaded and loaded.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(10.dp))

            ModelCatalog.VOICE_ENGINES.forEach { engine ->
                val selected = engine.id == selectedId
                val installed = rows.firstOrNull { it.model.id == engine.id }?.installed == true
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.16f)
                    else MaterialTheme.colorScheme.surface.copy(alpha = 0.4f),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                ) {
                    Row(
                        Modifier
                            .clickable { onSelect(engine.id) }
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(selected = selected, onClick = { onSelect(engine.id) })
                        Spacer(Modifier.width(6.dp))
                        Column(Modifier.weight(1f)) {
                            Text(
                                engine.name,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                engineBlurb(engine.id, engine.sizeMb),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (installed) {
                            Icon(
                                Icons.Default.CheckCircle,
                                contentDescription = "installed",
                                tint = MaterialTheme.colorScheme.secondary,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
            }

            // Never let the selection imply readiness it does not have.
            val selectedRow = rows.firstOrNull { it.model.id == selectedId }
            if (selectedRow != null && !selectedRow.installed) {
                Text(
                    "Selected engine is not downloaded yet — get it below before " +
                        "running the voice step.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.tertiary
                )
            }
        }
    }
}

private fun engineBlurb(id: String, sizeMb: Int): String = when (id) {
    "chatterbox_mix" -> "Clones · ~$sizeMb MB · LLM Q4, clone + vocoder FP32"
    "chatterbox_q4" -> "Clones · ~$sizeMb MB · everything Q4, smallest Chatterbox"
    "dhvaani" -> "Clones · ~$sizeMb MB · 13 Indic languages, faster than real time"
    // Stated first because it is the one thing a user must know before
    // choosing this engine, and it cannot be inferred from the size.
    "indri" -> "No cloning · ~$sizeMb MB · preset voices only, slow"
    else -> "~$sizeMb MB"
}

@Composable
private fun ModelCard(
    row: ModelRow,
    anyBusy: Boolean,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
    onCancel: () -> Unit
) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (row.installed) {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.secondary,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                }
                Column(Modifier.weight(1f)) {
                    Text(
                        row.model.name,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        buildString {
                            append("${row.model.sizeMb} MB")
                            if (!row.model.required) append(" · optional")
                            if (row.model.license.isNotBlank()) {
                                append(" · ").append(row.model.license)
                            }
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(Modifier.height(6.dp))
            Text(
                row.model.description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            if (row.model.note.isNotBlank()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    row.model.note,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.tertiary
                )
            }

            // Be explicit when a model can be fetched but not yet executed,
            // rather than letting it look ready and fail at Step 5.
            if (!row.model.runnable) {
                Spacer(Modifier.height(6.dp))
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.16f),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "PyTorch weights — ONNX Runtime cannot load these yet. " +
                            "Downloading stores them for the voice stage; it " +
                            "will not run until that stage ships.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.tertiary,
                        modifier = Modifier.padding(8.dp)
                    )
                }
            }

            if (row.downloading) {
                Spacer(Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        row.detail.ifBlank { "Starting…" },
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f)
                    )
                    TextButton(onClick = onCancel) { Text("Cancel") }
                }
                Spacer(Modifier.height(4.dp))
                if (row.progress >= 0f) {
                    LinearProgressIndicator(
                        progress = { row.progress },
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
            } else {
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (row.installed) {
                        OutlinedButton(
                            onClick = onDelete,
                            enabled = !anyBusy,
                            modifier = Modifier.height(42.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Icon(Icons.Default.Delete, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Remove")
                        }
                        OutlinedButton(
                            onClick = onDownload,
                            enabled = !anyBusy,
                            modifier = Modifier
                                .weight(1f)
                                .height(42.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) { Text("Re-download") }
                    } else {
                        Button(
                            onClick = onDownload,
                            enabled = !anyBusy,
                            modifier = Modifier
                                .weight(1f)
                                .height(42.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Icon(Icons.Default.CloudDownload, null, Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Download ${row.model.sizeMb} MB")
                        }
                    }
                }
            }

            row.error?.let {
                Spacer(Modifier.height(8.dp))
                Text(
                    "✗ $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}

private fun mb(b: Long): String = when {
    b <= 0 -> "—"
    b < 1024L * 1024 -> "%.0f KB".format(b / 1024.0)
    b < 1024L * 1024 * 1024 -> "%.0f MB".format(b / 1024.0 / 1024.0)
    else -> "%.1f GB".format(b / 1024.0 / 1024.0 / 1024.0)
}
