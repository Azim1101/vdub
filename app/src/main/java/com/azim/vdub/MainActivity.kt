package com.azim.vdub

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.azim.vdub.ui.NextStepButton
import com.azim.vdub.ui.SectionCard
import com.azim.vdub.ui.StepBadge
import com.azim.vdub.ui.SubtitlesSection
import com.azim.vdub.ui.TranslationSection
import com.azim.vdub.ui.TrimSection
import com.azim.vdub.ui.VideoInfoCard
import com.azim.vdub.ui.VideoPlayer
import com.azim.vdub.ui.VideoUploadSection
import com.azim.vdub.ui.Step1ViewModel
import com.azim.vdub.ui.theme.VdubTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            VdubTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Step1Screen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Step1Screen(vm: Step1ViewModel = hiltViewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val clipboard = LocalClipboardManager.current

    // ---- permissions ----
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* result surfaced through file access failures */ }

    LaunchedEffect(Unit) {
        val perms = buildList {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.READ_MEDIA_VIDEO)
                add(Manifest.permission.READ_MEDIA_AUDIO)
                add(Manifest.permission.POST_NOTIFICATIONS)
            } else {
                add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }
        permissionLauncher.launch(perms.toTypedArray())
    }

    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            vm.dismissMessage()
        }
    }

    // ---- pickers ----
    val videoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importVideo) }

    val drivePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importVideo) }

    val srtPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importSrt) }

    val translatedSrtPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importTranslatedSrt) }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🎬 vdub")
                        Spacer(Modifier.width(10.dp))
                        StepBadge(
                            text = if (state.step1Done) "Step 1 ✓" else "Step 1",
                            done = state.step1Done
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { vm.loadProject() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Reload project")
                    }
                    IconButton(onClick = { openAiFolder(context) }) {
                        Icon(Icons.Default.FolderOpen, contentDescription = "Storage access")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                    actionIconContentColor = MaterialTheme.colorScheme.onBackground
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
            // ---------- player ----------
            item {
                VideoPlayer(videoPath = state.videoPath, heightDp = 220)
            }
            item {
                VideoInfoCard(
                    durationMinutes = state.durationMinutes,
                    cueCount = state.cueCount,
                    lineCount = state.lineCount,
                    clipCount = state.clipCount,
                    sizeLabel = humanBytes(state.videoSizeBytes)
                )
            }

            // ---------- project name (resume-safe) ----------
            item {
                SectionCard(
                    number = "P",
                    title = "Project",
                    subtitle = "/AI/vdub_projects/${state.projectName}",
                    done = state.step1Done
                ) {
                    OutlinedTextField(
                        value = state.projectName,
                        onValueChange = vm::setProjectName,
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Project name") },
                        singleLine = true,
                        shape = RoundedCornerShape(12.dp),
                        enabled = !state.busy
                    )
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(
                            onClick = { vm.loadProject() },
                            enabled = !state.busy,
                            modifier = Modifier
                                .weight(1f)
                                .height(44.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) { Text("Open / Resume") }
                    }
                    if (state.knownProjects.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "On device: " + state.knownProjects.joinToString(", "),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // ---------- 1. video ----------
            item {
                VideoUploadSection(
                    hasVideo = state.hasVideo,
                    url = state.sourceUrl,
                    serverBase = state.serverBase.ifBlank { BuildConfig.DOWNLOAD_SERVER },
                    serverOnline = state.serverOnline,
                    busy = state.busy,
                    onPickGallery = { videoPicker.launch(arrayOf("video/*")) },
                    onPickDrive = {
                        drivePicker.launch(arrayOf("video/*", "application/octet-stream"))
                    },
                    onPasteUrl = {
                        clipboard.getText()?.text
                            ?.trim()
                            ?.takeIf { it.startsWith("http") }
                            ?.let(vm::setSourceUrl)
                    },
                    onUrlChange = vm::setSourceUrl,
                    onServerChange = vm::setServerBase,
                    onPingServer = vm::pingServer,
                    onDownload = vm::downloadVideo
                )
            }

            // ---------- 2. subtitles ----------
            item {
                SubtitlesSection(
                    cueCount = state.cueCount,
                    lineCount = state.lineCount,
                    mergeGapMs = state.mergeGapMs,
                    busy = state.busy,
                    onUploadSrt = {
                        srtPicker.launch(
                            arrayOf("application/x-subrip", "text/plain", "*/*")
                        )
                    },
                    onAutoAsr = vm::autoAsrPlaceholder,
                    onMergeGapChange = vm::setMergeGap
                )
            }

            // ---------- 3. translation ----------
            item {
                TranslationSection(
                    lineCount = state.lineCount,
                    translatedCount = state.translatedCount,
                    busy = state.busy,
                    onManualUpload = {
                        translatedSrtPicker.launch(
                            arrayOf("application/x-subrip", "text/plain", "*/*")
                        )
                    },
                    onAutoNllb = vm::autoTranslatePlaceholder,
                    onDownloadForManual = vm::exportScriptForTranslation
                )
            }

            // ---------- trim ----------
            item {
                TrimSection(
                    lineCount = state.lineCount,
                    clipCount = state.clipCount,
                    clipsSizeLabel = humanBytes(state.clipsSizeBytes),
                    canTrim = state.canTrim,
                    busy = state.busy,
                    job = state.job,
                    onTrim = vm::trim,
                    onCancel = vm::cancel
                )
            }

            // ---------- next ----------
            item {
                NextStepButton(enabled = state.step1Done && !state.busy) { /* Step 2 */ }
            }
            item { Spacer(Modifier.height(20.dp)) }
        }
    }
}

private fun humanBytes(bytes: Long): String = when {
    bytes <= 0 -> "—"
    bytes < 1024 * 1024 -> "%.0f KB".format(bytes / 1024.0)
    bytes < 1024L * 1024 * 1024 -> "%.1f MB".format(bytes / 1024.0 / 1024.0)
    else -> "%.2f GB".format(bytes / 1024.0 / 1024.0 / 1024.0)
}

/** All-files access is needed to read/write /storage/emulated/0/AI directly. */
private fun openAiFolder(context: android.content.Context) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R &&
        !Environment.isExternalStorageManager()
    ) {
        runCatching {
            context.startActivity(
                Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    "package:${context.packageName}".toUri()
                ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }.onFailure {
            context.startActivity(
                Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }
    }
}
