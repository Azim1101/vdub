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
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
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
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.azim.vdub.ui.ClipsPreviewSection
import com.azim.vdub.ui.EmotionLinesSection
import com.azim.vdub.ui.EmotionModelCard
import com.azim.vdub.ui.EmotionSection
import com.azim.vdub.ui.AutoTranslateSection
import com.azim.vdub.ui.NextStep4Button
import com.azim.vdub.ui.NextStep5Button
import com.azim.vdub.ui.Step4ViewModel
import com.azim.vdub.ui.TranslationLinesSection
import com.azim.vdub.ui.TranslationUploadSection
import com.azim.vdub.ui.Step3ViewModel
import com.azim.vdub.ui.ModelStatusCard
import com.azim.vdub.ui.NextStep3Button
import com.azim.vdub.ui.NextStepButton
import com.azim.vdub.ui.SettingsScreen
import com.azim.vdub.ui.SpeakerSection
import com.azim.vdub.ui.StorageWarningCard
import com.azim.vdub.ui.Step2ViewModel
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
                    VdubRoot()
                }
            }
        }
    }
}

/** Two-screen shell: Step 1 -> Step 2, with the project name carried across. */
@Composable
fun VdubRoot() {
    var screen by rememberSaveable { mutableStateOf(1) }
    var project by rememberSaveable { mutableStateOf("vdub_step") }

    val context = LocalContext.current
    when (screen) {
        1 -> Step1Screen(
            onProjectChanged = { project = it },
            onOpenSettings = { screen = 9 },
            onNext = { name ->
                project = name
                screen = 2
            }
        )
        2 -> Step2Screen(
            project = project,
            onBack = { screen = 1 },
            onOpenSettings = { screen = 9 },
            onNext = { screen = 3 }
        )
        3 -> Step3Screen(
            project = project,
            onBack = { screen = 2 },
            onOpenSettings = { screen = 9 },
            onNext = { screen = 4 }
        )
        4 -> Step4Screen(
            project = project,
            onBack = { screen = 3 },
            onOpenSettings = { screen = 9 }
        )
        else -> SettingsScreen(
            onBack = { screen = 1 },
            onGrantStorage = { openAiFolder(context) }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Step3Screen(
    project: String,
    onBack: () -> Unit,
    onOpenSettings: () -> Unit,
    onNext: () -> Unit,
    vm: Step3ViewModel = hiltViewModel()
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(project) { vm.load(project) }
    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            vm.dismissMessage()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🎬 vdub")
                        Spacer(Modifier.width(10.dp))
                        StepBadge(
                            text = if (state.step3Done) "Step 3 ✓" else "Step 3",
                            done = state.step3Done
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to Step 2")
                    }
                },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                    navigationIconContentColor = MaterialTheme.colorScheme.onBackground,
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
            item {
                EmotionModelCard(
                    present = state.modelPresent,
                    path = state.modelPath,
                    sizeBytes = state.modelSizeBytes,
                    job = state.job,
                    busy = state.busy,
                    onDownload = vm::downloadModel
                )
            }
            item {
                EmotionSection(
                    clipCount = state.clipCount,
                    counts = state.counts,
                    modelPresent = state.modelPresent,
                    busy = state.busy,
                    job = state.job,
                    onDetect = vm::detect,
                    onCancel = vm::cancel
                )
            }
            item {
                EmotionLinesSection(
                    lines = state.lines,
                    speakerName = state::speakerName,
                    onSetEmotion = vm::setEmotion
                )
            }
            item {
                NextStep4Button(
                    enabled = state.step3Done && !state.busy,
                    onClick = onNext
                )
            }
            item { Spacer(Modifier.height(20.dp)) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Step2Screen(
    project: String,
    onBack: () -> Unit,
    onOpenSettings: () -> Unit,
    onNext: () -> Unit,
    vm: Step2ViewModel = hiltViewModel()
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(project) { vm.load(project) }
    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            vm.dismissMessage()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🎬 vdub")
                        Spacer(Modifier.width(10.dp))
                        StepBadge(
                            text = if (state.step2Done) "Step 2 ✓" else "Step 2",
                            done = state.step2Done
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to Step 1")
                    }
                },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                    navigationIconContentColor = MaterialTheme.colorScheme.onBackground,
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
            item {
                ModelStatusCard(
                    present = state.modelPresent,
                    path = state.modelPath,
                    sizeBytes = state.modelSizeBytes,
                    busy = state.busy,
                    job = state.job,
                    onDownload = vm::downloadModel
                )
            }
            item {
                SpeakerSection(
                    speakers = state.speakers,
                    clipCount = state.clipCount,
                    embedCount = state.embedCount,
                    threshold = state.threshold,
                    useTargetK = state.useTargetK,
                    targetK = state.targetK,
                    modelPresent = state.modelPresent,
                    hasEmbeds = state.hasEmbeds,
                    busy = state.busy,
                    job = state.job,
                    onThresholdChange = vm::setThreshold,
                    onUseTargetK = vm::setUseTargetK,
                    onTargetKChange = vm::setTargetK,
                    onExtract = vm::extractAndCluster,
                    onRecluster = vm::recluster,
                    onCancel = vm::cancel,
                    onRename = vm::renameSpeaker
                )
            }
            item {
                ClipsPreviewSection(
                    lines = state.lines,
                    speakers = state.speakers,
                    nameFor = state::nameFor
                )
            }
            item {
                NextStep3Button(enabled = state.step2Done && !state.busy, onClick = onNext)
            }
            item { Spacer(Modifier.height(20.dp)) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Step4Screen(
    project: String,
    onBack: () -> Unit,
    onOpenSettings: () -> Unit,
    vm: Step4ViewModel = hiltViewModel()
) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    LaunchedEffect(project) { vm.load(project) }
    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            vm.dismissMessage()
        }
    }

    val srtPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importSrt) }

    val jsonPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? -> uri?.let(vm::importJson) }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🎬 vdub")
                        Spacer(Modifier.width(10.dp))
                        StepBadge(
                            text = if (state.step4Done) "Step 4 ✓" else "Step 4",
                            done = state.step4Done
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to Step 3")
                    }
                },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground,
                    navigationIconContentColor = MaterialTheme.colorScheme.onBackground,
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
            item {
                TranslationUploadSection(
                    source = state.source,
                    translatedCount = state.translatedCount,
                    total = state.total,
                    missing = state.missing,
                    progress = state.progress,
                    busy = state.busy,
                    job = state.job,
                    exportedPath = state.exportedPath,
                    onUploadSrt = {
                        srtPicker.launch(
                            arrayOf("application/x-subrip", "text/plain", "*/*")
                        )
                    },
                    onUploadJson = {
                        jsonPicker.launch(arrayOf("application/json", "text/plain", "*/*"))
                    },
                    onExportSrt = vm::exportSrt,
                    onExportJson = vm::exportJson
                )
            }
            item {
                AutoTranslateSection(
                    nllbInstalled = state.nllbInstalled,
                    busy = state.busy,
                    job = state.job,
                    onDownload = vm::downloadNllb,
                    onTranslate = vm::autoTranslateNotReady
                )
            }
            item {
                TranslationLinesSection(
                    lines = state.lines,
                    speakerName = state::speakerName,
                    onSetLine = vm::setLine
                )
            }
            item { NextStep5Button(enabled = state.complete && !state.busy) { } }
            item { Spacer(Modifier.height(20.dp)) }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Step1Screen(
    onProjectChanged: (String) -> Unit = {},
    onOpenSettings: () -> Unit = {},
    onNext: (String) -> Unit = {},vm: Step1ViewModel = hiltViewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val clipboard = LocalClipboardManager.current
    var confirmReset by remember { mutableStateOf(false) }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Clear \"${state.projectName}\"?") },
            text = {
                Text(
                    "Deletes the video, subtitles, clips and all step results " +
                        "for this project. Downloaded models are not touched."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmReset = false
                    vm.resetProject()
                }) { Text("Clear") }
            },
            dismissButton = {
                TextButton(onClick = { confirmReset = false }) { Text("Cancel") }
            }
        )
    }

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
    LaunchedEffect(state.projectName) { onProjectChanged(state.projectName) }

    // Re-check All-files access when the user comes back from Settings.
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val obs = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) vm.loadProject()
        }
        lifecycleOwner.lifecycle.addObserver(obs)
        onDispose { lifecycleOwner.lifecycle.removeObserver(obs) }
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
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings · Models")
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
            item {
                StorageWarningCard(
                    shared = state.storageShared,
                    path = state.storagePath,
                    onGrant = { openAiFolder(context) }
                )
            }
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
                        OutlinedButton(
                            onClick = { confirmReset = true },
                            enabled = !state.busy,
                            modifier = Modifier.height(44.dp),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            )
                        ) { Text("Clear all") }
                    }
                    if (!state.hasVideo && !state.hasSubtitles && state.clipCount == 0) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "Nothing loaded. Type a name and press Open / Resume, " +
                                "or just pick a video below.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
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
                    videoLabel = "input_video.mp4 · ${humanBytes(state.videoSizeBytes)}" +
                        if (state.durationMs > 0) " · %.1f min".format(state.durationMinutes)
                        else "",
                    url = state.sourceUrl,
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
                    onDownload = vm::downloadVideo,
                    onClearVideo = vm::clearVideo
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
                NextStepButton(enabled = state.step1Done && !state.busy) {
                    onNext(state.projectName)
                }
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
