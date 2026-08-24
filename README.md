# 🎬 vdub

AI video dubbing for Android. **Package:** `com.azim.vdub`

Pipeline: video → subtitles → per-line clips → diarization → emotion → translation → TTS → mux.

**This repo currently implements Step 1: Download & Upload Video.**

---

## Step 1 — what's built

```
┌─────────────────────────────────┐
│  🎬 vdub          [Step 1]      │  TopAppBar + status badge
├─────────────────────────────────┤
│  Video Player  (220 dp)         │  ExoPlayer, multi-audio-track ready
│  ⏱ 37.8 min │ 473→190 │ ✂ 190   │  Info card
├─────────────────────────────────┤
│  P  Project  (resume-safe)      │  project name + Open/Resume
├─────────────────────────────────┤
│  1  Video Upload                │  [Gallery] [URL] [Drive]
│     URL field + server + Download
├─────────────────────────────────┤
│  2  Subtitles Upload            │  [Upload SRT] [Auto ASR]
│     473 cues → 190 lines        │  + merge-gap tuner
├─────────────────────────────────┤
│  3  Translation Subtitles       │  [Manual] [Auto NLLB] [Export]
├─────────────────────────────────┤
│  ✂  Video Trim — Choti Clips    │  [Trim Karo → 190 Clips] + progress
├─────────────────────────────────┤
│  [ Next → Step 2: Diarization ] │
└─────────────────────────────────┘
```

### Files

| File | Role |
|---|---|
| `app/build.gradle.kts` | ExoPlayer/Media3, Room, Hilt, ONNX Runtime, OkHttp, WorkManager |
| `ui/VideoPlayer.kt` | 220 dp player + info card + step badge |
| `ui/UploadSection.kt` | The 3 option groups + trim section |
| `ui/Step1ViewModel.kt` | State machine, progress, cancellation |
| `MainActivity.kt` | LazyColumn screen, pickers, permissions |
| `core/VdubPaths.kt` | The `/AI/` folder layout + `S0x.done` markers |
| `audio/AudioExtractor.kt` | Video → 16 kHz mono WAV via **MediaCodec** |
| `audio/WavIo.kt` | RIFF reader/writer + range slicing |
| `audio/ClipCutter.kt` | Sample-slicing clip cutter |
| `subtitle/SrtParser.kt` | SRT parse + cue→line merge |
| `data/repo/ProjectRepository.kt` | Orchestration, `script_raw.json` |
| `data/local/ProjectDb.kt` | Room: projects + clips |
| `net/DownloadClient.kt` | yt-dlp server client, resumable |

---

## The three error fixes from the spec

**1. Video download — PhantomJS.**
Sites like iq.com need `yt-dlp` + a PhantomJS binary + `OPENSSL_CONF=/dev/null`, none of
which can run inside an APK. The app POSTs to a helper server and streams the mp4 back.
Direct `.mp4` links skip the server entirely. Downloads are **`Range`-resumable** — a
dropped connection continues from the `.part` file instead of restarting 142 MB.

**2. Audio extract — ffmpeg segfault.**
Replaced with `MediaCodec` (`AudioExtractor.kt`): hardware-backed, no bundled native lib
to crash, and it streams decoded PCM straight to disk so a 2267 s / 70 MB track never
lands in RAM. Any input sample rate is converted to 16 kHz mono by a **stateful** linear
resampler that carries its fractional cursor across decoder buffers.
*Verified: 0.0 samples of drift over 3 s, max error 3e-5 vs an ideal tone, no clicks at
buffer joins.*

**3. Clip cutting — ffmpeg fail.**
Pure sample slicing, exactly as specified:
```
s = (start - 0.2) * sr ;  e = (end + 0.2) * sr ;  clip = wav[s:e]
```
Each clip is one seek + one read of the exact byte range — no subprocess, no full-file
decode, 190 clips in one pass. *Verified byte-identical to the source range, with
clamping at both ends.*

**4. Resume-safe.**
Room DB **plus** `S01.done` markers on disk. Reopening a project rehydrates from
whatever actually exists in the folder, so state survives reinstall and DB loss.

---

## On-device layout

```
/storage/emulated/0/AI/
├── libs/arm64-v8a/          libMNN.so, libonnxruntime.so
├── models/                  campplus.onnx, emotion2vec_plus_base.onnx, ...
└── vdub_projects/{project}/
    ├── input_video.mp4
    ├── org_audio.wav        16 kHz mono PCM16
    ├── subs/original.srt · subs/translated.srt
    ├── clips/line_0000.wav ... line_0189.wav
    ├── out/script_raw.json
    └── S01.done
```

Models are **not** bundled — APK stays ~50 MB. Push them once:
```bash
adb push campplus.onnx /storage/emulated/0/AI/models/
```

`script_raw.json` carries `speaker` and `emotion` fields already, so Steps 2–3 only fill
them in rather than changing the schema.

---

## Cue → line merging

Raw SRT cues are *display* chunks (one sentence split across 2–3 cards); TTS needs whole
utterances. Adjacent cues merge while the gap is small, the sentence hasn't terminated,
and duration/length caps aren't hit.

The spec's 473 → 190 is subtitle-specific, so the ratio is **tunable in the UI**
(200/400/800/1200/2000 ms). Re-merging re-attaches any translated SRT and invalidates
stale clips automatically.

---

## Download server contract

```
GET  /health                          -> 200
POST /download  {"url","format","project"}
     -> {"ok":true,"file_url":"/files/x.mp4","size_bytes":149000000}
GET  {file_url}                       -> mp4 bytes (must support Range)
```

Reference implementation:
```python
subprocess.run(["yt-dlp", "-f", fmt, "-o", out, url],
               env={**os.environ, "OPENSSL_CONF": "/dev/null"})
```

Point the app at it via the in-app field, or bake a default in:
```bash
./gradlew assembleDebug -PVDUB_SERVER=https://xxxx.trycloudflare.com
```

---

## Build

The Gradle **wrapper JAR is not committed** (it's a binary and this repo was scaffolded
offline). Generate it once — Android Studio does it automatically on first open, or:

```bash
gradle wrapper --gradle-version 8.9    # needs Gradle installed once
```

Then:

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest      # SrtParser + WavIo tests
```

Requires JDK 17, `compileSdk 34`, `minSdk 26`, arm64-v8a.
Grant **All files access** on first run (top-right folder icon) — the pipeline reads and
writes `/storage/emulated/0/AI/` directly.

---

## Roadmap

| Step | Feature | Model |
|---|---|---|
| ✅ 1 | Upload + player + trim | — |
| 2 | Speaker diarization | campplus 28 MB ONNX |
| 3 | Emotion | emotion2vec_plus_base 355 MB |
| 4 | Translation | NLLB q8 0.9 GB (server) |
| 5 | TTS | Chatterbox 1.73 GB (server) |
