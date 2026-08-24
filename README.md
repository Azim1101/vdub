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
│  ✂  Audio Trim — Choti Clips    │  [Trim Karo → 190 Clips] + progress
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

### The video is never cut

Only `org_audio.wav` is sliced. `input_video.mp4` stays whole, on purpose:

- diarization, emotion and TTS all read **audio only** — no model looks at frames
- the final mux needs the **full-length video** to attach the dubbed track to
- cutting a 142 MB mp4 190 times means re-encoding: minutes of work, GBs of disk,
  and generation loss — for output nothing downstream consumes

Audio slicing does the same job in seconds.

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

## URL download — what works on-device

There is no helper server. `VideoResolver` runs in the app:

| Input | Result |
|---|---|
| Direct `.mp4` / `.webm` / `.m3u8` link | ✅ downloads |
| Page with `<video src>`, `og:video`, JSON `contentUrl` | ✅ finds and downloads |
| iQIYI, YouTube, Netflix, Hotstar | ❌ named and refused |

The refusal is deliberate. yt-dlp is ~200k lines of Python with per-site
extractors, and iQIYI additionally signs its streams in JavaScript — that cannot
run inside an APK. For those, download on a PC and pick the file from **Gallery**.

## Storage

Writing to `/storage/emulated/0/AI` needs **All-files access**, which Android
only grants from Settings. Until then the app uses its own external dir
(`/Android/data/com.azim.vdub/files/AI/`) — no permission required, everything
works, but `adb push` and other apps cannot see it. A banner offers the upgrade,
and models are looked up in **both** roots.

---

## Download the APK

**[⬇ vdub-step1-debug.apk](https://github.com/Azim1101/vdub/releases/download/step1-latest/vdub-step1-debug.apk)**  (~28 MB)

Every push to this branch rebuilds it and replaces the `step1-latest` release.
On first run, grant **All files access** via the folder icon — the pipeline reads and
writes `/storage/emulated/0/AI/` directly.

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

## Step 2 — Speaker Diarization

Tags every clip with a speaker, writes `out/script_speakers.json` + `S02.done`.

**Getting the model:** tap **Download model (28 MB)** in the app — no PC or adb
needed. It pulls the 3D-Speaker CAM++ ONNX export, trying four mirrors
(huggingface.co and hf-mirror.com), resumes if the connection drops, and
validates the file before installing it.

Manual placement still works if you already have it:
```bash
adb push campplus.onnx /storage/emulated/0/AI/models/
```
It must be a real **ONNX export** — a FunASR `.bin`/`.pt` checkpoint will not
load. The app checks the size and protobuf header up front, so a half-downloaded
file or an HTML error page is caught immediately instead of failing later inside
ONNX Runtime.

### Two corrections to the original plan

**1. campplus does not take a waveform.** Its graph starts *after* feature
extraction, so the input is `(batch, frames, 80)` kaldi fbank with CMN applied.
`Fbank.kt` implements it: 25 ms window, 10 ms shift, Povey window, HTK mel,
radix-2 FFT. Verified against a naive DFT (max error 4e-12), kaldi frame counts
(1 s → 98 frames), and tone-to-bin placement.

**2. The threshold direction was inverted.** Raising 0.55 → 0.75 gives *more*
speakers, not fewer. The pairwise `if sim > thr: same speaker` merge is
**single-linkage**, which chains — one ambiguous clip bridges two speakers, and
lowering the threshold to stop that shatters them into singletons. That is how
3 speakers became 29.

This uses **average linkage**, comparing cluster means, so the threshold is the
*stopping* similarity:

| Threshold | Effect |
|---|---|
| Lower (0.20–0.45) | merges more → **fewer** speakers |
| Higher (0.70+) | stops sooner → **more** speakers |

On simulated 190-clip data with campplus-like statistics, 0.20–0.60 all recover
exactly 120/50/20 at 100% purity; 0.70 fragments into 28 — close to the 29
originally observed. A unit test pins this direction.

**Since you usually know the count, use "I know the count" mode** — clustering to
exactly *k* is more robust than any threshold.

Embeddings cache to `speaker_embeds.bin`, so re-tuning is instant rather than
190 fresh inferences.

## Everything runs on the phone — no server

The "models add up to 4 GB, a 6 GB phone cannot do it" reasoning sums the
models. But the pipeline is **sequential**: diarization finishes before emotion
starts, which finishes before translation starts. Each stage closes its ONNX
session before the next opens, so only one model is resident at a time.

| | |
|---|---|
| Disk, if you download everything | ~1.6 GB |
| **Peak RAM** | **~600 MB** (largest single model) |

That is why nothing needs to be offloaded. A unit test pins the invariant.

## Settings → Models

One screen to manage every model, grouped by pipeline step. Download, resume,
cancel, re-download, remove — with size, licence and purpose shown, plus disk
used and free.

| Model | Size | Step | Required |
|---|---|---|---|
| CAM++ speaker embedding | 28 MB | Speakers | yes |
| SenseVoice ASR | 249 MB | Transcribe | no — only if you have no SRT |
| emotion2vec+ base | 373 MB | Emotion | yes |
| NLLB-200 distilled 600M | 620 MB | Translate | yes · CC-BY-NC |
| Kokoro-82M TTS (Hindi voices) | 92 MB | Voice | yes |

Every file has two mirrors (huggingface.co and hf-mirror.com), transfers resume,
and downloads are validated before install.

## Roadmap

| Step | Feature | Status |
|---|---|---|
| 1 | Upload + player + trim | ✅ done |
| 2 | Speaker diarization | ✅ done |
| 3 | Emotion | ✅ done |
| 4 | Translation (NLLB) | next |
| 5 | TTS + mux (Kokoro) | next |

## Step 3 — Emotion

Tags each clip with one of nine emotions, writes `out/script_emotion.json` +
`S03.done`. Every line can be corrected by tapping it.

**The classifier head is a separate file.** emotion2vec's ONNX graph outputs
frame *features*, not probabilities — `emotion2vec_head.json` carries the
`weight`/`bias`/`labels`. The app mean-pools over time, then applies
`W @ pooled + B` and softmax. Loading only the graph gives features that look
fine and classify nothing, so the head is required and its shapes are checked
against the model output.

**The two audio models want opposite input scales:**

| Model | Input |
|---|---|
| campplus | kaldi fbank on the ±32768 integer scale |
| emotion2vec | raw waveform normalised to [-1, 1] |

Feeding either one the other's scale produces confident nonsense rather than an
error, so each reader does its own conversion.

Emotion sets delivery strength for the voice stage: angry 1.4×, happy 1.1×,
sad 0.9×.
