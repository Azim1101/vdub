# 🎬 vdub

AI video dubbing for Android — **everything runs on the phone. No server, no account.**

**Package:** `com.azim.vdub` · Kotlin · Jetpack Compose · ONNX Runtime

```
video → subtitles → per-line clips → speakers → emotion → translation → voice → mux
```

| Step | Feature | Status |
|---|---|---|
| 1 | Video upload, player, subtitles, clip trim | ✅ done |
| 2 | Speaker diarization (CAM++) | ✅ done |
| 3 | Emotion detection (emotion2vec+) | ✅ done |
| 4 | Translation (NLLB-200) | model wired, stage next |
| 5 | Voice + mux (Chatterbox Hindi) | model wired, stage next — see below |

---

## Download

**[⬇ vdub-step1-debug.apk](https://github.com/Azim1101/vdub/releases/download/step1-latest/vdub-step1-debug.apk)** (~29 MB)

Rebuilt on every push. Models are **not** bundled — fetch the ones you need from
**Settings → Models** inside the app.

On first run, grant **All files access** (folder icon, top right) so the pipeline
can use `/storage/emulated/0/AI/`. If you skip it the app still works — it falls
back to its own private folder and tells you so.

---

## Why no server is needed

The usual objection is that the models add up to ~4 GB, so a 6 GB phone cannot
run them. That sums the models — but the pipeline is **sequential**. Diarization
finishes before emotion starts, which finishes before translation starts, and
each stage closes its ONNX session before the next opens.

| | |
|---|---|
| Disk, if you download everything | **1.79 GB** |
| **Peak RAM** | **591 MB** for Steps 1–4 · ~1.9 GB during Chatterbox TTS |

A unit test pins that invariant so it cannot quietly regress.

---

## Settings → Models

One screen to manage every model: download, resume, cancel, re-download, remove —
with size, licence, purpose, disk used and free space.

| Model | Size | Step | Required |
|---|---|---|---|
| CAM++ speaker embedding | 26 MB | Speakers | yes |
| SenseVoice ASR | 237 MB | Transcribe | no — only if you have no SRT |
| emotion2vec+ base | 355 MB | Emotion | yes |
| NLLB-200 distilled 600M | 591 MB | Translate | yes · CC-BY-NC |
| Chatterbox Hindi INT8 (voice cloning) | 628 MB | Voice | yes · MIT |

Every file has two mirrors (huggingface.co and hf-mirror.com), transfers are
Range-resumable, and each download is validated before install — an HTML error
page saved as `.onnx`, or a truncated transfer, is caught immediately instead of
failing later inside ONNX Runtime with an unreadable error.

---

## Step 1 — Video, subtitles, clips

```
┌─────────────────────────────────┐
│  🎬 vdub   [Step 1]      ⚙ 📁   │
├─────────────────────────────────┤
│  Video Player  (220 dp)         │  ExoPlayer, multi-audio-track ready
│  ⏱ 37.8 min │ 473→190 │ ✂ 190   │
├─────────────────────────────────┤
│  P  Project (resume-safe)       │
│  1  Video Upload   [Gallery] [URL] [Drive]
│  2  Subtitles      [Upload SRT] [Auto ASR] + merge-gap tuner
│  3  Translation    [Manual] [Auto NLLB] [Export]
│  ✂  Audio Trim     [Trim Karo → 190 Clips]
│  [ Next → Step 2 ]              │
└─────────────────────────────────┘
```

### The three failures from the original notes

**1. ffmpeg segfaulted extracting audio** → replaced with **MediaCodec**.
Hardware-backed, no bundled native library to crash, and PCM streams straight to
disk so a 2267 s / 70 MB track never lands in RAM. Any input rate is converted to
16 kHz mono by a *stateful* resampler that carries its fractional cursor across
decoder buffers. *Verified: 0.0 samples of drift over 3 s, max error 3e-5 vs an
ideal tone, no clicks at buffer joins.*

**2. ffmpeg failed cutting clips** → pure sample slicing, as specified:
`clip = wav[(start-0.2)*sr : (end+0.2)*sr]`. One seek plus one read of the exact
byte range per clip — no subprocess, no full-file decode. *Verified
byte-identical to the source range, with clamping at both ends.*

**3. URL download needed PhantomJS** → see [URL download](#url-download) below.

### The video is never cut

Only `org_audio.wav` is sliced. `input_video.mp4` stays whole, on purpose:

- diarization, emotion and TTS read **audio only** — no model looks at frames
- the final mux needs the **full-length video** to attach the dubbed track to
- cutting a 142 MB mp4 190 times means re-encoding: minutes of work, GBs of
  disk, and generation loss — for output nothing downstream consumes

### Cue → line merging

Raw SRT cues are *display* chunks (one sentence split across 2–3 cards); TTS needs
whole utterances. Adjacent cues merge while the gap is small, the sentence hasn't
terminated, and duration/length caps aren't hit.

The 473 → 190 ratio is subtitle-specific, so the gap is **tunable in the UI**
(200/400/800/1200/2000 ms) with a live line count. Re-merging re-attaches any
translated SRT and invalidates stale clips.

---

## Step 2 — Speaker Diarization

Tags every clip with a speaker → `out/script_speakers.json` + `S02.done`.

### Two corrections to the original plan

**1. campplus does not take a waveform.** Its ONNX graph starts *after* feature
extraction, so the input is `(batch, frames, 80)` kaldi fbank with CMN applied.
`Fbank.kt` implements it: 25 ms window, 10 ms shift, Povey window, HTK mel,
radix-2 FFT. *Verified against a naive DFT (max error 4e-12), kaldi frame counts
(1 s → 98), and tone-to-bin placement.*

**2. The threshold direction was inverted.** Raising 0.55 → 0.75 gives *more*
speakers, not fewer. The pairwise `if sim > thr: same speaker` merge is
**single-linkage**, which chains — one ambiguous clip bridges two speakers, and
lowering the threshold to stop that shatters them into singletons. That is how 3
speakers became 29.

This uses **average linkage**, comparing cluster means, so the threshold is the
*stopping* similarity:

| Threshold | Effect |
|---|---|
| Lower (0.20–0.45) | merges more → **fewer** speakers |
| Higher (0.70+) | stops sooner → **more** speakers |

On simulated 190-clip data with campplus-like statistics, 0.20–0.60 all recover
exactly 120/50/20 at 100 % purity; 0.70 fragments into 28 — close to the 29
originally observed. A unit test pins the direction.

**Since you usually know the count, use "I know the count"** — clustering to
exactly *k* is more robust than any threshold.

Embeddings cache to `speaker_embeds.bin`, so re-tuning is instant rather than
190 fresh inferences.

---

## Step 3 — Emotion

Tags each clip with one of nine emotions → `out/script_emotion.json` +
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

---

## URL download

There is no helper server. `VideoResolver` runs in the app:

| Input | Result |
|---|---|
| Direct `.mp4` / `.webm` / `.m3u8` link | ✅ downloads |
| Page with `<video src>`, `og:video`, JSON `contentUrl` | ✅ finds and downloads |
| iQIYI, YouTube, Netflix, Hotstar | ❌ named and refused |

The refusal is deliberate. yt-dlp is ~200k lines of Python with per-site
extractors, and iQIYI additionally signs its streams in JavaScript — that cannot
run inside an APK. For those, download on a PC and pick the file from **Gallery**.

---

## Storage

Writing to `/storage/emulated/0/AI` needs **All-files access**, which Android
only grants from Settings. Until then the app uses its own external directory
(`/Android/data/com.azim.vdub/files/AI/`) — no permission required, everything
works, but `adb push` and other apps cannot reach it. A banner offers the
upgrade, and models are looked up in **both** roots.

```
/storage/emulated/0/AI/
├── models/                      campplus.onnx, emotion2vec.onnx, …
└── vdub_projects/{project}/
    ├── input_video.mp4
    ├── org_audio.wav            16 kHz mono PCM16
    ├── subs/original.srt · subs/translated.srt
    ├── clips/line_0000.wav … line_0189.wav
    ├── out/script_raw.json
    ├── out/script_speakers.json
    ├── out/script_emotion.json
    ├── out/speaker_embeds.bin   cached, so re-clustering is instant
    └── S01.done · S02.done · S03.done
```

**Resume-safe:** Room DB **plus** `S0x.done` markers on disk. Reopening a project
rehydrates from whatever actually exists in the folder, so state survives a
reinstall or DB loss — not just an app restart.

---

## Source layout

| Area | Files |
|---|---|
| **Audio** | `AudioExtractor` (MediaCodec → 16 kHz), `WavIo` (RIFF + slicing), `ClipCutter`, `Fbank` (kaldi filterbank) |
| **Models** | `SpeakerEmbedder` (CAM++), `SpeakerCluster` (average-linkage AHC), `EmotionClassifier` (emotion2vec + head) |
| **Subtitles** | `SrtParser` (parse, merge, render) |
| **Net** | `VideoResolver`, `DownloadClient`, `ModelDownloader` |
| **Data** | `VdubPaths`, `ModelCatalog`, `ProjectRepository`, `ProjectDb` (Room) |
| **UI** | `MainActivity` (4 screens), `Step1/2/3ViewModel`, `SettingsScreen`, section composables |

---

## Build

The Gradle **wrapper JAR is not committed**. Android Studio generates it on first
open, or:

```bash
gradle wrapper --gradle-version 8.9
./gradlew assembleDebug
./gradlew testDebugUnitTest      # 45 unit tests
adb install app/build/outputs/apk/debug/app-debug.apk
```

JDK 17 · `compileSdk 34` · `minSdk 26` · arm64-v8a.

CI builds the APK on every push and republishes the `step1-latest` release.

### Tests

45 unit tests covering the parts that fail silently rather than loudly:

| Suite | Guards |
|---|---|
| `SrtParserTest` | merge is lossless, ids sequential, SRT round-trips |
| `WavIoTest` | slices byte-identical to source, clamping at both ends |
| `FbankTest` | kaldi frame counts, mel placement, CMN, no NaN on silence |
| `SpeakerClusterTest` | recovers 3 speakers from 190 clips, **threshold direction**, no single-link chaining |
| `EmotionTest` | head parsing, label cleanup, exaggeration mapping |
| `ModelCatalogTest` | mirrors are https, ids unique, **peak RAM < total** |
| `VideoResolverTest` | media extraction from realistic markup |

---

## Known limitations

- **Chatterbox ships PyTorch weights, not ONNX.** `t3_hi_int8.safetensors` +
  `ve.pt` cannot be loaded by ONNX Runtime, which is the only inference engine
  in the app today. Settings will download and store them, and says plainly
  that they will not run until Step 5 ships an inference path for that format.
  It also needs the S3Gen vocoder (~1.06 GB) fetched separately, and ~1.9 GB RAM
  while speaking — the one stage that genuinely strains a 6 GB phone.
- Chatterbox output carries a **PerTh watermark** (upstream behaviour), and you
  should have permission from whoever owns a voice before cloning it.
- **BGM separation (TIGER-DnR) is not included.** That repo ships only
  `safetensors`; converting to ONNX requires a PyTorch export step that cannot
  run on a phone.
- **NLLB is CC-BY-NC** — non-commercial use only.
- Steps 4 and 5 have their models wired into the catalog but the stages
  themselves are not built yet.
