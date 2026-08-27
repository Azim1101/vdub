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
| 5 | Voice cloning + speech (4 engines) | ✅ done |
| 6 | Timing fit + video mux | ✅ done |

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
| Disk — the models you need | **1.13 GB** (DhVaani voice) / 1.72 GB (Q4) / 2.40 GB (mix) |
| **Peak RAM** | **591 MB** analysis · **0.6–1.6 GB** while speaking, by engine |

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
| Chatterbox Q4 (small + fast) | 791 MB | Voice | one of four · MIT |
| Chatterbox mix (Q4 LLM) | 1487 MB | Voice | one of four · MIT |
| DhVaani 0.5 (Indic, fast) | 182 MB | Voice | one of four · Apache-2.0 |
| Indri 0.1 (preset voices) | 465 MB | Voice | one of four · research only |
| Chatterbox Hindi INT8 (PyTorch) | 628 MB | Voice | no — not runnable yet |

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

## Voice engines — pick one

Four engines, selectable in **Settings → Voice engine**. Only the selected one
is downloaded and loaded, so peak RAM is one engine, never the sum.

| | Download | RAM | Speed | Clones? | Licence |
|---|---|---|---|---|---|
| **DhVaani 0.5** | **182 MB** | ~600 MB | **0.84× RTF** | ✅ | Apache-2.0 |
| Chatterbox Q4 | 791 MB | ~1.1 GB | ~1 min/line | ✅ | MIT |
| Chatterbox mix | 1487 MB | ~1.6 GB | ~1 min/line | ✅ | MIT |
| Indri 0.1 | 465 MB | ~900 MB | ~15× RTF | ❌ preset | research only |

**DhVaani is the one to use for Hindi.** It is six times smaller than
Chatterbox, generates *faster than real time* on a phone CPU, is Apache-2.0,
and carries no watermark — while still cloning zero-shot. The speed comes from
its structure: it is flow matching, not an autoregressive loop, so a line is
denoised in a fixed 8 passes rather than emitted one token at a time. It also
conditions on the reference *transcript* as well as its audio, so the app
passes the source text of the exact clips it cloned from.

### The two Chatterbox packs

| | Q4 | mix |
|---|---|---|
| speech_encoder (clone identity) | Q4 | **FP32** |
| conditional_decoder (audio) | Q4 | **FP32** |
| language_model | Q4 | Q4 |

The split is deliberate. The encoder reads the reference clip and fixes *who*
the voice sounds like; the decoder turns tokens into the waveform and fixes
*how good* it sounds. The language model is where the weight actually sits
(1984 MB → 337 MB), so quantizing only it halves the download while leaving the
voice-carrying parts untouched.

Two implementation details worth knowing:

- **Separate folders.** Every engine gets its own. The four share leaf names
  (`vocab.json`, `tokenizer.json`, `language_model.onnx`) for entirely
  different weights, so a flat layout would have one silently overwrite
  another.
- **`language_model_q4.onnx` keeps its upstream name** in the mix pack. An
  external-data graph hardcodes its `.onnx_data` filename, so renaming it fails
  at load time rather than at download time.

Both need `repetition_penalty = 1.2` — the upstream default of 2.0 makes the
quantized language model loop forever.

### Indri, and the decoder that had to be repaired

Indri is a 124M GPT-2 that emits [Mimi](https://huggingface.co/kyutai/mimi)
codec tokens. It is here because it is small and its Hindi presets are natural,
but it **cannot clone** — it conditions on a `[spkr_NN]` token, not on a
reference clip. Step 5 says so before a multi-hour run rather than after, and
assigns each speaker a different preset so at least they stay distinct.

Getting it to produce audio at all took some work, and the reasoning is worth
recording:

Indri's repo ships the language model only — Mimi does not export to ONNX from
`transformers`, so upstream points at llama.cpp for the waveform. A community
ONNX export of Mimi does exist, but its decoder input is frozen at **32
codebooks** while Indri emits **8**.

The obvious fix is to pad the other 24 with index 0. It runs, and it is wrong:

| padding | SNR vs the 8-codebook decode upstream performs |
|---|---|
| zeros | **6.8 dB** — audibly wrong |
| cancelling indices | **34.4 dB** — inaudible |

Measured in PyTorch with no quantization involved, so it is the padding and not
the export. The reason is that Mimi's residual quantizer decodes by *summing*
one embedding per codebook:

```
latent = proj( Σ_q  codebook_q[ index_q ] )
```

so index 0 in the unused 24 adds a constant `C = Σ codebook_q[0]` to every
frame — and index 0 is an ordinary trained vector, not silence. ‖C‖ is 1.38,
more than three times a typical single codebook vector (0.42).

But nothing requires index 0. Any index is legal, and 24 codebooks × 2048
entries leaves plenty of freedom to choose a set that cancels. Solving disjoint
pairs exactly and then running coordinate descent brings ‖C‖ from **1.383 to
0.045** — a 31× reduction, and the decode from 6.8 dB to 34.4 dB. Those indices
are pinned in `MimiDecoder.PADDING_INDICES` and covered by a test, because
"tidying" them back to zeros would still produce audio, just worse.

The decoder ships as **fp16**: int8's own quantization error is 15.6 dB even
when handed all 32 real codebooks — worse than the padding error it would be
masking — while fp16 matches fp32 at half the size.

Indri is also slow. Its exported graph has no KV cache, so every token re-runs
the whole sequence: measured 6.6 tok/s, about 15× slower than real time.

## Step 4 — Translation

**Uploading your own translation is the primary path** — it skips machine
translation entirely, so nothing is downloaded and no model runs.

| Upload | Matched by | Use when |
|---|---|---|
| **SRT** | time overlap | you have a subtitle file from anywhere |
| **JSON** | `utt` | you exported from here and filled in `hi` |

SRT is matched by overlap so a translator's file needn't share our line
boundaries; JSON is matched by id so re-timing cannot misalign it. Export is
offered in both formats.

**A partial upload is reported, not hidden.** Lines with no match stay blank,
the count says how many are missing, and `S04.done` is only written when every
line has text — otherwise a half-covered SRT would pass and those lines would go
silent at the voice stage. Individual lines can be fixed by hand.

Auto-translate with NLLB is present but says plainly that the on-device stage is
not wired up yet.

## Steps 5 & 6 — Speech and the dubbed video

**Speak** turns every translated line into audio in its speaker's cloned voice.
**Build** fits those clips to the original timing and writes `dubbed_video.mp4`.

Four graphs run per line:

| Graph | Job |
|---|---|
| `speech_encoder` | reference clip → speaker identity |
| `embed_tokens` | tokens → embeddings, emotion applied as exaggeration |
| `language_model` | speech tokens, autoregressive with a 30-layer KV cache |
| `conditional_decoder` | tokens → 24 kHz waveform |

**Timing is fitted by overlap-add, not resampling.** Hindi rarely matches the
source duration, and resampling a line to fit raises its pitch — at 2× a 220 Hz
tone measures 878 Hz, a chipmunk. The overlap-add path holds it within 7% while
hitting the target duration exactly. Lines that still do not fit overflow and
are reported rather than being cut mid-word, and a long line can never stamp
over the next one's opening words.

**The mux copies the video track verbatim** and encodes only the new audio, so a
142 MB file takes seconds and loses no quality.

**Speaking is resumable.** Clips already on disk are skipped, which matters when
190 lines take roughly three hours. Speakers are enrolled once and reused.

The tokenizer is byte-level BPE read from `tokenizer.json` — there is no
`transformers` at run time. Text is NFKD-normalised first, matching the export;
without it Devanagari matras land on the wrong tokens and the audio is
mispronounced rather than failing outright.

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

**The app also reopens where you left off.** The last project is remembered and
the shell lands on the first unfinished step, read from those markers — no
pressing Next through work that is already done. A banner explains the jump.

**Long jobs keep running with the screen off.** Trim, embedding, emotion and
speaking run under a foreground service holding a partial wake lock; without
one Android suspends the process at lock and a three-hour run silently stalls.
Progress appears in the notification, and speaking skips clips already on disk,
so an interrupted run continues rather than restarting.

---

## Source layout

| Area | Files |
|---|---|
| **Audio** | `AudioExtractor` (MediaCodec → 16 kHz), `WavIo` (RIFF + slicing), `ClipCutter`, `Fbank` (kaldi filterbank) |
| **Voice** | `TtsEngine` (the interface Step 5 speaks to), `ChatterboxTts`, `DhVaaniTts`, `IndriTts`, `MimiDecoder`, `VoiceEngine` (id → files → class), `NpzReader` |
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

- **Voice cloning uses the ONNX Chatterbox, not the lite safetensors.** The
  project's own `vdub-hindi-dubbing-lite` ships `t3_hi_int8.safetensors` and a
  PyTorch `ve.pt`, and expects the 1.06 GB S3Gen vocoder from upstream — none of
  which ONNX Runtime can execute. `onnx-community/chatterbox-multilingual-ONNX`
  `verify01234/chatterbox-multilingual-ONNX-q4` is the same model family with a
  real ONNX export, includes Hindi, and keeps zero-shot cloning and exaggeration
  control. Two packs are offered — see [Voice engines](#voice-engines--pick-one).
  The lite entry is kept but marked optional and not runnable.
- **Speaking is the heavy stage on Chatterbox** — 1.1–1.6 GB RAM and roughly a
  minute per line, versus 591 MB for everything before it. Generation must use
  `repetition_penalty = 1.2`; the upstream default of 2.0 makes the quantized
  build loop forever. DhVaani is the way out of that: 600 MB and faster than
  real time.
- Chatterbox output carries a **PerTh watermark** (upstream behaviour), and you
  should have permission from whoever owns a voice before cloning it. DhVaani
  does not watermark, which does not make cloning someone's voice any more
  yours to do.
- **Indri cannot clone and is research-licensed.** It offers preset speakers
  only; Step 5 says so before you start. Its LM export also has no KV cache, so
  it is roughly 15× slower than real time.
- **BGM separation (TIGER-DnR) is not included.** That repo ships only
  `safetensors`; converting to ONNX requires a PyTorch export step that cannot
  run on a phone.
- **NLLB is CC-BY-NC** — non-commercial use only.
- **No background separation yet.** "Keep background audio" mixes the original
  track underneath so music and effects survive, but the original dialogue stays
  faintly audible with it. Off by default. Proper separation needs TIGER-DnR,
  which has no ONNX export.
- **Auto-translation is still not wired** — upload a translated SRT or JSON.
