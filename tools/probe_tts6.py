#!/usr/bin/env python3
"""Check the hand-written Kotlin DSP against the reference implementation.

The Kotlin engines re-implement, by hand, work that the reference does in
numpy: DhVaani's Vocos filterbank and ISTFT, and Indri's GPT-2 tokenizer. Unit
tests can show those are self-consistent; they cannot show they match what the
model was trained against. A filterbank that is subtly wrong still produces
audio — of a different voice — and a tokenizer off by one id still speaks, just
the wrong words.

This runs the *same inputs* through the reference here, and writes the expected
outputs into a Kotlin test fixture, so the JVM tests can assert against real
numbers instead of against themselves.
"""
from __future__ import annotations

import json
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path("/tmp/ttsprobe")
CACHE.mkdir(parents=True, exist_ok=True)
DHVAANI = "Bbkblo/DhVaani-0.5-ONNX"
INDRI = "Bbkblo/indri-0.1-124m-tts-ONNX"

SR = 24000
N_FFT, HOP, N_MELS = 1024, 256, 100


def get(repo: str, path: str) -> Path:
    local = CACHE / repo.replace("/", "__") / path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    print(f"  fetch {repo}/{path}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vdub-probe"})
    with urllib.request.urlopen(req, timeout=900) as r, open(local, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return local


def tone(n: int, seed: int = 7) -> np.ndarray:
    """Deterministic test signal the Kotlin side can regenerate exactly."""
    rng = np.random.RandomState(seed)
    t = np.arange(n) / SR
    s = (0.6 * np.sin(2 * np.pi * 220.0 * t)
         + 0.3 * np.sin(2 * np.pi * 440.0 * t)
         + 0.1 * np.sin(2 * np.pi * 1000.0 * t))
    return s.astype(np.float32)


def vocos_fbank(wav, fb, win):
    pad = N_FFT // 2
    wav_p = np.pad(wav, (pad, pad), mode="constant")
    if wav_p.size < N_FFT:
        wav_p = np.pad(wav_p, (0, N_FFT - wav_p.size))
    n_frames = 1 + (len(wav_p) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        wav_p, shape=(n_frames, N_FFT),
        strides=(wav_p.strides[0] * HOP, wav_p.strides[0])).copy()
    spec = np.fft.rfft(frames * win, n=N_FFT, axis=1)
    mel = np.abs(spec).astype(np.float32) @ fb
    logmel = np.log(np.clip(mel, 1e-7, None))
    want = int((round(len(wav)) + HOP // 2) // HOP)
    if logmel.shape[0] > want:
        logmel = logmel[:want]
    elif logmel.shape[0] < want:
        logmel = np.pad(logmel, ((0, want - logmel.shape[0]), (0, 0)), mode="edge")
    return logmel.astype(np.float32)


def istft_same(spec, window, n_fft, hop, win_length):
    B, N, T = spec.shape
    pad = (win_length - hop) // 2
    ifft = np.fft.irfft(spec, n=n_fft, axis=1) * window[None, :, None]
    out_size = (T - 1) * hop + win_length
    y = np.zeros((B, out_size), dtype=np.float32)
    env = np.zeros((out_size,), dtype=np.float32)
    w2 = window.astype(np.float32) ** 2
    for t in range(T):
        sl = slice(t * hop, t * hop + win_length)
        y[:, sl] += ifft[:, :, t]
        env[sl] += w2
    y = y[:, pad:out_size - pad]
    env = np.maximum(env[pad:out_size - pad], 1e-11)
    return y / env[None, :]


def main() -> int:
    out: dict = {}

    # ------------------------------------------------------------ DhVaani
    print("=" * 70)
    print("DhVaani: filterbank + ISTFT reference values")
    print("=" * 70)
    melz = np.load(get(DHVAANI, "mel_fb.npz"))
    head = np.load(get(DHVAANI, "vocos_head.npz"))
    fb = melz["fb"].astype(np.float32)
    win = melz["window"].astype(np.float32)

    # the filterbank itself, so Kotlin can prove it read the npz correctly
    out["mel_fb_shape"] = list(fb.shape)
    out["mel_fb_sum"] = float(fb.sum())
    out["mel_fb_row3_sum"] = float(fb[3].sum())
    out["mel_window_sum"] = float(win.sum())
    out["mel_window_first8"] = [float(x) for x in win[:8]]
    print(f"  fb{fb.shape} sum={fb.sum():.5f}  window sum={win.sum():.5f}")

    # a fixed-length tone through the real fbank
    for n_samples in (4096, 12000):
        wav = tone(n_samples)
        feats = vocos_fbank(wav, fb, win)
        key = f"fbank_{n_samples}"
        out[key] = {
            "frames": int(feats.shape[0]),
            "mels": int(feats.shape[1]),
            "sum": float(feats.sum()),
            "frame0_first8": [float(x) for x in feats[0][:8]],
            "frame0_argmax": int(np.argmax(feats[0])),
            "last_frame_argmax": int(np.argmax(feats[-1])),
        }
        print(f"  {key}: frames={feats.shape[0]} sum={feats.sum():.4f} "
              f"argmax0={np.argmax(feats[0])}")

    # ISTFT round trip on a known spectrum
    hwin = head["window"].astype(np.float32)
    n_fft = int(head["n_fft"]); hop = int(head["hop_length"])
    win_length = int(head["win_length"])
    bins = n_fft // 2 + 1
    T = 12
    rng = np.random.RandomState(3)
    mag = np.abs(rng.randn(1, bins, T)).astype(np.float32) * 0.1
    ph = (rng.randn(1, bins, T) * 0.5).astype(np.float32)
    spec = mag * (np.cos(ph) + 1j * np.sin(ph))
    audio = istft_same(spec, hwin, n_fft, hop, win_length)[0]
    out["istft"] = {
        "n_fft": n_fft, "hop": hop, "win_length": win_length,
        "frames": T,
        "samples": int(audio.size),
        "sum": float(audio.sum()),
        "abs_sum": float(np.abs(audio).sum()),
        "first8": [float(x) for x in audio[:8]],
    }
    print(f"  istft: {T} frames -> {audio.size} samples, "
          f"abs_sum={np.abs(audio).sum():.5f}")

    # A pure cosine must survive analysis->synthesis: this catches a wrong
    # inverse-FFT scale or a missing hermitian mirror, which a random spectrum
    # can hide.
    n = 8192
    x = np.cos(2 * np.pi * 500 * np.arange(n) / SR).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    nf = 1 + (len(xp) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(
        xp, shape=(nf, n_fft), strides=(xp.strides[0] * hop, xp.strides[0])).copy()
    S = np.fft.rfft(fr * hwin, n=n_fft, axis=1).T[None, :, :]
    rec = istft_same(S, hwin, n_fft, hop, win_length)[0]
    m = min(rec.size, x.size)
    err = np.abs(rec[:m] - x[:m]).max()
    print(f"  cosine round trip: max abs error = {err:.2e}")
    out["istft_roundtrip_max_err"] = float(err)

    # ------------------------------------------------------------- Indri
    print("\n" + "=" * 70)
    print("Indri: tokenizer golden ids")
    print("=" * 70)
    from transformers import AutoTokenizer
    tok_dir = CACHE / "indri_tok"
    tok_dir.mkdir(exist_ok=True)
    for f in ["tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
              "special_tokens_map.json", "tokenizer_config.json"]:
        (tok_dir / f).write_bytes(get(INDRI, f).read_bytes())
    tk = AutoTokenizer.from_pretrained(str(tok_dir))

    phrases = [
        "hello", "hello world", "hi my name is indri", "this is a test.",
        "namaste, aap kaise hain", "mera naam indri hai",
        "don't stop", "3 apples and 42 pears", "  double  spaces ",
        "aaj mausam accha hai", "yeh ek pariksha hai",
    ]
    out["tokens"] = {}
    for p in phrases:
        ids = tk.encode(p)
        out["tokens"][p] = ids
        print(f"  {p!r:30} -> {ids}")

    out["specials"] = {
        s: tk.encode(s)[0] for s in
        ["[text]", "[convert]", "[mimi]", "[stop]", "[spkr_69]", "[spkr_60]",
         "[spkr_68]", "[spkr_53]", "[spkr_70]", "[spkr_62]", "[spkr_75]",
         "[spkr_77]", "[spkr_66]", "[spkr_63]"]
    }
    print(f"  specials: {out['specials']}")

    # A full Indri prompt, exactly as IndriTts builds it.
    text = "namaste, aap kaise hain"
    prompt = ([out["specials"]["[text]"]] + tk.encode(text)
              + [out["specials"]["[convert]"], out["specials"]["[mimi]"],
                 out["specials"]["[spkr_69]"]])
    out["full_prompt"] = {"text": text, "ids": prompt}
    print(f"  full prompt ({len(prompt)}) = {prompt}")

    dest = Path("app/src/test/resources/tts_reference.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {dest} ({dest.stat().st_size} bytes)")
    print("=" * 70)
    print(json.dumps(out, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
