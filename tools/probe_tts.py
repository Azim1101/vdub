#!/usr/bin/env python3
"""Read the real ONNX signatures of the TTS engines we are about to wire in.

The dev sandbox cannot reach huggingface.co, and a wrong guess about an input
name or a tensor rank does not fail at build time — it fails on a phone, after
a multi-hundred-megabyte download. So the graphs are inspected where the
network exists, and the answers are pinned into Kotlin.

Downloads only what is needed to answer a question:
  - every small file (tokens, configs)
  - the int8 graphs, which are the ones the app will ship
Then actually runs an end-to-end synthesis for each engine, because a matching
signature still says nothing about whether the pipeline produces audio.
"""
from __future__ import annotations

import json
import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

CACHE = Path("/tmp/ttsprobe")
CACHE.mkdir(parents=True, exist_ok=True)

DHVAANI = "Bbkblo/DhVaani-0.5-ONNX"
INDRI = "Bbkblo/indri-0.1-124m-tts-ONNX"
MIMI = "onnx-community/kyutai-mimi-ONNX"


def get(repo: str, path: str) -> Path:
    """Download `path` from `repo` once, return the local file."""
    local = CACHE / repo.replace("/", "__") / path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    print(f"  fetch {repo}/{path}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vdub-probe"})
    with urllib.request.urlopen(req, timeout=600) as r, open(local, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"        {local.stat().st_size:,} bytes", flush=True)
    return local


def describe(sess: ort.InferenceSession, label: str) -> None:
    print(f"\n### {label}")
    for i in sess.get_inputs():
        print(f"  IN   {i.name:<22} {str(i.type):<26} {i.shape}")
    for o in sess.get_outputs():
        print(f"  OUT  {o.name:<22} {str(o.type):<26} {o.shape}")


def load(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


# --------------------------------------------------------------- DhVaani

SR = 24000
N_FFT, HOP, N_MELS = 1024, 256, 100
FEAT_SCALE = 0.1


def vocos_fbank(wav: np.ndarray, fb: np.ndarray, win: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    wav_p = np.pad(wav, (pad, pad), mode="constant")
    if wav_p.size < N_FFT:
        wav_p = np.pad(wav_p, (0, N_FFT - wav_p.size))
    n_frames = 1 + (len(wav_p) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        wav_p, shape=(n_frames, N_FFT),
        strides=(wav_p.strides[0] * HOP, wav_p.strides[0]),
    ).copy()
    spec = np.fft.rfft(frames * win, n=N_FFT, axis=1)
    mel = np.abs(spec).astype(np.float32) @ fb
    logmel = np.log(np.clip(mel, 1e-7, None))
    num_frames = int((round(len(wav)) + HOP // 2) // HOP)
    if logmel.shape[0] > num_frames:
        logmel = logmel[:num_frames]
    elif logmel.shape[0] < num_frames:
        logmel = np.pad(logmel, ((0, num_frames - logmel.shape[0]), (0, 0)), mode="edge")
    return logmel.astype(np.float32)


def probe_dhvaani() -> None:
    print("\n" + "=" * 70)
    print("DhVaani-0.5 ONNX  (ZipVoice flow-matching, zero-shot cloning)")
    print("=" * 70)

    te_p = get(DHVAANI, "text_encoder_int8.onnx")
    fm_p = get(DHVAANI, "fm_decoder_int8.onnx")
    vo_p = get(DHVAANI, "vocoder_backbone.onnx")
    head_p = get(DHVAANI, "vocos_head.npz")
    melfb_p = get(DHVAANI, "mel_fb.npz")
    tokens_p = get(DHVAANI, "tokens.txt")
    model_json_p = get(DHVAANI, "model.json")

    print("\n--- model.json ---")
    print(model_json_p.read_text()[:1500])

    te, fm, vo = load(te_p), load(fm_p), load(vo_p)
    describe(te, "text_encoder_int8.onnx")
    describe(fm, "fm_decoder_int8.onnx")
    describe(vo, "vocoder_backbone.onnx")

    head = np.load(head_p)
    print("\n### vocos_head.npz")
    for k in head.files:
        v = head[k]
        print(f"  {k:<18} shape={getattr(v, 'shape', ())} dtype={v.dtype} "
              f"value={v if v.ndim == 0 else ''}")

    melz = np.load(melfb_p)
    print("\n### mel_fb.npz")
    for k in melz.files:
        print(f"  {k:<18} shape={melz[k].shape} dtype={melz[k].dtype}")

    # tokens.txt: how is it delimited, and what is in it?
    raw = tokens_p.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    print("\n### tokens.txt")
    tab = "\t"
    print(f"  lines={len(lines)}  tab_delimited={tab in lines[0]}")
    for ln in lines[:6]:
        print(f"  {ln!r}")
    print("  ...")
    for ln in lines[-4:]:
        print(f"  {ln!r}")

    t2i = {}
    for ln in lines:
        if not ln:
            continue
        parts = ln.rstrip("\n").split("\t")
        if len(parts) == 2:
            t2i[parts[0]] = int(parts[1])
    print(f"  parsed {len(t2i)} tokens; "
          f"has_devanagari={'न' in t2i} has_space={' ' in t2i} "
          f"has_latin_a={'a' in t2i}")

    # ---- end-to-end synthesis from a synthetic reference clip
    print("\n--- end-to-end run ---")
    fb = melz["fb"].astype(np.float32)
    win = melz["window"].astype(np.float32)

    rng = np.random.RandomState(0)
    # 3 s of a vaguely voice-like signal; content does not matter, shapes do
    t = np.arange(int(SR * 3.0)) / SR
    ref = (0.3 * np.sin(2 * np.pi * 140 * t)
           + 0.15 * np.sin(2 * np.pi * 280 * t)
           + 0.05 * rng.randn(t.size)).astype(np.float32)

    feats = vocos_fbank(ref, fb, win)
    prompt_features = (feats[None, :, :] * FEAT_SCALE).astype(np.float32)
    prompt_len = np.array(prompt_features.shape[1], dtype=np.int64)
    print(f"  prompt frames = {int(prompt_len)}")

    def ids(text: str) -> np.ndarray:
        out = [t2i[c] for c in text if c in t2i]
        return np.asarray(out, dtype=np.int64)[None, :]

    text = "यह एक परीक्षण वाक्य है।"
    tokens = ids(text)
    prompt_tokens = ids("यह संदर्भ है।")
    print(f"  text ids={tokens.shape} prompt ids={prompt_tokens.shape}")

    tc = te.run(None, {
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_features_len": prompt_len,
        "speed": np.array(1.0, dtype=np.float32),
    })[0].astype(np.float32)
    print(f"  text_condition {tc.shape}")

    B, T, D = tc.shape
    x = rng.randn(B, T, D).astype(np.float32)
    sc = np.zeros((B, T, D), dtype=np.float32)
    pl = int(prompt_len)
    sc[:, :min(pl, T), :] = prompt_features[:, :min(pl, T), :]

    num_step, t_shift = 8, 0.5
    ts = np.linspace(0.0, 1.0, num_step + 1, dtype=np.float32)
    ts = (t_shift * ts / (1.0 + (t_shift - 1.0) * ts)).astype(np.float32)
    for s in range(num_step):
        v = fm.run(None, {
            "t": np.array(float(ts[s]), dtype=np.float32),
            "x": x,
            "text_condition": tc,
            "speech_condition": sc,
            "guidance_scale": np.array(1.0, dtype=np.float32),
        })[0].astype(np.float32)
        x = x + v * float(ts[s + 1] - ts[s])
    print(f"  after {num_step} euler steps: x {x.shape}")

    mels = np.transpose(x[:, pl:, :], (0, 2, 1)) / FEAT_SCALE
    print(f"  mels to vocoder {mels.shape}")
    hidden = vo.run(None, {vo.get_inputs()[0].name: mels.astype(np.float32)})[0]
    print(f"  vocoder backbone out {hidden.shape}")

    W = head["linear_weight"].astype(np.float32)
    b = head["linear_bias"].astype(np.float32)
    hwin = head["window"].astype(np.float32)
    n_fft, hop, win_len = int(head["n_fft"]), int(head["hop_length"]), int(head["win_length"])
    y = hidden @ W.T + b
    y = np.transpose(y, (0, 2, 1))
    mag, ph = np.split(y, 2, axis=1)
    mag = np.exp(np.clip(mag, None, np.log(1e2)))
    S = mag * (np.cos(ph) + 1j * np.sin(ph))

    Bn, N, Tn = S.shape
    pad = (win_len - hop) // 2
    ifft = np.fft.irfft(S, n=n_fft, axis=1) * hwin[None, :, None]
    out_size = (Tn - 1) * hop + win_len
    acc = np.zeros((Bn, out_size), dtype=np.float32)
    env = np.zeros((out_size,), dtype=np.float32)
    w2 = hwin.astype(np.float32) ** 2
    for k in range(Tn):
        sl = slice(k * hop, k * hop + win_len)
        acc[:, sl] += ifft[:, :, k]
        env[sl] += w2
    audio = (acc[:, pad:out_size - pad] / np.maximum(env[pad:out_size - pad], 1e-11))[0]

    print(f"  AUDIO samples={audio.size} ({audio.size / SR:.2f}s @ {SR}Hz) "
          f"peak={np.abs(audio).max():.3f} rms={np.sqrt((audio ** 2).mean()):.4f}")
    print(f"  istft: n_fft={n_fft} hop={hop} win_length={win_len} "
          f"window_len={hwin.size} pad={pad}")
    print("  DhVaani: RUNNABLE end to end ✅")


# ----------------------------------------------------------------- Indri

def probe_indri() -> None:
    print("\n" + "=" * 70)
    print("Indri-0.1-124m ONNX LM  +  Mimi ONNX decoder")
    print("=" * 70)

    lm_p = get(INDRI, "indri_lm_int8.onnx")
    cfg_p = get(INDRI, "config.json")
    gen_p = get(INDRI, "generation_config.json")
    vocab_p = get(INDRI, "vocab.json")
    merges_p = get(INDRI, "merges.txt")
    added_p = get(INDRI, "added_tokens.json")
    special_p = get(INDRI, "special_tokens_map.json")

    print("\n--- config.json ---")
    print(cfg_p.read_text()[:1200])
    print("\n--- generation_config.json ---")
    print(gen_p.read_text()[:600])
    print("\n--- special_tokens_map.json ---")
    print(special_p.read_text()[:400])

    lm = load(lm_p)
    describe(lm, "indri_lm_int8.onnx")

    vocab = json.loads(vocab_p.read_text())
    added = json.loads(added_p.read_text())
    print("\n### tokenizer")
    print(f"  vocab.json entries      = {len(vocab)}")
    print(f"  added_tokens.json       = {len(added)}")
    print(f"  merges.txt lines        = {len(merges_p.read_text().splitlines())}")
    combined_max = max(max(vocab.values()), max(added.values()))
    print(f"  max id                  = {combined_max} (vocab size {combined_max + 1})")

    for name in ["[text]", "[convert]", "[mimi]", "[stop]", "[spkr_63]",
                 "[spkr_68]", "[spkr_69]", "[spkr_70]", "[spkr_60]", "[spkr_53]"]:
        print(f"  {name:<12} -> {added.get(name, vocab.get(name))}")

    # how are the audio tokens laid out?
    audio_ids = {k: v for k, v in added.items() if k.startswith("[aud_")}
    print(f"  [aud_*] tokens          = {len(audio_ids)}")
    sample = sorted(audio_ids.items(), key=lambda kv: kv[1])[:3]
    print(f"  first audio tokens      = {sample}")
    spkr = sorted(k for k in added if k.startswith("[spkr_"))
    print(f"  speaker tokens          = {len(spkr)}: {spkr[:16]}")

    OFFSET, NCB, CB = 50257, 8, 2048
    stop = added.get("[stop]", vocab.get("[stop]"))
    print(f"  OFFSET={OFFSET} NCB={NCB} CB={CB} stop={stop}")

    # ---- generate a few audio tokens with the alternating-codebook mask
    print("\n--- LM generation (greedy-ish, top-k) ---")
    # BPE is not implemented here; the probe only needs *some* valid prompt to
    # confirm the loop and the mask work, so use the special tokens plus a few
    # single-character ids.
    def tid(tok: str):
        return added.get(tok, vocab.get(tok))

    prompt = [tid("[text]")]
    for ch in "namaste":
        i = vocab.get(ch)
        if i is not None:
            prompt.append(i)
    prompt += [tid("[convert]"), tid("[mimi]"), tid("[spkr_69]")]
    if any(p is None for p in prompt):
        print(f"  !! unresolved prompt tokens: {prompt}")
        return
    seq = np.asarray(prompt, dtype=np.int64)
    start = seq.shape[0]
    print(f"  prompt ids ({start}) = {seq.tolist()}")

    rng = np.random.RandomState(0)
    max_new = 64
    for _ in range(max_new):
        S = seq.shape[0]
        logits = lm.run(None, {
            "input_ids": seq[None, :],
            "attention_mask": np.ones((1, S), dtype=np.int64),
            "position_ids": np.arange(S, dtype=np.int64)[None, :],
        })[0][0, -1]
        cb = (S - start) % NCB
        masked = np.full_like(logits, -1e9)
        lo, hi = OFFSET + cb * CB, OFFSET + (cb + 1) * CB
        masked[lo:hi] = logits[lo:hi]
        masked[stop] = logits[stop]
        top = np.argpartition(masked, -15)[-15:]
        p = np.exp(masked[top] - masked[top].max())
        p /= p.sum()
        nxt = int(rng.choice(top, p=p))
        seq = np.concatenate([seq, [nxt]])
        if nxt == stop:
            print("  hit [stop]")
            break

    audio_tokens = [int(t) - OFFSET for t in seq[start:] if t != stop]
    ok = all((i % NCB) * CB <= t < ((i % NCB) + 1) * CB
             for i, t in enumerate(audio_tokens))
    print(f"  generated {len(audio_tokens)} audio tokens, codebook_layout_ok={ok}")
    print(f"  logits width = {logits.shape[0]}")

    if len(audio_tokens) < NCB:
        print("  !! too few tokens to decode")
        return

    # ---- Mimi decode: does the ONNX decoder turn those into a waveform?
    print("\n--- Mimi decoder ---")
    dec_p = get(MIMI, "onnx/decoder_model_int8.onnx")
    mimi_cfg = json.loads(get(MIMI, "config.json").read_text())
    print(f"  num_quantizers={mimi_cfg['num_quantizers']} "
          f"codebook_size={mimi_cfg['codebook_size']} "
          f"frame_rate={mimi_cfg['frame_rate']} sr={mimi_cfg['sampling_rate']}")

    dec = load(dec_p)
    describe(dec, "mimi decoder_model_int8.onnx")

    # Indri emits tokens interleaved across 8 codebooks; deserialise to
    # (num_codebooks, frames) and subtract each codebook's base.
    arr = np.asarray(audio_tokens, dtype=np.int64)
    cbs = [arr[i::NCB] for i in range(NCB)]
    n = min(len(c) for c in cbs)
    codes = np.vstack([cbs[i][:n] - CB * i for i in range(NCB)])[None, :, :]
    print(f"  codes {codes.shape} min={codes.min()} max={codes.max()}")

    n_q = int(mimi_cfg["num_quantizers"])
    inp = dec.get_inputs()[0]
    print(f"  decoder expects {inp.name} {inp.shape}")

    for label, arr_in in [
        (f"{NCB} codebooks (as generated)", codes),
        (f"{n_q} codebooks (zero-padded)",
         np.concatenate([codes, np.zeros((1, n_q - NCB, n), dtype=np.int64)], axis=1)),
    ]:
        try:
            wav = dec.run(None, {inp.name: arr_in.astype(np.int64)})[0]
            a = np.asarray(wav).reshape(-1)
            print(f"  {label}: OK -> {np.asarray(wav).shape} "
                  f"({a.size / 24000:.2f}s) peak={np.abs(a).max():.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: FAILED -> {str(e)[:300]}")


def main() -> int:
    failures = []
    for name, fn in [("DhVaani", probe_dhvaani), ("Indri", probe_indri)]:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(name)
            print(f"\n!!! {name} probe raised:")
            traceback.print_exc()
    print("\n" + "=" * 70)
    print("failures:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
