#!/usr/bin/env python3
"""Round two: the questions the first probe raised.

1. Mimi's ONNX decoder has a fixed 32-codebook input, but Indri emits 8.
   Zero-padding *ran*, but Mimi's quantizer is residual — it sums the
   dequantized vector of every codebook it is handed, and index 0 is a real
   vector, not silence. So 24 spurious vectors get added to every frame.
   Measure that against a true 32-codebook decode before shipping it.
2. If padding is bad, is there a Mimi export that takes 8?
3. Speed. The Indri graph has no KV cache — every step re-runs the whole
   sequence. On a phone that could be unusable, so time it here.
4. Golden token ids from the real tokenizers, to pin the hand-written Kotlin
   tokenizers against.
"""
from __future__ import annotations

import json
import sys
import time
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
SR = 24000


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
    print(f"        {local.stat().st_size:,} bytes", flush=True)
    return local


def load(path: Path, threads: int = 2) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def sig(sess, label):
    print(f"\n### {label}")
    for i in sess.get_inputs():
        print(f"  IN   {i.name:<20} {i.type:<22} {i.shape}")
    for o in sess.get_outputs():
        print(f"  OUT  {o.name:<20} {o.type:<22} {o.shape}")


def speechlike(seconds: float, seed: int = 1) -> np.ndarray:
    """A crude voiced signal: pitch sweep + formants + noise burst."""
    rng = np.random.RandomState(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig_ = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate([1.0, 0.5, 0.33, 0.25, 0.2], start=1):
        sig_ += amp * np.sin(k * phase)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * t)
    sig_ = sig_ * env + 0.02 * rng.randn(n)
    sig_ /= np.abs(sig_).max()
    return (0.7 * sig_).astype(np.float32)


# ------------------------------------------------------- 1 & 2: Mimi codebooks

def probe_mimi_codebooks() -> None:
    print("\n" + "=" * 70)
    print("Mimi: does zero-padding 8 codebooks to 32 corrupt the audio?")
    print("=" * 70)

    enc = load(get(MIMI, "onnx/encoder_model_int8.onnx"))
    dec = load(get(MIMI, "onnx/decoder_model_int8.onnx"))
    sig(enc, "encoder_model_int8.onnx")
    sig(dec, "decoder_model_int8.onnx")

    wav = speechlike(2.0)[None, None, :]
    codes = enc.run(None, {enc.get_inputs()[0].name: wav})[0]
    codes = np.asarray(codes, dtype=np.int64)
    print(f"\n  encoded codes {codes.shape} min={codes.min()} max={codes.max()}")
    n_q = codes.shape[1]

    def decode(c):
        out = dec.run(None, {dec.get_inputs()[0].name: c.astype(np.int64)})[0]
        return np.asarray(out).reshape(-1)

    ref = decode(codes)                                  # all 32 — ground truth
    print(f"  full {n_q}-codebook decode: {ref.size} samples "
          f"peak={np.abs(ref).max():.3f} rms={np.sqrt((ref**2).mean()):.4f}")

    def report(label, cand):
        m = min(cand.size, ref.size)
        a, b = ref[:m], cand[:m]
        denom = np.sqrt((a ** 2).mean()) * np.sqrt((b ** 2).mean())
        corr = float((a * b).mean() / denom) if denom > 0 else float("nan")
        err = a - b
        snr = 10 * np.log10(max((a ** 2).mean(), 1e-20) / max((err ** 2).mean(), 1e-20))
        print(f"  {label:<34} rms={np.sqrt((b**2).mean()):.4f} "
              f"corr={corr:+.3f} snr={snr:6.2f} dB")
        return corr, snr

    # what Indri would actually hand us: first 8 codebooks, rest zeroed
    pad = codes.copy()
    pad[:, 8:, :] = 0
    report("first 8 + zeros (planned)", decode(pad))

    # for reference: how much does dropping 24 codebooks cost at all?
    for keep in (8, 16, 24):
        p = codes.copy()
        p[:, keep:, :] = 0
        report(f"first {keep} + zeros", decode(p))

    # is a zero index actually harmless? compare against padding with a repeat
    # of codebook 0's own values, and against the true 32.
    print("\n  -> if 'first 8 + zeros' correlates well with the full decode,")
    print("     zero-padding is a faithful way to run an 8-codebook model.")

    print("\n--- alternative Mimi exports ---")
    for repo, path in [
        ("BMekiker/mimi-onnx-streaming", "config.json"),
        ("maai-kyoto/continuous-mimi-onnx", "config.json"),
    ]:
        try:
            print(f"  {repo}: {get(repo, path).read_text()[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"  {repo}: unavailable ({str(e)[:120]})")


# ------------------------------------------------------------- 3: Indri speed

def probe_indri_speed_and_tokens() -> None:
    print("\n" + "=" * 70)
    print("Indri: generation speed (no KV cache in the graph) + golden tokens")
    print("=" * 70)

    lm = load(get(INDRI, "indri_lm_int8.onnx"))
    added = json.loads(get(INDRI, "added_tokens.json").read_text())
    vocab = json.loads(get(INDRI, "vocab.json").read_text())

    OFFSET, NCB, CB = 50257, 8, 2048
    stop = added["[stop]"]

    prompt = [added["[text]"]]
    for ch in "hello there this is a test of the speech system":
        i = vocab.get(ch)
        if i is not None:
            prompt.append(i)
    prompt += [added["[convert]"], added["[mimi]"], added["[spkr_69]"]]
    seq = np.asarray(prompt, dtype=np.int64)
    start = seq.shape[0]

    rng = np.random.RandomState(0)
    n_new = 240                       # 30 frames = 2.4 s of audio at 12.5 fps
    t0 = time.time()
    for _ in range(n_new):
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
            break
    dt = time.time() - t0
    made = seq.shape[0] - start
    audio_sec = (made / NCB) / 12.5
    print(f"\n  {made} tokens in {dt:.1f}s  = {made/dt:.1f} tok/s")
    print(f"  that is {audio_sec:.2f}s of audio -> RTF {dt/max(audio_sec,1e-6):.1f}x "
          f"on a 2-core runner")
    print(f"  a 190-line clip at ~3s each would be "
          f"{190 * 3 * dt / max(audio_sec, 1e-6) / 60:.0f} min of LM time")

    # ---- golden tokenizer vectors, from the real tokenizer
    print("\n--- tokenizer golden vectors ---")
    try:
        from transformers import AutoTokenizer
        tok_dir = CACHE / "indri_tok"
        tok_dir.mkdir(exist_ok=True)
        for f in ["tokenizer.json", "vocab.json", "merges.txt",
                  "added_tokens.json", "special_tokens_map.json",
                  "tokenizer_config.json"]:
            (tok_dir / f).write_bytes(get(INDRI, f).read_bytes())
        tk = AutoTokenizer.from_pretrained(str(tok_dir))
        print(f"  loaded {type(tk).__name__}, vocab={tk.vocab_size}")
        for s in [
            "hello",
            "hello world",
            "hi my name is indri",
            "namaste, aap kaise hain",
            "नमस्ते दोस्तों",
            "this is a test.",
            "  double  spaces ",
            "Mixed CASE Text",
        ]:
            print(f"  {s!r:<32} -> {tk.encode(s)}")
        print(f"  specials: [text]={tk.encode('[text]')} "
              f"[convert]={tk.encode('[convert]')} [mimi]={tk.encode('[mimi]')} "
              f"[stop]={tk.encode('[stop]')} [spkr_69]={tk.encode('[spkr_69]')}")
    except Exception as e:  # noqa: BLE001
        print(f"  transformers unavailable: {str(e)[:200]}")

    # ---- what does tokenizer.json look like structurally?
    tj = json.loads(get(INDRI, "tokenizer.json").read_text())
    print("\n--- tokenizer.json structure ---")
    print(f"  top keys      = {list(tj.keys())}")
    print(f"  model.type    = {tj.get('model', {}).get('type')}")
    print(f"  pre_tokenizer = {json.dumps(tj.get('pre_tokenizer'))[:300]}")
    print(f"  normalizer    = {json.dumps(tj.get('normalizer'))[:200]}")
    print(f"  decoder       = {json.dumps(tj.get('decoder'))[:200]}")
    print(f"  added_tokens  = {len(tj.get('added_tokens', []))}")
    mdl = tj.get("model", {})
    print(f"  model.vocab   = {len(mdl.get('vocab', {}))} entries")
    print(f"  model.merges  = {len(mdl.get('merges', []))} entries")
    if mdl.get("merges"):
        print(f"  merge[0]      = {mdl['merges'][0]!r}")


# ---------------------------------------------------------- 4: DhVaani speed

def probe_dhvaani_speed() -> None:
    print("\n" + "=" * 70)
    print("DhVaani: speed and tokenizer goldens")
    print("=" * 70)

    te = load(get(DHVAANI, "text_encoder_int8.onnx"))
    fm = load(get(DHVAANI, "fm_decoder_int8.onnx"))
    vo = load(get(DHVAANI, "vocoder_backbone.onnx"))
    melz = np.load(get(DHVAANI, "mel_fb.npz"))
    head = np.load(get(DHVAANI, "vocos_head.npz"))

    fb = melz["fb"].astype(np.float32)
    win = melz["window"].astype(np.float32)
    print(f"  mel fb {fb.shape}, window {win.shape}")
    print(f"  window[:4]={win[:4]} sum={win.sum():.4f}")
    print(f"  fb col0 nonzero at {np.nonzero(fb[:, 0])[0][:6]}")
    print(f"  fb sum={fb.sum():.3f} max={fb.max():.5f}")

    t2i = {}
    for ln in get(DHVAANI, "tokens.txt").read_text(encoding="utf-8").split("\n"):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) == 2:
            t2i[parts[0]] = int(parts[1])

    print("\n--- DhVaani char tokenizer goldens ---")
    for s in ["नमस्ते", "यह एक परीक्षण है।", "hello", "a b", "नमस्ते dosto"]:
        ids = [t2i[c] for c in s if c in t2i]
        skipped = [c for c in s if c not in t2i]
        print(f"  {s!r:<26} -> {ids}  skipped={skipped}")
    print(f"  '_'={t2i.get('_')} ' '={t2i.get(' ')} '.'={t2i.get('.')} "
          f"'।'={t2i.get('।')} ','={t2i.get(',')}")

    # timing at 8 and 16 steps for a ~4 s line
    n_fft, hop = 1024, 256
    ref = speechlike(3.0)

    def fbank(w):
        pad = n_fft // 2
        wp = np.pad(w, (pad, pad))
        nf = 1 + (len(wp) - n_fft) // hop
        fr = np.lib.stride_tricks.as_strided(
            wp, shape=(nf, n_fft), strides=(wp.strides[0] * hop, wp.strides[0])).copy()
        mel = np.abs(np.fft.rfft(fr * win, n=n_fft, axis=1)).astype(np.float32) @ fb
        lm_ = np.log(np.clip(mel, 1e-7, None))
        want = int((round(len(w)) + hop // 2) // hop)
        if lm_.shape[0] > want:
            lm_ = lm_[:want]
        elif lm_.shape[0] < want:
            lm_ = np.pad(lm_, ((0, want - lm_.shape[0]), (0, 0)), mode="edge")
        return lm_.astype(np.float32)

    pf = (fbank(ref)[None] * 0.1).astype(np.float32)
    pl = np.array(pf.shape[1], dtype=np.int64)
    ids = np.asarray([t2i[c] for c in "यह एक लंबा परीक्षण वाक्य है जो कुछ सेकंड चलता है।"
                      if c in t2i], dtype=np.int64)[None]
    pids = np.asarray([t2i[c] for c in "यह संदर्भ वाक्य है।" if c in t2i],
                      dtype=np.int64)[None]

    for steps in (4, 8, 16):
        t0 = time.time()
        tc = te.run(None, {"tokens": ids, "prompt_tokens": pids,
                           "prompt_features_len": pl,
                           "speed": np.array(1.0, dtype=np.float32)})[0]
        B, T, D = tc.shape
        rng = np.random.RandomState(0)
        x = rng.randn(B, T, D).astype(np.float32)
        sc = np.zeros((B, T, D), dtype=np.float32)
        p = int(pl)
        sc[:, :min(p, T), :] = pf[:, :min(p, T), :]
        ts = np.linspace(0, 1, steps + 1, dtype=np.float32)
        ts = (0.5 * ts / (1.0 + (0.5 - 1.0) * ts)).astype(np.float32)
        for s in range(steps):
            v = fm.run(None, {"t": np.array(float(ts[s]), dtype=np.float32), "x": x,
                              "text_condition": tc, "speech_condition": sc,
                              "guidance_scale": np.array(1.0, dtype=np.float32)})[0]
            x = x + v * float(ts[s + 1] - ts[s])
        mels = np.transpose(x[:, p:, :], (0, 2, 1)) / 0.1
        hid = vo.run(None, {"mels": mels.astype(np.float32)})[0]
        y = hid @ head["linear_weight"].astype(np.float32).T + head["linear_bias"]
        frames = y.shape[1]
        dt = time.time() - t0
        audio_s = frames * hop / SR
        print(f"  {steps:>2} steps: {dt:5.1f}s wall for {audio_s:.2f}s audio "
              f"-> RTF {dt/max(audio_s,1e-6):.2f}x   (text frames T={T}, out={frames})")


def main() -> int:
    bad = []
    for name, fn in [
        ("mimi", probe_mimi_codebooks),
        ("indri", probe_indri_speed_and_tokens),
        ("dhvaani", probe_dhvaani_speed),
    ]:
        try:
            fn()
        except Exception:  # noqa: BLE001
            bad.append(name)
            print(f"\n!!! {name} raised:")
            traceback.print_exc()
    print("\n" + "=" * 70)
    print("failures:", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
