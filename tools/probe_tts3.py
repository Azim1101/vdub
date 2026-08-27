#!/usr/bin/env python3
"""The decisive Indri question: is zero-padding 8 codebooks to 32 correct?

Round two compared a padded 8-codebook decode against a *32*-codebook decode
and got 6.64 dB. That number is not the answer — a 32-codebook decode is a
higher-fidelity signal, so some of that gap is just detail Indri never
produces. Indri emits 8 codebooks by design, and upstream decodes exactly
those 8 through the PyTorch Mimi.

So the real comparison is:

    torch  Mimi.decode(codes[:, :8, :])      <- what upstream Indri does
    onnx   decoder(codes[:, :8] + 24 zeros)  <- what the app would do

If those two agree, padding is a faithful way to run the fixed-32 ONNX graph
and Indri can ship. If they do not, the app would produce audible artefacts on
every line and needs a different decoder.
"""
from __future__ import annotations

import sys
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

CACHE = Path("/tmp/ttsprobe")
CACHE.mkdir(parents=True, exist_ok=True)
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
    return local


def speechlike(seconds: float, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    s = np.zeros(n)
    for k, amp in enumerate([1.0, 0.5, 0.33, 0.25, 0.2], start=1):
        s += amp * np.sin(k * phase)
    s = s * (0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * t)) + 0.02 * rng.randn(n)
    return (0.7 * s / np.abs(s).max()).astype(np.float32)


def compare(label: str, ref: np.ndarray, cand: np.ndarray) -> float:
    m = min(ref.size, cand.size)
    a, b = ref[:m], cand[:m]
    den = np.sqrt((a ** 2).mean()) * np.sqrt((b ** 2).mean())
    corr = float((a * b).mean() / den) if den > 0 else float("nan")
    err = a - b
    snr = 10 * np.log10(max((a ** 2).mean(), 1e-20) / max((err ** 2).mean(), 1e-20))
    print(f"  {label:<44} corr={corr:+.4f}  snr={snr:7.2f} dB  "
          f"rms={np.sqrt((b ** 2).mean()):.4f}")
    return snr


def main() -> int:
    print("=" * 72)
    print("Indri x Mimi: is the fixed-32 ONNX decoder usable with 8 codebooks?")
    print("=" * 72)

    import torch
    from transformers import MimiModel

    print(f"  torch {torch.__version__}")
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.eval()
    n_q_total = model.config.num_quantizers
    print(f"  mimi num_quantizers={n_q_total} codebook_size={model.config.codebook_size}")

    dec = ort.InferenceSession(
        str(get(MIMI, "onnx/decoder_model_int8.onnx")),
        providers=["CPUExecutionProvider"],
    )
    dec_fp32 = ort.InferenceSession(
        str(get(MIMI, "onnx/decoder_model.onnx")),
        providers=["CPUExecutionProvider"],
    )
    name = dec.get_inputs()[0].name

    wav = speechlike(2.0)
    with torch.no_grad():
        codes = model.encode(torch.from_numpy(wav)[None, None, :]).audio_codes
    print(f"  encoded {tuple(codes.shape)}")

    def onnx_decode(sess, c: np.ndarray) -> np.ndarray:
        return np.asarray(sess.run(None, {name: c.astype(np.int64)})[0]).reshape(-1)

    codes_np = codes.numpy().astype(np.int64)
    eight = codes_np[:, :8, :]
    padded = np.concatenate(
        [eight, np.zeros((1, n_q_total - 8, eight.shape[2]), dtype=np.int64)], axis=1
    )

    print("\n--- ground truth: PyTorch Mimi, 8 codebooks (what Indri does) ---")
    with torch.no_grad():
        ref8 = model.decode(codes[:, :8, :]).audio_values.numpy().reshape(-1)
    print(f"  {ref8.size} samples  rms={np.sqrt((ref8 ** 2).mean()):.4f} "
          f"peak={np.abs(ref8).max():.3f}")

    print("\n--- candidates ---")
    with torch.no_grad():
        ref8_pad_torch = model.decode(
            torch.from_numpy(padded)
        ).audio_values.numpy().reshape(-1)
    snr_torch_pad = compare("torch, 8 + 24 zeros", ref8, ref8_pad_torch)
    snr_onnx_pad = compare("onnx int8, 8 + 24 zeros  (the app's path)", ref8,
                           onnx_decode(dec, padded))
    snr_onnx_fp32 = compare("onnx fp32, 8 + 24 zeros", ref8,
                            onnx_decode(dec_fp32, padded))

    # How good is the ONNX graph at all? Full 32 both ways isolates quantization
    # error from padding error.
    with torch.no_grad():
        ref32 = model.decode(codes).audio_values.numpy().reshape(-1)
    print("\n--- isolating quantization error (full 32, torch vs onnx) ---")
    compare("onnx int8, all 32", ref32, onnx_decode(dec, codes_np))
    compare("onnx fp32, all 32", ref32, onnx_decode(dec_fp32, codes_np))

    print("\n--- is index 0 a real vector, or effectively silence? ---")
    # If padding with zeros were harmless, decoding 8 real + 24 zeros would
    # equal decoding just those 8. Measure the injected energy directly.
    with torch.no_grad():
        only_zeros = model.decode(
            torch.zeros((1, n_q_total, eight.shape[2]), dtype=torch.long)
        ).audio_values.numpy().reshape(-1)
    print(f"  decode(all zeros) rms={np.sqrt((only_zeros ** 2).mean()):.5f} "
          f"peak={np.abs(only_zeros).max():.5f}")
    with torch.no_grad():
        z24 = model.decode(
            torch.cat([torch.zeros((1, 8, eight.shape[2]), dtype=torch.long),
                       torch.zeros((1, n_q_total - 8, eight.shape[2]), dtype=torch.long)],
                      dim=1)
        ).audio_values.numpy().reshape(-1)
    print(f"  decode(8 zeros + 24 zeros) rms={np.sqrt((z24 ** 2).mean()):.5f}")

    print("\n" + "=" * 72)
    verdict = max(snr_onnx_pad, snr_onnx_fp32)
    print(f"padding SNR vs true 8-codebook decode: torch={snr_torch_pad:.2f} dB, "
          f"onnx best={verdict:.2f} dB")
    if snr_torch_pad > 30:
        print("VERDICT: zero-padding is faithful in torch -> the fixed-32 graph")
        print("         can be fed 8 codebooks; remaining gap is ONNX quantization.")
    else:
        print("VERDICT: zero-padding CHANGES the audio -> the fixed-32 ONNX decoder")
        print("         cannot stand in for an 8-codebook decode.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
