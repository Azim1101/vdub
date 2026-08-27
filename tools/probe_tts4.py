#!/usr/bin/env python3
"""Can the fixed-32 Mimi decoder be fed 8 codebooks *faithfully*?

Round three proved plain zero-padding is not usable: 6.83 dB against the
8-codebook decode upstream actually performs, in pure torch, so it is the
padding and not the ONNX quantization.

But look at *why* it fails. Mimi's residual quantizer decodes by summing one
embedding per codebook and running a linear output projection:

    latent = proj( sum_q  codebook_q[ idx_q ] )

Feeding index 0 to the 24 unused codebooks therefore adds

    C = sum_{q>=8} codebook_q[0]

to every frame — and because `proj` is linear, the error in latent space is
exactly `W·C`, a constant, no matter what the real 8 codebooks contain.

C is only zero-ish by accident. But nothing forces the padding to be index 0:
any index is legal, and with 24 codebooks × 2048 vectors there is enormous
freedom to choose a set whose sum cancels. Pick indices minimising ||C|| and
the injected error should collapse.

This probe finds that set and measures whether it does.
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


def score(label: str, ref: np.ndarray, cand: np.ndarray) -> float:
    m = min(ref.size, cand.size)
    a, b = ref[:m], cand[:m]
    den = np.sqrt((a ** 2).mean()) * np.sqrt((b ** 2).mean())
    corr = float((a * b).mean() / den) if den > 0 else float("nan")
    err = a - b
    snr = 10 * np.log10(max((a ** 2).mean(), 1e-20) / max((err ** 2).mean(), 1e-20))
    print(f"  {label:<46} corr={corr:+.4f}  snr={snr:8.2f} dB")
    return snr


def main() -> int:
    import torch
    from transformers import MimiModel

    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    n_q = model.config.num_quantizers
    print(f"mimi: num_quantizers={n_q} semantic={model.config.num_semantic_quantizers}")

    rvq = model.quantizer
    print(f"quantizer type: {type(rvq).__name__}")
    print(f"  attrs: {[a for a in dir(rvq) if 'quantizer' in a or 'proj' in a]}")

    # ---- pull out the codebook embeddings for the acoustic quantizers
    sem = rvq.semantic_residual_vector_quantizer
    aco = rvq.acoustic_residual_vector_quantizer
    print(f"  semantic layers={len(sem.layers)} acoustic layers={len(aco.layers)}")

    def embed_of(layer):
        cb = layer.codebook
        for attr in ("embed", "embed_avg", "weight", "_codebook"):
            v = getattr(cb, attr, None)
            if isinstance(v, torch.Tensor):
                return v.detach().float()
            if v is not None and hasattr(v, "embed"):
                return v.embed.detach().float()
        raise RuntimeError(f"cannot find codebook weights on {type(cb).__name__}: "
                           f"{[a for a in dir(cb) if not a.startswith('__')][:20]}")

    e0 = embed_of(aco.layers[0])
    print(f"  acoustic codebook tensor {tuple(e0.shape)} dtype={e0.dtype}")

    # Codebooks 8..31 overall = acoustic layers 7..30 (layer 0 is codebook 1).
    n_sem = model.config.num_semantic_quantizers
    pad_layers = list(range(8 - n_sem, len(aco.layers)))
    print(f"  padding acoustic layers {pad_layers[0]}..{pad_layers[-1]} "
          f"({len(pad_layers)} codebooks)")

    embeds = [embed_of(aco.layers[i]) for i in pad_layers]
    dim = embeds[0].shape[1]

    zeros_sum = torch.stack([e[0] for e in embeds]).sum(0)
    print(f"\n  ||C|| with all-zero indices     = {zeros_sum.norm().item():.6f}")

    # ---- greedy cancellation, then a few refinement sweeps
    idx = [0] * len(embeds)
    running = zeros_sum.clone()
    for pos, e in enumerate(embeds):
        running = running - e[idx[pos]]
        d = (running.unsqueeze(0) + e).norm(dim=1)
        best = int(torch.argmin(d))
        idx[pos] = best
        running = running + e[best]
    print(f"  ||C|| after greedy pass         = {running.norm().item():.6f}")

    for sweep in range(6):
        improved = False
        for pos, e in enumerate(embeds):
            without = running - e[idx[pos]]
            d = (without.unsqueeze(0) + e).norm(dim=1)
            best = int(torch.argmin(d))
            if best != idx[pos]:
                improved = True
            idx[pos] = best
            running = without + e[best]
        print(f"  ||C|| after sweep {sweep + 1}             = {running.norm().item():.6f}")
        if not improved:
            break

    norm_zero = zeros_sum.norm().item()
    norm_opt = running.norm().item()
    print(f"\n  chosen padding indices = {idx}")
    print(f"  ||C||: {norm_zero:.5f} -> {norm_opt:.5f}  "
          f"({norm_zero / max(norm_opt, 1e-9):.0f}x smaller)")

    # a scale reference: how big is a typical real codebook vector?
    print(f"  typical ||codebook vector|| = {embeds[0].norm(dim=1).mean().item():.5f}")

    # ---- does that translate into audio?
    dec_i8 = ort.InferenceSession(str(get(MIMI, "onnx/decoder_model_int8.onnx")),
                                  providers=["CPUExecutionProvider"])
    dec_f32 = ort.InferenceSession(str(get(MIMI, "onnx/decoder_model.onnx")),
                                   providers=["CPUExecutionProvider"])
    name = dec_i8.get_inputs()[0].name

    results = {}
    for seed in (1, 2, 3):
        wav = speechlike(2.0, seed=seed)
        with torch.no_grad():
            codes = model.encode(torch.from_numpy(wav)[None, None, :]).audio_codes
            ref8 = model.decode(codes[:, :8, :]).audio_values.numpy().reshape(-1)

        c = codes.numpy().astype(np.int64)
        eight = c[:, :8, :]
        T = eight.shape[2]

        zero_pad = np.concatenate(
            [eight, np.zeros((1, n_q - 8, T), dtype=np.int64)], axis=1)
        opt_pad = np.concatenate(
            [eight, np.tile(np.asarray(idx, dtype=np.int64)[None, :, None], (1, 1, T))],
            axis=1)

        print(f"\n--- seed {seed} ---")
        with torch.no_grad():
            t_opt = model.decode(torch.from_numpy(opt_pad)).audio_values.numpy().reshape(-1)
        s_torch = score("torch, 8 + cancelling indices", ref8, t_opt)

        def run(sess, arr):
            return np.asarray(sess.run(None, {name: arr})[0]).reshape(-1)

        s_zero_i8 = score("onnx int8, 8 + zeros", ref8, run(dec_i8, zero_pad))
        s_opt_i8 = score("onnx int8, 8 + cancelling indices", ref8, run(dec_i8, opt_pad))
        s_opt_f32 = score("onnx fp32, 8 + cancelling indices", ref8, run(dec_f32, opt_pad))
        results[seed] = (s_torch, s_zero_i8, s_opt_i8, s_opt_f32)

    print("\n" + "=" * 72)
    avg = np.mean([v for v in results.values()], axis=0)
    print(f"average over {len(results)} clips:")
    print(f"  torch      8 + cancelling : {avg[0]:7.2f} dB")
    print(f"  onnx int8  8 + zeros      : {avg[1]:7.2f} dB   <- rejected in round 3")
    print(f"  onnx int8  8 + cancelling : {avg[2]:7.2f} dB")
    print(f"  onnx fp32  8 + cancelling : {avg[3]:7.2f} dB")

    print("\nPADDING_INDICES =", json.dumps(idx))
    if avg[2] > 20 or avg[3] > 20:
        print("VERDICT: cancelling indices make the fixed-32 graph faithful ✅")
    else:
        print("VERDICT: still not faithful — Indri needs a real 8-codebook decoder ❌")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
