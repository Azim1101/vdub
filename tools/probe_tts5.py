#!/usr/bin/env python3
"""Finish the Indri decoder question.

Round four found the mechanism and most of the fix. Two things it also showed,
which change what "good" means here:

  * cancelling padding indices took the error from 6.43 dB to 18.41 dB (torch),
    so the approach is right and the greedy search was just weak — it stalled
    at ||C|| = 0.450, which is larger than a single codebook vector (0.415).
  * int8 is the real ceiling: decoding all 32 codebooks through the int8 graph
    scores 17.00 dB against torch, i.e. the quantization costs more than the
    padding does. Chasing padding below that is pointless while int8 is used.

So: search properly (pairwise exact + annealing), and compare every published
precision of the decoder, on quality, size and speed. Then pick.
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


def score(ref, cand):
    m = min(ref.size, cand.size)
    a, b = ref[:m], cand[:m]
    den = np.sqrt((a ** 2).mean()) * np.sqrt((b ** 2).mean())
    corr = float((a * b).mean() / den) if den > 0 else float("nan")
    err = a - b
    snr = 10 * np.log10(max((a ** 2).mean(), 1e-20) / max((err ** 2).mean(), 1e-20))
    return corr, snr


def optimise_padding(embeds, seed=0):
    """Choose one index per padding codebook so their sum cancels.

    Pairwise exact solve first — for two codebooks the best pair is a full
    2048x2048 search, which is one matrix op — then coordinate descent with
    random restarts over whatever is left.
    """
    import torch

    n = len(embeds)
    dim = embeds[0].shape[1]

    def total(idx):
        return torch.stack([embeds[k][idx[k]] for k in range(n)]).sum(0)

    # --- pairwise: minimise ||e_a[i] + e_b[j]|| exactly for each disjoint pair
    idx = [0] * n
    for a in range(0, n - 1, 2):
        b = a + 1
        ea, eb = embeds[a], embeds[b]
        # ||ea_i + eb_j||^2 = |ea_i|^2 + |eb_j|^2 + 2 ea_i . eb_j
        cross = ea @ eb.T
        d2 = (ea.pow(2).sum(1)[:, None] + eb.pow(2).sum(1)[None, :] + 2 * cross)
        flat = int(torch.argmin(d2))
        idx[a], idx[b] = flat // eb.shape[0], flat % eb.shape[0]
    if n % 2:
        idx[-1] = int(torch.argmin(embeds[-1].norm(dim=1)))
    print(f"  after exact pairwise      ||C|| = {total(idx).norm().item():.6f}")

    best_idx = list(idx)
    best_norm = total(idx).norm().item()

    g = torch.Generator().manual_seed(seed)
    for restart in range(6):
        cur = list(best_idx) if restart == 0 else [
            int(torch.randint(0, e.shape[0], (1,), generator=g)) for e in embeds
        ]
        running = total(cur)
        for _ in range(60):
            improved = False
            for k in range(n):
                without = running - embeds[k][cur[k]]
                # ||without + e||^2 minimised over the codebook
                d2 = embeds[k].pow(2).sum(1) + 2 * (embeds[k] @ without) + without.pow(2).sum()
                pick = int(torch.argmin(d2))
                if pick != cur[k]:
                    improved = True
                cur[k] = pick
                running = without + embeds[k][pick]
            if not improved:
                break
        nrm = running.norm().item()
        if nrm < best_norm:
            best_norm, best_idx = nrm, list(cur)
        print(f"  restart {restart}: ||C|| = {nrm:.6f}   (best {best_norm:.6f})")

    return best_idx, best_norm


def main() -> int:
    import torch
    from transformers import MimiModel

    model = MimiModel.from_pretrained("kyutai/mimi").eval()
    n_q = model.config.num_quantizers
    n_sem = model.config.num_semantic_quantizers
    aco = model.quantizer.acoustic_residual_vector_quantizer

    def embed_of(layer):
        return layer.codebook.embed.detach().float()

    pad_layers = list(range(8 - n_sem, len(aco.layers)))
    embeds = [embed_of(aco.layers[i]) for i in pad_layers]
    print(f"padding {len(embeds)} codebooks, dim={embeds[0].shape[1]}")
    zeros_norm = torch.stack([e[0] for e in embeds]).sum(0).norm().item()
    print(f"  ||C|| all-zero indices    = {zeros_norm:.6f}")
    print(f"  typical ||codebook vec||  = {embeds[0].norm(dim=1).mean().item():.6f}")

    t0 = time.time()
    idx, norm = optimise_padding(embeds)
    print(f"\n  best ||C|| = {norm:.6f}  ({zeros_norm / max(norm, 1e-9):.0f}x better "
          f"than zeros, search took {time.time() - t0:.0f}s)")
    print(f"  PADDING_INDICES = {json.dumps(idx)}")

    # ---------------------------------------------------------- precisions
    variants = [
        ("fp32", "onnx/decoder_model.onnx"),
        ("fp16", "onnx/decoder_model_fp16.onnx"),
        ("q4", "onnx/decoder_model_q4.onnx"),
        ("int8", "onnx/decoder_model_int8.onnx"),
    ]
    sessions = {}
    for tag, path in variants:
        p = get(MIMI, path)
        try:
            sessions[tag] = (ort.InferenceSession(
                str(p), providers=["CPUExecutionProvider"]), p.stat().st_size)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}: cannot load ({str(e)[:120]})")

    print("\n" + "=" * 74)
    print("Decoder precision: quality of an 8-codebook decode vs PyTorch")
    print("=" * 74)
    print(f"{'variant':<8} {'MB':>7}  {'zeros-pad':>18}  {'cancelling-pad':>18}  {'all-32':>12}")

    clips = []
    for seed in (1, 2, 3):
        wav = speechlike(2.0, seed=seed)
        with torch.no_grad():
            codes = model.encode(torch.from_numpy(wav)[None, None, :]).audio_codes
            ref8 = model.decode(codes[:, :8, :]).audio_values.numpy().reshape(-1)
            ref32 = model.decode(codes).audio_values.numpy().reshape(-1)
        clips.append((codes.numpy().astype(np.int64), ref8, ref32))

    with torch.no_grad():
        c0 = torch.from_numpy(clips[0][0])
        T0 = c0.shape[2]
        opt0 = torch.cat([c0[:, :8, :], torch.tensor(idx)[None, :, None].expand(1, -1, T0)], 1)
        t_opt = model.decode(opt0).audio_values.numpy().reshape(-1)
    print(f"{'torch':<8} {'-':>7}  {'6.83 (r3)':>18}  "
          f"{score(clips[0][1], t_opt)[1]:>17.2f}  {'-':>12}")

    results = {}
    for tag, (sess, size) in sessions.items():
        name = sess.get_inputs()[0].name
        zs, os_, a32 = [], [], []
        for codes_np, ref8, ref32 in clips:
            T = codes_np.shape[2]
            eight = codes_np[:, :8, :]
            zero_pad = np.concatenate([eight, np.zeros((1, n_q - 8, T), np.int64)], 1)
            opt_pad = np.concatenate(
                [eight, np.tile(np.asarray(idx, np.int64)[None, :, None], (1, 1, T))], 1)

            def run(arr):
                return np.asarray(sess.run(None, {name: arr.astype(np.int64)})[0]).reshape(-1)

            zs.append(score(ref8, run(zero_pad))[1])
            os_.append(score(ref8, run(opt_pad))[1])
            a32.append(score(ref32, run(codes_np))[1])
        results[tag] = (np.mean(zs), np.mean(os_), np.mean(a32))
        print(f"{tag:<8} {size / 1e6:>7.0f}  {np.mean(zs):>17.2f}  "
              f"{np.mean(os_):>17.2f}  {np.mean(a32):>11.2f}")

    # ---------------------------------------------------------------- speed
    print("\n" + "=" * 74)
    print("Decoder speed (2-core runner) — 3 s of audio")
    print("=" * 74)
    frames = int(3.0 * 12.5)
    dummy = np.concatenate(
        [np.random.RandomState(0).randint(0, 2048, (1, 8, frames)).astype(np.int64),
         np.tile(np.asarray(idx, np.int64)[None, :, None], (1, 1, frames))], axis=1)
    for tag, (sess, size) in sessions.items():
        name = sess.get_inputs()[0].name
        sess.run(None, {name: dummy})
        t0 = time.time()
        for _ in range(3):
            out = sess.run(None, {name: dummy})[0]
        dt = (time.time() - t0) / 3
        secs = np.asarray(out).reshape(-1).size / SR
        print(f"  {tag:<6} {dt:6.2f}s for {secs:.2f}s audio -> RTF {dt / secs:.3f}x")

    print("\n" + "=" * 74)
    best = max(results.items(), key=lambda kv: kv[1][1])
    print(f"best padded quality: {best[0]} at {best[1][1]:.2f} dB")
    print("PADDING_INDICES =", json.dumps(idx))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
