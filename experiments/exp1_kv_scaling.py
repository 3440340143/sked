"""
Experiment 1 -- KV footprint and prefill cost vs. sequence length.

Runs the real SKED implementation at increasing sequence lengths under two
attention regimes and records:

  (a) the exact KV tensor footprint (read off the live cache tensors), and
  (b) prefill wall-clock time.

Regimes:
  * "dense"    -- window set larger than the sequence, i.e. full causal attention
  * "windowed" -- sliding window of W tokens

Measurements are on CPU. Timings indicate scaling behaviour only; they say
nothing about GPU throughput.

Writes results/exp1_kv_scaling.json
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sked_core import SKEDModel, KVCache  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

VOCAB = 256
WINDOW = 256
LENGTHS = [256, 512, 1024, 2048, 4096]


def build(window):
    return SKEDModel(vocab_size=VOCAB, dim=128, num_heads=4, enc_layers=2, dec_layers=2,
                        enc_ffn=256, moe_hidden=192, num_experts=8, top_k=2,
                        window_size=window)


@torch.no_grad()
def measure(model, seq_len, repeats=3):
    ids = torch.randint(0, VOCAB, (1, seq_len))
    pos = torch.arange(seq_len).unsqueeze(0)

    best = float("inf")
    global_bytes = dec_local_bytes = enc_local_bytes = 0
    for _ in range(repeats):
        enc_caches = [KVCache() for _ in model.encoders]
        dec_caches = [KVCache() for _ in model.decoders]
        t0 = time.perf_counter()
        _, (gk, gv) = model._step(ids, pos, enc_caches, dec_caches, None)
        best = min(best, time.perf_counter() - t0)
        global_bytes = (gk.numel() + gv.numel()) * gk.element_size()
        dec_local_bytes = sum(c.k.numel() + c.v.numel() for c in dec_caches) * 4
        enc_local_bytes = sum(c.k.numel() + c.v.numel() for c in enc_caches) * 4

    return {
        "seq_len": seq_len,
        "prefill_ms": best * 1e3,
        "global_kv_mib": global_bytes / 2**20,
        "decoder_local_kv_mib": dec_local_bytes / 2**20,
        "encoder_local_kv_mib": enc_local_bytes / 2**20,
        "sked_total_kv_mib": (global_bytes + dec_local_bytes + enc_local_bytes) / 2**20,
        "dense_baseline_kv_mib": (4 * 2 * 1 * model.num_heads * model.head_dim * seq_len * 4) / 2**20,
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = {"window": WINDOW, "dense": [], "windowed": []}

    dense_model = build(window=10**9)
    windowed_model = build(window=WINDOW)

    print(f"{'L':>6} | {'dense ms':>9} {'wndw ms':>9} | "
          f"{'dense KV':>9} {'SKED KV':>9} {'ratio':>7}")
    print("-" * 66)

    for L in LENGTHS:
        d = measure(dense_model, L)
        w = measure(windowed_model, L)
        out["dense"].append(d)
        out["windowed"].append(w)
        ratio = w["sked_total_kv_mib"] / d["sked_total_kv_mib"]
        print(f"{L:>6} | {d['prefill_ms']:>9.1f} {w['prefill_ms']:>9.1f} | "
              f"{d['sked_total_kv_mib']:>8.1f}M {w['sked_total_kv_mib']:>8.1f}M {ratio:>6.1%}")

    with open(os.path.join(RESULTS, "exp1_kv_scaling.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'exp1_kv_scaling.json')}")


if __name__ == "__main__":
    main()