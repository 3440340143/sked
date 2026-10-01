"""
Experiment 4 -- cost of the ternary matmul path on CPU.

BitLinear158 ternarises the weights and int8-quantises the activations, but the
actual multiply still goes through a floating-point GEMM. On hardware without
native ternary instructions this should be *slower* than a plain linear layer
rather than faster, because the quantization is pure overhead.

This experiment measures that overhead and the output perturbation it causes.

Writes results/exp4_ternary_matmul.json
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sked_core import BitLinear158  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

SHAPES = [
    (512, 2048, 2048),    # tokens, in, out
    (2048, 2048, 2048),
    (512, 2048, 5632),
]


def timeit(fn, repeats=20, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    rows = []

    print(f"{'shape (T,in,out)':>22} | {'fp32 ms':>9} {'ternary ms':>11} {'slowdown':>9} "
          f"{'cos sim':>8} {'rel err':>8}")
    print("-" * 80)

    for tokens, fin, fout in SHAPES:
        lin = nn.Linear(fin, fout, bias=False)
        ter = BitLinear158(fin, fout, bias=False)
        ter.load_state_dict(lin.state_dict())   # identical weights, isolate the quantizer

        x = torch.randn(tokens, fin)

        with torch.no_grad():
            y_fp = lin(x)
            y_ter = ter(x)
            cos = torch.nn.functional.cosine_similarity(
                y_fp.reshape(-1), y_ter.reshape(-1), dim=0).item()
            rel = ((y_ter - y_fp).norm() / y_fp.norm()).item()

            ms_fp = timeit(lambda: lin(x))
            ms_ter = timeit(lambda: ter(x))

        rows.append({
            "tokens": tokens, "in": fin, "out": fout,
            "fp32_ms": ms_fp, "ternary_ms": ms_ter,
            "slowdown": ms_ter / ms_fp,
            "cosine_similarity": cos, "relative_error": rel,
        })
        print(f"{f'{tokens},{fin},{fout}':>22} | {ms_fp:>9.2f} {ms_ter:>11.2f} "
              f"{ms_ter/ms_fp:>8.2f}x {cos:>8.4f} {rel:>8.4f}")

    with open(os.path.join(RESULTS, "exp4_ternary_matmul.json"), "w", encoding="utf-8") as f:
        json.dump({"device": "cpu", "threads": torch.get_num_threads(), "rows": rows}, f, indent=2)
    print(f"\nwrote {os.path.join(RESULTS, 'exp4_ternary_matmul.json')}")


if __name__ == "__main__":
    main()