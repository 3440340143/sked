"""exp6: architecture ablation for the global-receptive-field invariant.

exp3/exp5 ran this task with ternary weights at dim=64 and neither the
full-attention upper bound nor any SKED variant learned it (all in 0.24-0.39),
so the ablation was uninterpretable -- the quantizer, not the architecture,
dominated the outcome. exp6 removes that confound:

  * ternary weights OFF (plain fp32 linear layers) so the ablation measures
    architecture, not quantisation
  * dim 128 instead of 64
  * enough optimisation steps for an induction circuit to form

The `--gate` mode runs the full-attention upper bound alone and prints its
learning curve. If the upper bound does not solve the task, nothing downstream
is worth measuring.

Variants
  dense_causal   full attention, no cross-attention      (upper bound)
  local_only     windowed, no cross-attention            (lower bound)
  sked_tail0     windowed encoder + cross-attention      (invariant violated)
  sked_tail1     dense-tail encoder + cross-attention    (invariant holds)

Run:  python experiments/exp6_ablation.py --probe   # training-free probe
      python experiments/exp6_ablation.py --gate    # upper bound only
      python experiments/exp6_ablation.py           # full ablation
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sked_core import SKEDModel, DENSE_WINDOW  # noqa: E402

VOCAB = 64
N_KEYS = 16
N_VALS = 16
N_PAIRS = 6
SEQ_LEN = 2 * N_PAIRS + 2
WINDOW = 4
SEEDS = (0, 1)
STEPS = 1500
BATCH = 64
LR = 1e-3
EVAL_EVERY = 250
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

COMMON = dict(vocab_size=VOCAB, dim=64, num_heads=4, enc_layers=2, dec_layers=2,
              enc_ffn=128, moe_hidden=128, num_experts=4, top_k=2, ternary=False)

VARIANTS = {
    "dense_causal": dict(window_size=DENSE_WINDOW, use_cross_attention=False),
    "local_only":   dict(window_size=WINDOW, use_cross_attention=False),
    "sked_tail0":   dict(window_size=WINDOW, use_cross_attention=True, enc_dense_tail=0),
    "sked_tail1":   dict(window_size=WINDOW, use_cross_attention=True, enc_dense_tail=1),
}


def build(variant):
    return SKEDModel(**COMMON, **VARIANTS[variant])


def make_batch(batch, generator):
    keys = torch.stack([torch.randperm(N_KEYS, generator=generator)[:N_PAIRS]
                        for _ in range(batch)])
    vals = torch.randint(N_KEYS, N_KEYS + N_VALS, (batch, N_PAIRS), generator=generator)
    sep = torch.full((batch, 1), N_KEYS + N_VALS, dtype=torch.long)
    qidx = torch.randint(0, N_PAIRS, (batch,), generator=generator)
    ar = torch.arange(batch)
    query = keys[ar, qidx].unsqueeze(1)
    # interleave as [k0,v0,k1,v1,...] so the task is standard MQAR: attend to the
    # matching key, then read the token immediately after it (induction head).
    seq = torch.stack([keys, vals], dim=2).reshape(batch, -1)
    seq = torch.cat([seq, sep, query], dim=1)
    return seq, vals[ar, qidx]


def probe(seq_len=SEQ_LEN):
    """Training-free receptive-field probe.

    Perturb position 0 and measure the induced change in ``global_k`` at the first
    and last positions. A zero at the last position means information from the
    start of the sequence cannot reach that particular KV entry.

    The tempting reading -- "so the shared KV is not global, and cross-attention
    must therefore be useless" -- is wrong, and the ablation below shows it. The
    decoder cross-attends over the *whole* ``global_k`` set: position p appears in
    ``global_k[p .. p+W-1]``, so a windowed encoder still leaves every position
    addressable. This probe measures a real but irrelevant quantity, and is kept
    as the record of a hypothesis that the ablation refuted.
    """
    rows = []
    for tail in (0, 1, COMMON["enc_layers"]):
        torch.manual_seed(0)
        m = SKEDModel(window_size=WINDOW, use_cross_attention=True,
                      enc_dense_tail=tail, **COMMON)
        m.eval()
        x = torch.randint(0, VOCAB, (1, seq_len))
        x2 = x.clone()
        x2[0, 0] = (x2[0, 0] + 1) % VOCAB
        pos = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            _, (gk1, _) = m._step(x, pos, None, None, None)
            _, (gk2, _) = m._step(x2, pos, None, None, None)
        rows.append({
            "enc_dense_tail": tail,
            "delta_at_pos0": (gk1[:, :, 0] - gk2[:, :, 0]).abs().max().item(),
            "delta_at_last": (gk1[:, :, -1] - gk2[:, :, -1]).abs().max().item(),
        })
    return rows


@torch.no_grad()
def evaluate(model, generator, n_batches=8):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        seq, target = make_batch(128, generator)
        pred = model(seq)[:, -1, :].argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += target.numel()
    model.train()
    return correct / total


def run(variant, seed, steps):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 999)
    model = build(variant)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    t0 = time.time()
    curve = []
    for step in range(steps):
        seq, target = make_batch(BATCH, g)
        loss = F.cross_entropy(model(seq)[:, -1, :], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            acc = evaluate(model, g)
            curve.append([step + 1, round(acc, 4)])
            print(f"    step {step+1:>5}  acc {acc:.3f}  loss {loss.item():.3f}", flush=True)
    acc = evaluate(model, g, n_batches=16)
    return {"variant": variant, "seed": seed, "params": n_params, "ternary": COMMON["ternary"],
            "final_acc": round(acc, 4), "sec": round(time.time() - t0, 1), "curve": curve}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    if args.probe:
        rows = probe()
        print(f"receptive-field probe (perturb token 0, seq_len={SEQ_LEN}, window={WINDOW})")
        print(f"{'enc_dense_tail':>14} | {'|d global_k[0]|':>16} | {'|d global_k[-1]|':>16}")
        print("-" * 56)
        for r in rows:
            print(f"{r['enc_dense_tail']:>14} | {r['delta_at_pos0']:>16.6f} | "
                  f"{r['delta_at_last']:>16.6f}")
        path = os.path.join(RESULTS, "exp6_receptive_field.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": {"seq_len": SEQ_LEN, "window": WINDOW}, "rows": rows}, f, indent=2)
        print(f"\nwrote {path}")
        return

    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    variants = ["dense_causal"] if args.gate else (
        [args.only] if args.only else list(VARIANTS))

    print(f"task: MQAR {N_PAIRS} pairs, seq_len={SEQ_LEN}, window={WINDOW}, "
          f"chance={1/N_VALS:.3f} | dim={COMMON['dim']} ternary={COMMON['ternary']} "
          f"steps={args.steps}")
    print(f"{'variant':>14} {'seed':>5} {'params':>10} {'final acc':>10} {'sec':>8}")
    print("-" * 56)
    out = []
    for v in variants:
        for s in seeds:
            print(f"  {v} seed {s}:", flush=True)
            r = run(v, s, args.steps)
            out.append(r)
            print(f"{r['variant']:>14} {r['seed']:>5} {r['params']:>10,} "
                  f"{r['final_acc']:>10.3f} {r['sec']:>8.1f}", flush=True)

    print("\nsummary (mean over seeds)")
    for v in variants:
        accs = [r["final_acc"] for r in out if r["variant"] == v]
        print(f"{v:>14} {sum(accs)/len(accs):>10.3f}")

    tag = "gate" if args.gate else (args.only or "ablation")
    path = os.path.join(RESULTS, f"exp6_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_pairs": N_PAIRS, "n_vals": N_VALS, "seq_len": SEQ_LEN,
                              "window": WINDOW, "steps": args.steps, "lr": LR,
                              "seeds": list(seeds), "chance": 1 / N_VALS,
                              "model": COMMON}, "runs": out}, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()