"""
Build the paper figures from results/*.json.

Produces PDF (vector) figures into paper/figures/.

  fig_kv_scaling.pdf    measured prototype KV footprint + analytic 32-layer scaling
  fig_ternary.pdf       ternary vs fp32 matmul cost and fidelity on CPU
  fig_ablation.pdf      associative-recall ablation (experiment 6)

Run:  python paper/make_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
})

C_BASE = "#444444"
C_SKED = "#0b6fa4"
C_ALT = "#c0392b"


def load(name):
    with open(os.path.join(RESULTS, name), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure: KV scaling
# ---------------------------------------------------------------------------
def fig_kv_scaling():
    d = load("exp1_kv_scaling.json")
    dense, wndw = d["dense"], d["windowed"]

    L = [r["seq_len"] for r in wndw]
    baseline = [r["dense_baseline_kv_mib"] for r in wndw]
    sked_w = [r["sked_total_kv_mib"] for r in wndw]
    sked_d = [r["sked_total_kv_mib"] for r in dense]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax.plot(L, baseline, "o-", color=C_BASE, lw=1.4, ms=3.5,
            label="per-layer KV (4 layers, fp32)")
    ax.plot(L, sked_d, "s--", color=C_ALT, lw=1.2, ms=3.2, alpha=0.85,
            label="SKED, window $\\geq L$")
    ax.plot(L, sked_w, "^-", color=C_SKED, lw=1.6, ms=4.0,
            label="SKED, $W{=}256$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("KV footprint (MiB)")
    ax.set_title("(a) measured prototype (4 layers, CPU)", fontsize=8)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.legend(loc="upper left")

    # analytic extrapolation to the spec configuration
    layers, heads, hdim, W = 32, 16, 128, 512
    Ls = [2 ** e for e in range(12, 18)]
    per_layer = [layers * 2 * heads * hdim * l * 1 / 2 ** 30 for l in Ls]
    sked_full = [(2 * heads * hdim * l + (layers // 2) * 2 * heads * hdim * W)
                 / 2 ** 30 for l in Ls]
    ax2.plot(Ls, per_layer, "o-", color=C_BASE, lw=1.4, ms=3.5,
             label="per-layer KV (32 layers, FP8)")
    ax2.plot(Ls, sked_full, "^-", color=C_SKED, lw=1.6, ms=4.0,
             label="SKED (1 global copy, $W{=}512$)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("sequence length $L$")
    ax2.set_ylabel("KV footprint (GiB)")
    ax2.set_title("(b) analytic, spec config (not measured)", fontsize=8)
    ax2.grid(alpha=0.25, lw=0.4, which="both")
    ax2.legend(loc="upper left")

    fig.tight_layout(pad=0.4)
    out = os.path.join(FIGDIR, "fig_kv_scaling.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure: ternary matmul cost
# ---------------------------------------------------------------------------
def fig_ternary():
    d = load("exp4_ternary_matmul.json")
    rows = d["rows"]
    labels = [f"$T{{=}}{r['tokens']}$\n${r['in']}{{\\times}}{r['out']}$" for r in rows]
    slow = [r["slowdown"] for r in rows]
    cos = [r["cosine_similarity"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    x = range(len(rows))
    ax.bar(x, slow, color=C_ALT, width=0.55, alpha=0.85)
    ax.axhline(1.0, color=C_BASE, lw=0.9, ls="--")
    ax.text(len(rows) - 0.5, 1.03, "break-even", ha="right", va="bottom",
            fontsize=6.5, color=C_BASE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("slowdown vs. fp32 ($\\times$)")
    ax.set_ylim(0, 2.4)
    ax.set_title("(a) wall-clock cost, CPU", fontsize=8)

    ax2.bar(x, cos, color=C_SKED, width=0.55, alpha=0.85)
    ax2.axhline(1.0, color=C_BASE, lw=0.9, ls="--")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("cosine similarity to fp32")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("(b) output fidelity", fontsize=8)

    fig.tight_layout(pad=0.4)
    out = os.path.join(FIGDIR, "fig_ternary.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure: associative-recall ablation
# ---------------------------------------------------------------------------
ORDER = ["local_only", "dense_causal", "sked_tail0", "sked_tail1"]
TICK = {
    "local_only":   "local only\nno cross-attn\n$W{=}4$",
    "dense_causal": "dense causal\nno cross-attn\n$W{\\geq}L$",
    "sked_tail0":   "SKED\ncross-attn\n$W{=}4$",
    "sked_tail1":   "SKED + dense tail\ncross-attn\n$W{=}4$",
}
LEG = {
    "local_only": "local only",
    "dense_causal": "dense causal",
    "sked_tail0": "SKED (windowed encoder)",
    "sked_tail1": "SKED + dense tail",
}
BAR_C = {
    "local_only": C_ALT,
    "dense_causal": C_BASE,
    "sked_tail0": C_SKED,
    "sked_tail1": "#7fb3d5",
}


def fig_ablation():
    d = load("exp6_ablation.json")
    cfg = d["config"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55))

    x = list(range(len(ORDER)))
    means = []
    for v in ORDER:
        accs = [r["final_acc"] for r in d["runs"] if r["variant"] == v]
        means.append(sum(accs) / len(accs))
    ax.bar(x, means, color=[BAR_C[v] for v in ORDER], width=0.6, alpha=0.9)
    for i, v in enumerate(ORDER):
        accs = [r["final_acc"] for r in d["runs"] if r["variant"] == v]
        ax.plot([i] * len(accs), accs, "o", color="white", ms=3.2,
                markeredgecolor="#222222", markeredgewidth=0.5, zorder=3)
    ax.axhline(cfg["chance"], color=C_BASE, lw=0.8, ls=":", alpha=0.9)
    ax.text(len(ORDER) - 0.5, cfg["chance"] + 0.03, f"chance {cfg['chance']:.3f}",
            ha="right", va="bottom", fontsize=6.2, color=C_BASE)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[v] for v in ORDER], fontsize=6.2)
    ax.set_ylabel("final recall accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) final accuracy, 2 seeds", fontsize=8)

    for v in ORDER:
        runs = [r for r in d["runs"] if r["variant"] == v]
        steps = [p[0] for p in runs[0]["curve"]]
        mean = [sum(r["curve"][i][1] for r in runs) / len(runs)
                for i in range(len(steps))]
        ax2.plot(steps, mean, "-", color=BAR_C[v], lw=1.5, label=LEG[v])
    ax2.axhline(cfg["chance"], color=C_BASE, lw=0.8, ls=":", alpha=0.9)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("recall accuracy")
    ax2.set_ylim(0, 1.08)
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.legend(loc="center right", fontsize=6.2)
    ax2.set_title("(b) learning curves, mean of 2 seeds", fontsize=8)

    fig.tight_layout(pad=0.4)
    out = os.path.join(FIGDIR, "fig_ablation.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_kv_scaling()
    fig_ternary()
    fig_ablation()