#!/usr/bin/env python3
"""Averaged FA3-baseline 3-case figure: mean over 2 independent repeats, all
three models measured back-to-back on the same GPU per cell."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
WS = [2048, 4096, 8192, 16384]
CASES = [(1, 65536, "64K, batch=1"), (4, 65536, "64K, batch=4"), (1, 262144, "256K, batch=1")]
SERIES = [("Baseline (FA3)", "tf", "#3945B2"),
          ("IPTTT (FA3)", "ipttt", "#6377DA"),
          ("IPTTCD (FA3+fused)", "v9", "#96A8FC")]
REPS = [1, 2]


def med(key, bs, L, W, rep):
    try:
        with open(f"{R}/g4_{key}_b{bs}_L{L}_w{W}_r{rep}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


def mean_ms(key, bs, L, W):
    vals = [med(key, bs, L, W, r) for r in REPS]
    vals = [v for v in vals if v]
    return (sum(vals) / len(vals), vals) if vals else (None, [])


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.9))

for pi, (bs, L, label) in enumerate(CASES):
    ax = axes[pi]
    x = np.arange(len(WS))
    width = 0.82 / len(SERIES)
    ymax = 0.0
    for si, (name, key, color) in enumerate(SERIES):
        vals = []
        for W in WS:
            m, _ = mean_ms(key, bs, L, W)
            vals.append(bs * L / (m / 1e3) / 1e3 if m else 0.0)
        ymax = max(ymax, *vals)
        pos = x + (si - (len(SERIES) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.94, color=color, label=name,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b_, v in zip(bars, vals):
            if v:
                ax.annotate(f"{v:.0f}", (b_.get_x() + b_.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 1.5),
                            ha="center", fontsize=5.4, color="#444444", zorder=4)
    ax.set_title(f"({'abc'[pi]}) Prefill throughput @ {label}", fontsize=9.3, pad=8)
    ax.set_ylabel("total TPS (K tokens/s)", fontsize=7.5)
    ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
    ax.set_ylim(0, ymax * 1.30)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=5.8, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.32, handlelength=1.2,
              labelspacing=0.24, loc="upper right")

fig.tight_layout(w_pad=1.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_g4_fa3_avg.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_g4_fa3_avg.png/.pdf")

print("\ngap IPTTCD(FA3+fused) vs IPTTT(FA3), mean over reps (per-rep gaps in brackets):")
for bs, L, label in CASES:
    row = []
    for W in WS:
        a, av = mean_ms("v9", bs, L, W)
        b_, bv = mean_ms("ipttt", bs, L, W)
        if a and b_:
            g = (a / b_ - 1) * 100
            per = [f"{(x / y - 1) * 100:+.1f}" for x, y in zip(av, bv)]
            row.append(f"w{W // 1024}k={g:+.1f}%[{','.join(per)}]")
        else:
            row.append(f"w{W // 1024}k=-")
    print(f"  {label}: " + "  ".join(row))
