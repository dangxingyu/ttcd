#!/usr/bin/env python3
"""Averaged 3-case figure for either kernel world.
Usage: plot_gx.py g5 FA3   (FA3 baseline, v9 = FA3-LSE fused + Triton conv)
       plot_gx.py g6 FA2   (FA2 baseline, v9 = Triton fused + Triton conv)
Aggregates all repeats found (g?_*_r*.json)."""
from __future__ import annotations

import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "g5"
WORLD = sys.argv[2] if len(sys.argv) > 2 else "FA3"
R = "profiling/results"
WS = [2048, 4096, 8192, 16384]
CASES = [(1, 65536, "64K, batch=1"), (4, 65536, "64K, batch=4"), (1, 262144, "256K, batch=1")]
SERIES = [(f"Baseline ({WORLD})", "tf", "#3945B2"),
          (f"IPTTT ({WORLD})", "ipttt", "#6377DA"),
          (f"IPTTCD ({WORLD}+fused)", "v9", "#96A8FC")]


def reps(key, bs, L, W):
    vals = []
    for p in sorted(glob.glob(f"{R}/{PREFIX}_{key}_b{bs}_L{L}_w{W}_r*.json")):
        try:
            v = json.load(open(p))["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
            if v:
                vals.append(v)
        except Exception:
            pass
    return vals


def mean_ms(key, bs, L, W):
    v = reps(key, bs, L, W)
    return (sum(v) / len(v), v) if v else (None, [])


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
out = f"{R}/fig_{PREFIX}_{WORLD.lower()}_avg"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {out}.png/.pdf")

print(f"\n[{WORLD}] gap IPTTCD vs IPTTT, mean over reps (n reps):")
for bs, L, label in CASES:
    row = []
    for W in WS:
        a, av = mean_ms("v9", bs, L, W)
        b_, bv = mean_ms("ipttt", bs, L, W)
        if a and b_:
            row.append(f"w{W // 1024}k={((a / b_ - 1) * 100):+.1f}%(n={min(len(av), len(bv))})")
        else:
            row.append(f"w{W // 1024}k=-")
    print(f"  {label}: " + "  ".join(row))
