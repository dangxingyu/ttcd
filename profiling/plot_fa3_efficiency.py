#!/usr/bin/env python3
"""FA3-world efficiency figures: (1) 3 panels (batch 1/2/4), x = SWA window,
bars = Baseline/IPTTT/IPTTCD, all with FlashAttention-3 installed;
(2) gap line chart per batch. L=64K, same node."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
L = 65536
WS = [2048, 4096, 8192, 16384]
BS = [1, 2, 4]
SERIES = [("Baseline (FA3)", "tf", "#3945B2", None),
          ("IPTTT (FA3)", "ipttt", "#6377DA", None),
          ("IPTTCD (FA3)", "v9fc", "#96A8FC", None)]


def med(key, bs, W):
    try:
        with open(f"{R}/f3_{key}_b{bs}_w{W}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})

# ---- figure 1: 3-panel bars ----
fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.9))
for pi, bs in enumerate(BS):
    ax = axes[pi]
    x = np.arange(len(WS))
    width = 0.82 / len(SERIES)
    ymax = 0.0
    for si, (name, key, color, hatch) in enumerate(SERIES):
        vals = []
        for W in WS:
            ms = med(key, bs, W)
            vals.append(bs * L / (ms / 1e3) / 1e3 if ms else 0.0)
        ymax = max(ymax, *vals)
        pos = x + (si - (len(SERIES) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.94, color=color, label=name, hatch=hatch,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b_, v in zip(bars, vals):
            if v:
                ax.annotate(f"{v:.0f}", (b_.get_x() + b_.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 1.5),
                            ha="center", fontsize=5.4, color="#444444", zorder=4)
    ax.set_title(f"({'abc'[pi]}) Prefill throughput @ 64K, batch = {bs} (FA3)", fontsize=9.3, pad=8)
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
    ax.legend(fontsize=6.0, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.32, handlelength=1.2,
              labelspacing=0.24, loc="upper right")
fig.tight_layout(w_pad=1.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_fa3_window_bsz.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_fa3_window_bsz.png/.pdf")

# ---- figure 2: gap lines ----
fig2, ax = plt.subplots(figsize=(4.6, 3.0))
x = np.arange(len(WS))
for bs, marker, color in [(1, "o", "#3945B2"), (2, "s", "#6377DA"), (4, "^", "#96A8FC")]:
    gaps = []
    for W in WS:
        a, b_ = med("v9fc", bs, W), med("ipttt", bs, W)
        gaps.append((a / b_ - 1) * 100 if a and b_ else np.nan)
    ax.plot(x, gaps, marker=marker, markersize=4.5, linewidth=1.4,
            color=color, label=f"batch = {bs}", zorder=3)
ax.axhline(0, color="#999999", linewidth=0.9, zorder=1)
ax.set_title("IPTTCD overhead vs IPTTT @ 64K (FA3)", fontsize=9.5, pad=8)
ax.set_ylabel("prefill time gap (%)", fontsize=7.5)
ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=2, labelsize=7.5)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(fontsize=6.6, frameon=True, edgecolor="#bbbbbb", fancybox=False,
          framealpha=1.0, borderpad=0.35, handlelength=1.3,
          labelspacing=0.3, loc="upper right")
fig2.tight_layout()
for ext in ("png", "pdf"):
    fig2.savefig(f"{R}/fig_fa3_gap_lines.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_fa3_gap_lines.png/.pdf")

print("\nFA3-world gap (IPTTCD vs IPTTT):")
for bs in BS:
    row = []
    for W in WS:
        a, b_ = med("v9fc", bs, W), med("ipttt", bs, W)
        row.append(f"w{W // 1024}k={((a / b_ - 1) * 100):+.1f}%" if a and b_ else f"w{W // 1024}k=-")
    print(f"  b={bs}: " + "  ".join(row))
