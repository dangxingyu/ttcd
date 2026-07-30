#!/usr/bin/env python3
"""Batch-size efficiency figure: throughput (total tokens/s) vs batch size at
L=64K and L=16K for Baseline / IPTTT / IPTTCD (merged) / IPTTCD+fused, plus
gap-vs-batch panel."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
SERIES = [("Baseline", "tf", "#3945B2", None),
          ("IPTTT", "ipttt", "#6377DA", None),
          ("IPTTCD", "v9", "#96A8FC", None),
          ("IPTTCD+fused", "v9f", "#96A8FC", "///")]
CASES = [("64K", "L64k", 65536, [1, 2, 4]), ("16K", "L16k", 16384, [1, 4, 8])]


def med(key, bs, ltag, L):
    try:
        with open(f"{R}/bsz_{key}_b{bs}_{ltag}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.8))

for pi, (lname, ltag, L, BS) in enumerate(CASES):
    ax = axes[pi]
    x = np.arange(len(BS))
    width = 0.86 / len(SERIES)
    ymax = 0.0
    for si, (name, key, color, hatch) in enumerate(SERIES):
        vals = []
        for bs in BS:
            ms = med(key, bs, ltag, L)
            vals.append(bs * L / (ms / 1e3) / 1e3 if ms else 0.0)  # K tok/s
        ymax = max(ymax, *vals)
        pos = x + (si - (len(SERIES) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.93, color=color, label=name, hatch=hatch,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b_, v in zip(bars, vals):
            if v:
                ax.annotate(f"{v:.0f}", (b_.get_x() + b_.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 1.5),
                            ha="center", fontsize=5.2, color="#444444", zorder=4)
    ax.set_title(f"({'ab'[pi]}) Prefill throughput @ {lname}", fontsize=9.5, pad=8)
    ax.set_ylabel("total TPS (K tokens/s)", fontsize=7.5)
    ax.set_xlabel("batch size", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BS], fontsize=7.5)
    ax.set_ylim(0, ymax * 1.32)

ax = axes[2]
STYLES = {("v9", "64K"): ("-", "o", "#96A8FC"), ("v9f", "64K"): ("-", "s", "#3945B2"),
          ("v9", "16K"): ("--", "o", "#96A8FC"), ("v9f", "16K"): ("--", "s", "#3945B2")}
for lname, ltag, L, BS in CASES:
    for key, lab in [("v9", "IPTTCD"), ("v9f", "+fused")]:
        gaps = []
        for bs in BS:
            a, b_ = med(key, bs, ltag, L), med("ipttt", bs, ltag, L)
            gaps.append((a / b_ - 1) * 100 if a and b_ else np.nan)
        ls, mk, color = STYLES[(key, lname)]
        ax.plot(np.arange(len(BS)), gaps, linestyle=ls, marker=mk, markersize=4.2,
                linewidth=1.3, color=color, label=f"{lab} @{lname}", zorder=3)
ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
ax.set_title("(c) gap vs IPTTT over batch size", fontsize=9.5, pad=8)
ax.set_ylabel("prefill time gap (%)", fontsize=7.5)
ax.set_xlabel("batch index (see panel ticks)", fontsize=7.5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["b1", "b2/b4", "b4/b8"], fontsize=7)
ax.set_ylim(-2, 16)

for ax in axes:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=5.6, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.3, handlelength=1.2,
              labelspacing=0.22, loc="upper left")

fig.tight_layout(w_pad=1.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_bsz_efficiency.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_bsz_efficiency.png/.pdf")

print("\ngap table (v9 / v9+fused vs ipttt):")
for lname, ltag, L, BS in CASES:
    for bs in BS:
        i = med("ipttt", bs, ltag, L)
        row = []
        for key in ("tf", "ipttt", "v9", "v9f"):
            ms = med(key, bs, ltag, L)
            row.append(f"{key}={ms:.0f}ms" if ms else f"{key}=-")
        g9 = med("v9", bs, ltag, L); gf = med("v9f", bs, ltag, L)
        gaps = f"  gap v9 {((g9/i-1)*100):+.1f}%  fused {((gf/i-1)*100):+.1f}%" if i and g9 and gf else ""
        print(f"  {lname} b{bs}: " + "  ".join(row) + gaps)
