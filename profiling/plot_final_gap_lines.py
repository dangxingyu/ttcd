#!/usr/bin/env python3
"""Final gap line chart: IPTTCD (final fused stack) vs IPTTT prefill time gap
over teacher SWA window, one line per batch size. L=64K, FA2 stack, same node."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
L = 65536
WS = [2048, 4096, 8192, 16384]
BS = [(1, "o", "#3945B2"), (2, "s", "#6377DA"), (4, "^", "#96A8FC")]


def med(key, bs, W):
    try:
        with open(f"{R}/fw_{key}_b{bs}_w{W}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, ax = plt.subplots(figsize=(4.6, 3.0))

x = np.arange(len(WS))
for bs, marker, color in BS:
    gaps = []
    for W in WS:
        a, b_ = med("v9fc", bs, W), med("ipttt", bs, W)
        gaps.append((a / b_ - 1) * 100 if a and b_ else np.nan)
    ax.plot(x, gaps, marker=marker, markersize=4.5, linewidth=1.4,
            color=color, label=f"batch = {bs}", zorder=3)

ax.axhline(0, color="#999999", linewidth=0.9, zorder=1)
ax.set_title("IPTTCD (final) overhead vs IPTTT @ 64K", fontsize=9.5, pad=8)
ax.set_ylabel("prefill time gap (%)", fontsize=7.5)
ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
ax.set_ylim(-4, 4)
ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=2, labelsize=7.5)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(fontsize=6.6, frameon=True, edgecolor="#bbbbbb", fancybox=False,
          framealpha=1.0, borderpad=0.35, handlelength=1.3,
          labelspacing=0.3, loc="upper right")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_final_gap_lines.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_final_gap_lines.png/.pdf")
