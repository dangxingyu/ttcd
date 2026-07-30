#!/usr/bin/env python3
"""Final post-optimization figure: 3 panels (batch = 1 / 2 / 4), x-axis =
teacher SWA window (student = W/2), bars = Baseline / IPTTT / IPTTCD (final
fused stack). L = 64K, same node, eval_nomask_cache."""
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
SERIES = [("Baseline", "tf", "#3945B2", None),
          ("IPTTT", "ipttt", "#6377DA", None),
          ("IPTTCD (final)", "v9fc", "#96A8FC", None)]


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
    ax.set_title(f"({'abc'[pi]}) Prefill throughput @ 64K, batch = {bs}", fontsize=9.5, pad=8)
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
    fig.savefig(f"{R}/fig_final_window_bsz.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_final_window_bsz.png/.pdf")

print("\ngap of IPTTCD(final) vs IPTTT:")
for bs in BS:
    row = []
    for W in WS:
        a, b_ = med("v9fc", bs, W), med("ipttt", bs, W)
        row.append(f"w{W // 1024}k={((a / b_ - 1) * 100):+.1f}%" if a and b_ else f"w{W // 1024}k=-")
    print(f"  b={bs}: " + "  ".join(row))
