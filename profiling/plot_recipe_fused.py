#!/usr/bin/env python3
"""Efficiency figure including the fused dual-window Triton kernel.

Panels (a-c): prefill TPS at L = 16K/64K/256K, x = teacher SWA window
(student = W/2). Series: Baseline / IPTTT / IPTTCD (merged fast path) /
IPTTCD + fused kernel (hatched). Panel (d): fused-kernel gap vs IPTTT.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
WS = [2048, 4096, 8192, 16384]
LS = [16384, 65536, 262144]
LINE_STYLES = {16384: "o", 65536: "s", 262144: "^"}
GAP_COLORS = {16384: "#3945B2", 65536: "#6377DA", 262144: "#96A8FC"}


def med(name, L):
    try:
        with open(f"{R}/{name}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


def fused_med(W, L):
    return med(f"o8_v3_w{W}", L) or med(f"o9_v3_w{W}", L)


def tps(ms, L):
    return L / (ms / 1e3) / 1e3 if ms else None


series = [
    ("Baseline", "#3945B2", None, lambda W, L: tps(med(f"ra_tf_w{W}", L), L)),
    ("IPTTT", "#6377DA", None, lambda W, L: tps(med(f"ra_ipttt_w{W}_s2", L), L)),
    ("IPTTCD", "#96A8FC", None, lambda W, L: tps(med(f"ra2_v9_w{W}_s2", L), L)),
    ("IPTTCD+fused", "#96A8FC", "///", lambda W, L: tps(fused_med(W, L), L)),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.8))

for pi, L in enumerate(LS):
    ax = axes[pi]
    x = np.arange(len(WS))
    width = 0.86 / len(series)
    ymax = 0.0
    for si, (name, color, hatch, fn) in enumerate(series):
        vals = [fn(W, L) or 0.0 for W in WS]
        ymax = max(ymax, *vals)
        pos = x + (si - (len(series) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.93, color=color, label=name, hatch=hatch,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b, v in zip(bars, vals):
            if v:
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 1.5),
                            ha="center", fontsize=5.2, color="#444444", zorder=4)
    ax.set_title(f"({'abc'[pi]}) Prefill TPS @ {L // 1024}K", fontsize=9.5, pad=8)
    ax.set_ylabel("Prefill TPS (K tokens/s)", fontsize=7.5)
    ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
    ax.set_ylim(0, ymax * 1.30)

ax = axes[3]
for L in LS:
    gaps = []
    for W in WS:
        a, b = fused_med(W, L), med(f"ra_ipttt_w{W}_s2", L)
        gaps.append((a / b - 1) * 100 if a and b else np.nan)
    ax.plot(np.arange(len(WS)), gaps, marker=LINE_STYLES[L], linestyle="-", markersize=4.5,
            linewidth=1.4, label=f"L={L // 1024}K", zorder=3, color=GAP_COLORS[L])
ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
ax.set_title("(d) IPTTCD+fused overhead vs IPTTT", fontsize=9.5, pad=8)
ax.set_ylabel("prefill time gap (%)", fontsize=7.5)
ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7)
ax.set_xticks(np.arange(len(WS)))
ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
ax.set_ylim(-2, 16)

for ax in axes:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=5.8, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.3, handlelength=1.1,
              labelspacing=0.22, loc="upper right")

fig.tight_layout(w_pad=1.6)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_efficiency_fused.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_efficiency_fused.png/.pdf")

print("\nfused gap vs IPTTT:")
for L in LS:
    row = []
    for W in WS:
        a, b = fused_med(W, L), med(f"ra_ipttt_w{W}_s2", L)
        row.append(f"w{W // 1024}k={((a / b - 1) * 100):+.1f}%" if a and b else "-")
    print(f"  L={L // 1024:3d}K: " + "  ".join(row))
