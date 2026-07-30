#!/usr/bin/env python3
"""Recipe-aligned profiling figure (student window = teacher window / 2).

Panels (a-c): prefill TPS grouped bars at L = 16K / 64K / 256K,
x = teacher SWA window, series = Baseline / IPTTT / IPTTCD (v2-style
parameterization: vis=1, chunk=W/2).
Panel (d): relative gap (IPTTCD vs IPTTT) as lines over window, one line
per sequence length.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
COLORS = {"Baseline": "#3945B2", "IPTTT": "#6377DA", "IPTTCD (opt)": "#96A8FC"}
WS = [2048, 4096, 8192, 16384]
LS = [16384, 65536, 262144]
LINE_STYLES = {16384: ("o", "-"), 65536: ("s", "-"), 262144: ("^", "-")}


def med(name, L):
    try:
        with open(f"{R}/{name}.json") as f:
            return json.load(f)["runs"].get(f"L{L}/eval_nomask_cache", {}).get("ms_median")
    except FileNotFoundError:
        return None


def tps(name, L):
    ms = med(name, L)
    return L / (ms / 1e3) / 1e3 if ms else None


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.8))

series = [("Baseline", lambda W: f"ra_tf_w{W}"),
          ("IPTTT", lambda W: f"ra_ipttt_w{W}_s2"),
          ("IPTTCD (opt)", lambda W: f"ra2_v9_w{W}_s2")]

for pi, L in enumerate(LS):
    ax = axes[pi]
    x = np.arange(len(WS))
    width = 0.82 / len(series)
    ymax = 0.0
    for si, (name, key_fn) in enumerate(series):
        vals = [tps(key_fn(W), L) or 0.0 for W in WS]
        ymax = max(ymax, *vals)
        pos = x + (si - (len(series) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.94, color=COLORS[name], label=name,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b, v in zip(bars, vals):
            if v:
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 1.5),
                            ha="center", fontsize=5.6, color="#444444", zorder=4)
    ax.set_title(f"({'abc'[pi]}) Prefill TPS @ {L // 1024}K", fontsize=9.5, pad=8)
    ax.set_ylabel("Prefill TPS (K tokens/s)", fontsize=7.5)
    ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
    ax.set_ylim(0, ymax * 1.30)

ax = axes[3]
GAP_COLORS = {16384: "#3945B2", 65536: "#6377DA", 262144: "#96A8FC"}
for L in LS:
    gaps_post = []
    for W in WS:
        b = med(f"ra_ipttt_w{W}_s2", L)
        a_post = med(f"ra2_v9_w{W}_s2", L)
        gaps_post.append((a_post / b - 1) * 100 if a_post and b else np.nan)
    marker, _ = LINE_STYLES[L]
    ax.plot(np.arange(len(WS)), gaps_post, marker=marker, linestyle="-", markersize=4.5,
            linewidth=1.4, label=f"L={L // 1024}K", zorder=3, color=GAP_COLORS[L])
ax.set_title("(d) IPTTCD overhead vs IPTTT", fontsize=9.5, pad=8)
ax.set_ylabel("prefill time gap (%)", fontsize=7.5)
ax.set_xlabel("teacher SWA window (student = W/2)", fontsize=7)
ax.set_xticks(np.arange(len(WS)))
ax.set_xticklabels([f"{W // 1024}k" for W in WS], fontsize=7.5)
ax.set_ylim(0, 24)

for ax in axes:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=6.2, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.35, handlelength=1.2,
              labelspacing=0.25, loc="upper right")

fig.tight_layout(w_pad=1.6)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_recipe_aligned.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_recipe_aligned.png / .pdf")
