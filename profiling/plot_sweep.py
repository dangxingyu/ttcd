#!/usr/bin/env python3
"""Figure-3(b)-style ablation plot for the chunk/window sweeps @64K prefill.

(a) ttt_chunk sweep (teacher window fixed 8192): IPTTT vs IPTTCD vs
    IPTTCD+patch bars per chunk; transformer baseline as dashed reference line.
(b) SWA window sweep (ttt_chunk fixed 4096): Baseline/IPTTT/IPTTCD bars per
    window incl. full attention.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
MODE = "eval_nomask_cache"
L = 65536
COLORS = {"Baseline": "#3945B2", "IPTTT": "#6377DA", "IPTTCD": "#96A8FC"}


def med(path):
    try:
        with open(path) as f:
            r = json.load(f)["runs"].get(f"L{L}/{MODE}", {})
        return r.get("ms_median")
    except FileNotFoundError:
        return None


def tps(ms):
    return L / (ms / 1e3) / 1e3 if ms else None


plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9))

# ---- (a) chunk sweep -------------------------------------------------------
CHUNKS = [256, 512, 1024, 2048, 4096, 8192]
series_a = [("IPTTT", "ipttt", COLORS["IPTTT"], None),
            ("IPTTCD", "v9", COLORS["IPTTCD"], None),
            ("IPTTCD+patch", "v9p", COLORS["IPTTCD"], "///")]
ax = axes[0]
x = np.arange(len(CHUNKS))
width = 0.82 / len(series_a)
ymax = 0.0
for si, (name, key, color, hatch) in enumerate(series_a):
    vals = [tps(med(f"{R}/swc_{key}_c{c}.json")) or 0.0 for c in CHUNKS]
    ymax = max(ymax, *vals)
    pos = x + (si - (len(series_a) - 1) / 2) * width
    bars = ax.bar(pos, vals, width * 0.94, color=color, label=name, hatch=hatch,
                  edgecolor="#222222", linewidth=0.5, zorder=3)
    for b, v in zip(bars, vals):
        if v:
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=5.6, color="#444444", zorder=4)
tf_ref = tps(med(f"{R}/swc_tf_ref.json"))
if tf_ref:
    ax.axhline(tf_ref, color=COLORS["Baseline"], linestyle="--", linewidth=1.1, zorder=2)
    ax.annotate(f"Baseline {tf_ref:.0f}", (-0.45, tf_ref),
                xytext=(0, 3), textcoords="offset points", ha="left",
                fontsize=6.5, color=COLORS["Baseline"])
    ymax = max(ymax, tf_ref)
ax.set_title("(a) ttt_chunk sweep (SWA window 8192)", fontsize=9.5, pad=8)
ax.set_ylabel("Prefill TPS (K tokens/s)", fontsize=7.5)
ax.set_xlabel("ttt_chunk (= student window)", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([str(c) for c in CHUNKS], fontsize=7.5)
ax.set_ylim(0, ymax * 1.30)

# ---- (b) window sweep ------------------------------------------------------
WINDOWS = [2048, 4096, 8192, 16384, "none"]
wlab = {"none": "full"}
series_b = [("Baseline", "tf"), ("IPTTT", "ipttt"), ("IPTTCD", "v9")]
ax = axes[1]
x = np.arange(len(WINDOWS))
width = 0.82 / len(series_b)
ymax = 0.0
for si, (name, key) in enumerate(series_b):
    vals = [tps(med(f"{R}/sww_{key}_w{w}.json")) or 0.0 for w in WINDOWS]
    ymax = max(ymax, *vals)
    pos = x + (si - (len(series_b) - 1) / 2) * width
    bars = ax.bar(pos, vals, width * 0.94, color=COLORS[name], label=name,
                  edgecolor="#222222", linewidth=0.5, zorder=3)
    for b, v in zip(bars, vals):
        if v:
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=5.6, color="#444444", zorder=4)
ax.set_title("(b) SWA window sweep (ttt_chunk 4096)", fontsize=9.5, pad=8)
ax.set_ylabel("Prefill TPS (K tokens/s)", fontsize=7.5)
ax.set_xlabel("teacher SWA window", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([wlab.get(w, str(w)) for w in WINDOWS], fontsize=7.5)
ax.set_ylim(0, ymax * 1.30)

for ax in axes:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=6.2, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.35, handlelength=1.1,
              labelspacing=0.25, loc="upper right")

fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig_sweep_ablation.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig_sweep_ablation.png / .pdf")

# gap table
print("\nchunk sweep @64K (v9/ipttt gap):")
for c in CHUNKS:
    a, b = med(f"{R}/swc_v9_c{c}.json"), med(f"{R}/swc_ipttt_c{c}.json")
    p = med(f"{R}/swc_v9p_c{c}.json")
    if a and b:
        extra = f"  patch {p:.0f}ms" if p else ""
        print(f"  chunk={c:5d}: v9 {a:6.0f}ms  ipttt {b:6.0f}ms  gap +{(a / b - 1) * 100:5.1f}%{extra}")
print("\nwindow sweep @64K (v9/ipttt gap):")
for w in WINDOWS:
    a, b = med(f"{R}/sww_v9_w{w}.json"), med(f"{R}/sww_ipttt_w{w}.json")
    t = med(f"{R}/sww_tf_w{w}.json")
    if a and b:
        tf_s = f"  tf {t:.0f}ms" if t else ""
        print(f"  window={str(w):>6s}: v9 {a:6.0f}ms  ipttt {b:6.0f}ms  gap +{(a / b - 1) * 100:5.1f}%{tf_s}")
