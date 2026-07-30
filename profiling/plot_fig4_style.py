#!/usr/bin/env python3
"""Figure-4-style efficiency plot (In-Place TTT paper, arXiv 2604.06169):
one row, four panels of grouped bars —
(a) prefill throughput SWA, (b) throughput full-attn, (c) peak memory SWA,
(d) peak memory full-attn — comparing transformer baseline / IPTTT / IPTTCDv9
from the profiling/results benchmark JSONs (eval_mask_cache mode, H100).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "profiling/results"
MODE = "eval_mask_cache"
# validated 3-shade categorical ladder (dataviz six-checks pass, light surface)
COLORS = {"Baseline": "#3945B2", "IPTTT": "#6377DA", "IPTTCD": "#96A8FC"}
SERIES = ["Baseline", "IPTTT", "IPTTCD"]


def load(path):
    with open(path) as f:
        return json.load(f)["runs"]


def get(runs, L, field):
    r = runs.get(f"L{L}/{MODE}", {})
    return r.get(field)


# --- gather data ------------------------------------------------------------
swa = {
    "Baseline": {**load(f"{R}/seq_tf_swa8k.json"), **load(f"{R}/r2_tf_long.json")},
    "IPTTT": {**load(f"{R}/seq_ipttt_swa8k.json"), **load(f"{R}/r2_ipttt_long.json")},
    "IPTTCD": {**load(f"{R}/seq_v9_swa8k.json"), **load(f"{R}/r2_v9_long.json")},
}
full = {
    "Baseline": load(f"{R}/r2_tf_fa.json"),
    "IPTTT": load(f"{R}/r2_ipttt_fa.json"),
    "IPTTCD": load(f"{R}/r2_v9_fa.json"),
}
SWA_LS = [16384, 65536, 262144]
FULL_LS = [16384, 65536]
lab = lambda L: f"{L // 1024}k"

def tps(runs, L):
    ms = get(runs, L, "ms_median")
    return L / (ms / 1e3) / 1e3 if ms else None  # K tokens/s

def mem(runs, L):
    return get(runs, L, "peak_mem_gb")

panels = [
    ("(a) Throughput (SWA)", swa, SWA_LS, tps, "Prefill TPS (K tokens/s)"),
    ("(b) Throughput (Full)", full, FULL_LS, tps, "Prefill TPS (K tokens/s)"),
    ("(c) Memory (SWA)", swa, SWA_LS, mem, "Peak Memory (GB)"),
    ("(d) Memory (Full)", full, FULL_LS, mem, "Peak Memory (GB)"),
]

# --- draw -------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.7))

for ax, (title, data, Ls, fn, ylab) in zip(axes, panels):
    n_g, n_s = len(Ls), len(SERIES)
    width = 0.82 / n_s
    x = np.arange(n_g)
    ymax = 0.0
    for si, s in enumerate(SERIES):
        vals = [fn(data[s], L) or 0.0 for L in Ls]
        ymax = max(ymax, *vals)
        pos = x + (si - (n_s - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.94, color=COLORS[s], label=s,
                      edgecolor="#222222", linewidth=0.5, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=5.8, color="#444444", zorder=4)
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab(L) for L in Ls], fontsize=8)
    ax.set_ylim(0, ymax * 1.32)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=6.2, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.35, handlelength=1.1, handleheight=0.9,
              labelspacing=0.25, loc="upper left")

fig.suptitle("")
fig.tight_layout(w_pad=1.6)
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/fig4_style_efficiency.{ext}", dpi=220, bbox_inches="tight")
print(f"saved {R}/fig4_style_efficiency.png / .pdf")

# console table for the caption numbers
for title, data, Ls, fn, _ in panels:
    print(f"\n{title}")
    for s in SERIES:
        print(f"  {s:9s} " + "  ".join(f"{lab(L)}={fn(data[s], L):.1f}" for L in Ls if fn(data[s], L)))
