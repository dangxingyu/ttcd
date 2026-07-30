#!/usr/bin/env python3
"""Per-token KL(p_base ‖ p_fused) and TV-distance distributions from the
fused-kernel A/B on the real checkpoint (books docs @64K, all shards).
Log-x histograms + CDFs with quantile markers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#222222",
})
fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.4))

QS = [("p95", 95, "#3945B2"), ("p99", 99, "#6377DA"), ("p99.9", 99.9, "#b91c1c")]

import glob as _glob

def _load_all(prefix):
    parts = sorted(_glob.glob(f"profiling/results/{prefix}_per_token*.pt"))
    return torch.cat([torch.load(p) for p in parts])

for row, (name, prefix, unit) in enumerate([
    ("KL(base ‖ fused)", "kl", "nats"),
    ("TV distance", "tv", ""),
]):
    x = _load_all(prefix).double().numpy()
    x = np.clip(x, 1e-8, None)
    qv = [(qn, np.percentile(x, p), color) for qn, p, color in QS]
    mx = x.max()

    ax = axes[row][0]
    bins = np.logspace(np.log10(x.min()), np.log10(mx * 1.05), 90)
    ax.hist(x, bins=bins, color="#6377DA", edgecolor="#222222", linewidth=0.25, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"({'ac'[row]}) per-token {name} — histogram", fontsize=9.5, pad=8)
    ax.set_xlabel(f"{name} {unit}".strip(), fontsize=7.5)
    ax.set_ylabel("token count (log)", fontsize=7.5)
    for qn, v, color in qv:
        ax.axvline(v, linestyle="--", linewidth=1.0, color=color, alpha=0.9, zorder=4,
                   label=f"{qn} = {v:.1e}")
    ax.legend(fontsize=6.4, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.35, handlelength=1.3,
              labelspacing=0.3, loc="upper left")

    ax = axes[row][1]
    srt = np.sort(x)
    cdf = np.arange(1, len(srt) + 1) / len(srt)
    ax.plot(srt, cdf, color="#3945B2", linewidth=1.6, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.04)
    ax.set_title(f"({'bd'[row]}) {name} — CDF", fontsize=9.5, pad=8)
    ax.set_xlabel(f"{name} {unit}".strip(), fontsize=7.5)
    ax.set_ylabel("fraction of tokens", fontsize=7.5)
    for qn, v, color in qv:
        ax.axvline(v, linestyle="--", linewidth=1.0, color=color, alpha=0.9, zorder=2,
                   label=f"{qn} = {v:.1e}")
    ax.plot([], [], " ", label=f"max = {mx:.1e}")
    ax.legend(fontsize=6.4, frameon=True, edgecolor="#bbbbbb", fancybox=False,
              framealpha=1.0, borderpad=0.35, handlelength=1.3,
              labelspacing=0.3, loc="upper left")

for ax in axes.flat:
    ax.grid(axis="both", linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.tight_layout(w_pad=2.0, h_pad=2.2)
for ext in ("png", "pdf"):
    fig.savefig(f"profiling/results/fig_kl_dist.{ext}", dpi=220, bbox_inches="tight")
print("saved profiling/results/fig_kl_dist.png/.pdf")
