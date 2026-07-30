#!/usr/bin/env python3
"""Plot NIAH accuracy vs context length (6 tasks × multiple models).

Reads eval_results/niah_<model>.json produced by scripts/eval_niah.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


TASKS = [
    ("niah_single_1", "S-NIAH-1 (number, noise haystack)"),
    ("niah_single_2", "S-NIAH-2 (number, essay haystack)"),
    ("niah_single_3", "S-NIAH-3 (UUID, essay haystack)"),
    ("niah_multikey_1", "Multi-key-1 (4 keys, essay)"),
    ("niah_multikey_2", "Multi-key-2 (8 keys, distractor haystack)"),
    ("niah_multikey_3", "Multi-key-3 (26 UUID keys, distractor haystack)"),
    ("niah_multivalue", "Multi-value (1 key, 4 values)"),
    ("niah_multiquery", "Multi-query (4 keys, 4 queries)"),
]


def load_results(model_tags):
    """model_tags: list of strings matching eval_results/niah_<tag>.json."""
    out = {}
    for tag in model_tags:
        path = f"eval_results/niah_{tag}.json"
        if not os.path.exists(path):
            print(f"  [skip] {path} missing")
            continue
        with open(path) as f:
            out[tag] = json.load(f).get("per_task", {})
    return out


def plot_panel(models_results, model_color_map, output_path: str,
               ctx_lens=(4096, 8192, 16384, 32768, 65536),
               title_suffix=""):
    # 8 tasks → 2×4 grid
    fig, axes = plt.subplots(2, 4, figsize=(28, 12), dpi=150, sharey=True)
    axes = axes.flatten()

    for ax, (task_key, task_label) in zip(axes, TASKS):
        for tag, results in models_results.items():
            accs = []
            ctxs_used = []
            for c in ctx_lens:
                k = f"{task_key}_ctx{c}"
                if k in results:
                    accs.append(results[k]["accuracy"])
                    ctxs_used.append(c)
            if not accs:
                continue
            color = model_color_map.get(tag, None)
            ax.plot(ctxs_used, accs, marker="o", markersize=7, linewidth=2.2,
                    color=color, label=tag, alpha=0.9)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(list(ctx_lens)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$2^{{{int(round(np.log2(x)))}}}$"))
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Context length", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(task_label, fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
        ax.tick_params(labelsize=11)

    title = f"RULER NIAH accuracy vs context length  {title_suffix}".strip()
    fig.suptitle(title, fontsize=16, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "qwen3_base", "qwen3_cpt", "qwen3_swa_base", "qwen3_swa_ct", "qwen3_ipttcdv9",
        "qwen3_base_yarn",
        "smollm2_base", "smollm2_cpt", "smollm2_swa_base", "smollm2_swa_ct", "smollm2_ipttcdv9",
        "smollm2_base_yarn",
    ])
    ap.add_argument("--output", default="eval_results/niah_accuracy_all.png")
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    # Color per model family
    color_map = {
        "qwen3_base":       "#d62728",
        "qwen3_cpt":        "#ff7f0e",
        "qwen3_swa_base":   "#8c564b",
        "qwen3_swa_ct":     "#9467bd",
        "qwen3_ipttcdv9":   "#1f77b4",
        "qwen3_base_yarn":  "#e377c2",
        "smollm2_base":       "#c43d5a",
        "smollm2_cpt":        "#ff9a3c",
        "smollm2_swa_base":   "#8b6f47",
        "smollm2_swa_ct":     "#5c4689",
        "smollm2_ipttcdv9":   "#0d5e9c",
        "smollm2_base_yarn":  "#d97ca0",
    }

    results = load_results(args.models)
    if not results:
        print("No results to plot."); return

    plot_panel(results, color_map, args.output, title_suffix=args.title_suffix)

    # Also plot each family separately for readability
    qwen3_only = {k: v for k, v in results.items() if k.startswith("qwen3_")}
    smol_only = {k: v for k, v in results.items() if k.startswith("smollm2_")}
    if qwen3_only:
        plot_panel(qwen3_only, color_map, args.output.replace(".png", "_qwen3.png"),
                   title_suffix="(Qwen3-0.6B)")
    if smol_only:
        plot_panel(smol_only, color_map, args.output.replace(".png", "_smollm2.png"),
                   title_suffix="(SmolLM2-360M)")


if __name__ == "__main__":
    main()
