#!/usr/bin/env python3
"""Merge per-model fixed-final-block eval outputs into one combined result."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple per-model fixed-final-block eval outputs."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Per-model eval directories or fixed_final_block_doc_ppl.json files.",
    )
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument(
        "--ymin_margin",
        type=float,
        default=None,
        help="If set, use (global_min_y - ymin_margin) as the lower plot bound.",
    )
    return parser.parse_args()


def load_eval_result(path_str: str) -> Tuple[Path, Dict, OrderedDict[str, List[Dict]]]:
    path = Path(path_str)
    if path.is_dir():
        path = path / "fixed_final_block_doc_ppl.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval result not found: {path}")

    payload = json.loads(path.read_text())
    results = payload.get("results", {})
    if len(results) < 1:
        raise ValueError(f"Expected at least one model result in {path}, got {len(results)}")
    return path, payload, OrderedDict(results)


def validate_compatible(ref_path: Path, ref_payload: Dict, cur_path: Path, cur_payload: Dict):
    ref_meta = ref_payload["meta"]
    cur_meta = cur_payload["meta"]
    for key in (
        "seq_len",
        "final_block_size",
        "context_lens",
        "selected_total_tokens",
        "min_doc_tokens",
        "num_documents",
    ):
        if ref_meta.get(key) != cur_meta.get(key):
            raise ValueError(
                f"Incompatible meta[{key}] between {ref_path} and {cur_path}: "
                f"{ref_meta.get(key)} != {cur_meta.get(key)}"
            )
    if ref_payload["document_stats"] != cur_payload["document_stats"]:
        raise ValueError(f"document_stats mismatch between {ref_path} and {cur_path}")


def plot_curve(
    all_results: Dict[str, List[Dict]],
    out_dir: str,
    ymax: float | None = None,
    ymin_margin: float | None = None,
):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    all_ys: List[float] = []
    for i, (name, rows) in enumerate(all_results.items()):
        valid = [r for r in rows if r["perplexity"] is not None]
        xs = [r["context_len"] for r in valid]
        ys = [r["perplexity"] for r in valid]
        all_ys.extend(ys)
        ax.plot(xs, ys, marker="o", lw=1.8, color=colors[i % len(colors)], label=name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prefix Context Length")
    ax.set_ylabel("Perplexity on Fixed Final Block")
    ax.set_title("Fixed Final Block PPL vs Extended Context (Single Document)")
    ymin = None
    if ymin_margin is not None and all_ys:
        ymin = min(all_ys) - ymin_margin
    if ymin is not None and ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    elif ymin is not None:
        ax.set_ylim(bottom=ymin)
    elif ymax is not None:
        ax.set_ylim(top=ymax)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(fig_dir, f"fixed_final_block_doc_ppl_curve.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loaded = [load_eval_result(p) for p in args.inputs]
    ref_path, ref_payload, _ = loaded[0]
    for cur_path, cur_payload, _ in loaded[1:]:
        validate_compatible(ref_path, ref_payload, cur_path, cur_payload)

    all_results = OrderedDict()
    for _, _, model_results in loaded:
        for model_name, rows in model_results.items():
            if model_name in all_results:
                raise ValueError(f"Duplicate model name while merging: {model_name}")
            all_results[model_name] = rows

    out_json = {
        "meta": ref_payload["meta"],
        "document_stats": ref_payload["document_stats"],
        "results": all_results,
    }

    output_dir = Path(args.output_dir)
    (output_dir / "document_stats.json").write_text(
        json.dumps(ref_payload["document_stats"], indent=2) + "\n"
    )
    (output_dir / "fixed_final_block_doc_ppl.json").write_text(
        json.dumps(out_json, indent=2) + "\n"
    )

    context_lens = ref_payload["meta"]["context_lens"]
    model_names = list(all_results.keys())
    with (output_dir / "fixed_final_block_doc_ppl.csv").open("w", encoding="utf-8") as handle:
        handle.write("context_len," + ",".join(model_names) + "\n")
        for i, ctx in enumerate(context_lens):
            vals = []
            for model_name in model_names:
                ppl = all_results[model_name][i]["perplexity"]
                vals.append("" if ppl is None else f"{ppl:.8f}")
            handle.write(f"{ctx}," + ",".join(vals) + "\n")

    plot_curve(
        all_results,
        str(output_dir),
        ymax=args.ymax,
        ymin_margin=args.ymin_margin,
    )
    print(f"[INFO] Merged {len(all_results)} model results into: {output_dir}")


if __name__ == "__main__":
    main()
