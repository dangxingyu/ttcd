#!/usr/bin/env python3
"""Plot per-token NLL curves from eval_token_ppl.py JSON outputs.

Usage:
    python scripts/plot_token_ppl.py \
        --input eval_results/ipttcdv9_ct_token_ppl_65k_step10000.json:IPTTCDv9-CT \
        --input eval_results/qwen3_baseline_65k.json:Qwen3-base \
        --output eval_results/token_ppl_curve.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSON_PATH:LABEL pair, can be repeated",
    )
    parser.add_argument(
        "--output",
        default="eval_results/token_ppl_curve.png",
    )
    parser.add_argument("--metric", choices=["nll", "ppl"], default="nll")
    parser.add_argument("--title", default="books_65k",
                        help="Dataset name shown in the plot title")
    parser.add_argument("--tail_start", type=int, default=100,
                        help="Position from which to compute y-limits (ignores early spike)")
    parser.add_argument("--y_margin", type=float, default=0.15,
                        help="Fractional margin around tail min/max (log space if log-y)")
    parser.add_argument("--ema_alpha", type=float, default=0.0,
                        help="EMA smoothing factor (0 disables, default). Used only if --window=0.")
    parser.add_argument("--window", type=int, default=1024,
                        help="Centered moving-average window size (in positions). "
                             "Default 1024: each point is mean of NLL over the surrounding "
                             "1024 positions (window/2 on each side, edges shrink). 0 disables.")
    parser.add_argument("--show_raw", action="store_true",
                        help="Also overlay the unsmoothed curve as a faint line")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    def ema(x: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0 or alpha >= 1:
            return x
        out = np.empty_like(x, dtype=np.float64)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def centered_mean(x: np.ndarray, window: int) -> np.ndarray:
        """Mean over a centered window of `window` positions (window/2 each side).
        Edges shrink: y[0] uses positions [0, window/2+1)."""
        if window <= 1:
            return x
        n = len(x)
        # Cumulative-sum trick: O(n) and numerically exact.
        csum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
        half = window // 2
        idx = np.arange(n)
        lo = np.maximum(idx - half, 0)
        hi = np.minimum(idx + half + 1, n)
        return (csum[hi] - csum[lo]) / (hi - lo)

    tail_mins, tail_maxs = [], []
    for spec in args.input:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, spec
        with open(path) as f:
            data = json.load(f)
        positions = np.asarray(data["positions"], dtype=np.float64)
        nll = np.asarray(data["avg_nll"], dtype=np.float64)
        # Smooth in NLL (log-PPL) space — visually nicer when y is log
        if args.window and args.window > 1:
            nll_smooth = centered_mean(nll, args.window)
        elif args.ema_alpha > 0:
            nll_smooth = ema(nll, args.ema_alpha)
        else:
            nll_smooth = nll
        values_raw = nll if args.metric == "nll" else np.exp(nll)
        values_smooth = nll_smooth if args.metric == "nll" else np.exp(nll_smooth)

        avg_ppl = float(np.exp(nll.mean()))
        line, = ax.plot(positions, values_smooth,
                        label=f"{label} (avg PPL={avg_ppl:.2f})",
                        linewidth=1.7, alpha=0.95)
        if args.show_raw:
            ax.plot(positions, values_raw, color=line.get_color(),
                    linewidth=0.5, alpha=0.25)

        tail_mask = positions >= args.tail_start
        if tail_mask.any():
            tail_mins.append(values_smooth[tail_mask].min())
            tail_maxs.append(values_smooth[tail_mask].max())

    ax.set_xscale("log", base=2)

    # Power-of-2 x-tick labels (2, 4, 8, ..., 65536)
    xmin, xmax = ax.get_xlim()
    lo_exp = max(0, int(np.floor(np.log2(max(xmin, 1)))))
    hi_exp = int(np.ceil(np.log2(max(xmax, 2))))
    major_ticks = [2 ** e for e in range(lo_exp, hi_exp + 1)]
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$2^{{{int(round(np.log2(x)))}}}$"))

    # y-limits from plateau region (positions >= tail_start), ignoring early spike
    if tail_mins and tail_maxs:
        lo, hi = min(tail_mins), max(tail_maxs)
        pad = max((hi - lo) * args.y_margin, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Token position", fontsize=13)
    ax.set_ylabel("log PPL" if args.metric == "nll" else "Perplexity", fontsize=13)
    ax.set_title(f"Per-token language modeling loss on {args.title}", fontsize=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
