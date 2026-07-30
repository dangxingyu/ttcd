#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?tps:\s*(?P<tps>[0-9,]+).*?mfu:\s*(?P<mfu>[0-9.]+)%",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Titan training step metrics.")
    parser.add_argument("--log", required=True, help="Path to combined training log.")
    parser.add_argument("--tokens-per-step", type=float, required=True, help="Global tokens per step.")
    parser.add_argument("--min-step", type=int, default=1, help="Inclusive lower step bound.")
    parser.add_argument("--max-step", type=int, default=None, help="Inclusive upper step bound.")
    parser.add_argument("--output", required=True, help="Where to write the JSON summary.")
    parser.add_argument("--label", default="", help="Optional label stored in the output JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_step: dict[int, dict[str, list[float]]] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            match = METRIC_RE.search(clean)
            if match is None:
                continue
            step = int(match.group("step"))
            if step < args.min_step:
                continue
            if args.max_step is not None and step > args.max_step:
                continue
            tps = float(match.group("tps").replace(",", ""))
            mfu = float(match.group("mfu"))
            bucket = per_step.setdefault(step, {"tps": [], "mfu": []})
            bucket["tps"].append(tps)
            bucket["mfu"].append(mfu)

    selected_steps = sorted(per_step)
    if not selected_steps:
        raise SystemExit(
            f"No step metrics found in {log_path} for range [{args.min_step}, {args.max_step}]."
        )

    step_records: list[dict[str, float]] = []
    for step in selected_steps:
        avg_tps = stats.mean(per_step[step]["tps"])
        avg_mfu = stats.mean(per_step[step]["mfu"])
        step_records.append(
            {
                "step": step,
                "tps": avg_tps,
                "mfu_pct": avg_mfu,
                "sec_per_step": args.tokens_per_step / avg_tps,
            }
        )

    tps_values = [r["tps"] for r in step_records]
    mfu_values = [r["mfu_pct"] for r in step_records]
    sps_values = [r["sec_per_step"] for r in step_records]

    summary = {
        "label": args.label,
        "log": str(log_path),
        "tokens_per_step": args.tokens_per_step,
        "min_step": args.min_step,
        "max_step": args.max_step,
        "num_logged_points": len(step_records),
        "step_start": selected_steps[0],
        "step_end": selected_steps[-1],
        "avg_tps": stats.mean(tps_values),
        "median_tps": stats.median(tps_values),
        "avg_sec_per_step": stats.mean(sps_values),
        "median_sec_per_step": stats.median(sps_values),
        "avg_mfu_pct": stats.mean(mfu_values),
        "median_mfu_pct": stats.median(mfu_values),
        "per_step": step_records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
