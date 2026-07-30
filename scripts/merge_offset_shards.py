#!/usr/bin/env python3
"""Merge per-seed offset-eval JSONs into one weighted-average JSON.

Per-position NLL across shards is averaged weighted by num_samples:
  merged_nll[p] = sum(shard_nll[p] * shard_n) / sum(shard_n)

Usage:
    python scripts/merge_offset_shards.py \
        --inputs eval_results/qwen3_base_books3_65k_offRand32k_seed{42,43,44,45}.json \
        --output eval_results/qwen3_base_books3_65k_offRand32k_merged.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    total_n = 0
    sum_nll = None
    seq_len = None
    offset_max = None
    for path in args.inputs:
        with open(path) as f:
            d = json.load(f)
        n = d["num_samples"]
        nll = np.asarray(d["avg_nll"], dtype=np.float64)
        if sum_nll is None:
            sum_nll = nll * n
            seq_len = d["seq_len"]
            offset_max = d.get("offset_max", 0)
        else:
            assert len(nll) == len(sum_nll), f"seq_len mismatch in {path}"
            sum_nll += nll * n
        total_n += n
        print(f"  {path}: n={n}, mean_ppl={float(np.exp(nll.mean())):.3f}")

    merged = sum_nll / total_n
    out = {
        "seq_len": seq_len,
        "num_samples": total_n,
        "offset_max": offset_max,
        "merged_from": args.inputs,
        "avg_nll": merged.tolist(),
        "avg_ppl": float(np.exp(merged.mean())),
        "positions": list(range(1, seq_len)),
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"\nMerged {len(args.inputs)} shards, total n={total_n}")
    print(f"Merged avg PPL: {out['avg_ppl']:.3f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
