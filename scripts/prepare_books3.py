#!/usr/bin/env python3
"""Download books3 from HuggingFace and filter for long documents (>=65K tokens).

Output: parquet shards in data/books3_65k/ with 'text' column,
matching the format expected by flame's data loader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/books3_65k")
    parser.add_argument("--tokenizer", default="tokenizers/Qwen3-0.6B-Base")
    parser.add_argument("--min_tokens", type=int, default=65536)
    parser.add_argument("--shard_size", type=int, default=500,
                        help="Number of docs per parquet shard")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Max total docs to keep (0 = unlimited)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer vocab size: {tok.vocab_size}")
    print(f"Filtering for docs with >= {args.min_tokens} tokens")

    ds = load_dataset("Geralt-Targaryen/books3", split="train", streaming=True)

    kept = 0
    skipped = 0
    shard_idx = 0
    buf = []

    for i, sample in enumerate(ds):
        text = sample["text"]
        # Quick char-length pre-filter to avoid tokenizing short docs
        # 65K tokens ≈ 200K chars minimum (conservative)
        if len(text) < 150000:
            skipped += 1
            if (i + 1) % 5000 == 0:
                print(f"  Scanned {i+1} docs, kept {kept}, skipped {skipped}")
            continue

        n_tokens = len(tok.encode(text[:400000], add_special_tokens=False))
        if n_tokens < args.min_tokens:
            skipped += 1
            if (i + 1) % 5000 == 0:
                print(f"  Scanned {i+1} docs, kept {kept}, skipped {skipped}")
            continue

        buf.append(text)
        kept += 1

        if len(buf) >= args.shard_size:
            shard_path = out / f"shard_{shard_idx:04d}.parquet"
            table = pa.table({"text": buf})
            pq.write_table(table, shard_path)
            print(f"  Wrote {shard_path} ({len(buf)} docs, total kept={kept})")
            buf = []
            shard_idx += 1

        if args.max_docs > 0 and kept >= args.max_docs:
            break

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i+1} docs, kept {kept}, skipped {skipped}")

    # Write remaining
    if buf:
        shard_path = out / f"shard_{shard_idx:04d}.parquet"
        table = pa.table({"text": buf})
        pq.write_table(table, shard_path)
        print(f"  Wrote {shard_path} ({len(buf)} docs, total kept={kept})")

    print(f"\nDone. Total: scanned {i+1}, kept {kept}, skipped {skipped}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
