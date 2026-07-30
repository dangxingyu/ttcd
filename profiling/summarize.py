#!/usr/bin/env python3
"""Render profiling/results/*.json benchmark outputs into comparison tables."""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict

def main(pattern: str = "profiling/results/*.json"):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".trace.json.gz") or "prof" in path and not path.endswith(".json"):
            pass
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if "runs" not in data:
            continue
        label = data["name"] + ("" if data.get("ablation", "full") == "full" else f"[{data['ablation']}]")
        rows[label] = data["runs"]

    modes = sorted({k.split("/", 1)[1] for runs in rows.values() for k in runs})
    seqlens = sorted({int(k.split("/", 1)[0][1:]) for runs in rows.values() for k in runs})

    for mode in modes:
        print(f"\n=== {mode} (median ms) ===")
        header = f"{'model':30s}" + "".join(f"{f'L={L}':>12s}" for L in seqlens)
        print(header)
        for label, runs in sorted(rows.items()):
            cells = []
            for L in seqlens:
                r = runs.get(f"L{L}/{mode}")
                if r is None:
                    cells.append(f"{'-':>12s}")
                elif "error" in r:
                    cells.append(f"{'ERR':>12s}")
                else:
                    cells.append(f"{r['ms_median']:>12.1f}")
            print(f"{label:30s}" + "".join(cells))

if __name__ == "__main__":
    main(*sys.argv[1:])
