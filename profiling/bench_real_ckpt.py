#!/usr/bin/env python3
"""Time the REAL eval pipeline: DCP checkpoints + real NIAH prompts, invoked
exactly like scripts/eval_niah.py does (model.generate with use_cache=True).

For each model: per-sample timings of (a) pure prefill forward
(model(input_ids, use_cache=True, logits_to_keep=1)) and (b) full
generate(max_new_tokens=64). Prints per-sample and median stats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
for p in (_project_dir, os.path.join(_project_dir, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_niah import load_dcp_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default="tokenizers/Qwen3-0.6B-Base")
    ap.add_argument("--data", default="data/niah/qwen3/ctx65536/niah_single_1/validation.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = load_dcp_model(args.config, args.checkpoint, "cuda")

    prompts = []
    with open(args.data) as f:
        for line in f:
            d = json.loads(line)
            p = d["input"] + (d.get("answer_prefix") or "")
            prompts.append(p)
            if len(prompts) >= args.n + 1:
                break

    prefill_ms, gen_ms, toks = [], [], []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            ms_p, _ = timed(lambda: model(input_ids=ids, use_cache=True, logits_to_keep=1))
            ms_g, out = timed(lambda: model.generate(
                input_ids=ids, max_new_tokens=args.max_new, do_sample=False,
                use_cache=True, pad_token_id=tok.eos_token_id or 0))
        if i == 0:
            continue  # warmup
        prefill_ms.append(ms_p)
        gen_ms.append(ms_g)
        toks.append(ids.shape[1])
        print(f"[{args.name}] sample {i}: L={ids.shape[1]} prefill={ms_p:.1f}ms "
              f"generate({args.max_new})={ms_g:.1f}ms", flush=True)

    med = lambda x: sorted(x)[len(x) // 2]
    print(f"[{args.name}] MEDIAN prefill={med(prefill_ms):.1f}ms generate={med(gen_ms):.1f}ms "
          f"decode/tok={(med(gen_ms)-med(prefill_ms))/args.max_new:.1f}ms L~{med(toks)}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"name": args.name, "prefill_ms": prefill_ms, "gen_ms": gen_ms,
                   "tokens": toks, "max_new": args.max_new}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
