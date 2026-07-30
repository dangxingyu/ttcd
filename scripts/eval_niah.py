#!/usr/bin/env python3
"""Evaluate a model on RULER NIAH JSONL data produced by RULER's prepare.py.

Input data layout (written by scripts/niah_gen_ruler.sh):
    {data_dir}/ctx{N}/{task}/validation.jsonl

Each JSONL line has fields (RULER format):
    index, input, outputs, length, length_w_model_temp, answer_prefix, token_position_answer

Scoring uses RULER's official string_match_all metric (fraction of ground-truth
strings found as substrings of the prediction; averaged over samples × 100).

Supports:
  - HF model path (--hf_model)
  - flame DCP checkpoint (--config + --checkpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import custom_models  # noqa: F401
import fla  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# RULER official scoring (verbatim from 3rdparty/RULER/scripts/eval/synthetic/constants.py)
def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
         for pred, ref in zip(preds, refs)]
    ) / len(preds) * 100
    return round(score, 2)


# tokens_to_generate per RULER 3rdparty/RULER/scripts/data/synthetic/constants.py
TOKENS_TO_GENERATE = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}


def load_hf_model(model_path: str, device: str, max_pos: int | None = None,
                  rope_scaling: dict | None = None):
    print(f"Loading HF model from {model_path}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if max_pos is not None:
        cfg.max_position_embeddings = max_pos
    if rope_scaling is not None:
        cfg.rope_scaling = rope_scaling
        print(f"  [rope_scaling] {rope_scaling}")
    m = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    m = m.to(device=device)
    m.eval()
    return m


def load_dcp_model(config_path: str, ckpt: str, device: str):
    print(f"Loading DCP from {ckpt}")
    cfg = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    cfg.fuse_cross_entropy = False
    cfg.fuse_linear_cross_entropy = False
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    reader = DCP.filesystem.FileSystemReader(ckpt)
    md = reader.read_metadata()
    dcp_keys = set(md.state_dict_metadata.keys())
    hf_state = m.state_dict()
    sample_key = next(iter(hf_state))
    if f"model.{sample_key}" in dcp_keys:
        wrapped = {f"model.{k}": v for k, v in hf_state.items() if f"model.{k}" in dcp_keys}
        DCP.load(wrapped, storage_reader=reader)
        loaded = {k[len("model."):]: v for k, v in wrapped.items()}
    else:
        wrapped = {k: v for k, v in hf_state.items() if k in dcp_keys}
        DCP.load(wrapped, storage_reader=reader)
        loaded = dict(wrapped)
    missing, unexpected = m.load_state_dict(loaded, strict=False)
    tolerable = {k for k in missing if "lm_head" in k or "embeddings" in k}
    hard = set(missing) - tolerable
    if hard:
        raise RuntimeError(f"Missing keys: {sorted(hard)[:5]}")
    if unexpected:
        print(f"  (ignored unexpected: {sorted(unexpected)[:3]} ...)")
    m = m.to(device=device, dtype=torch.bfloat16)
    m.eval()
    return m


@torch.no_grad()
def eval_one_file(model, tokenizer, jsonl_path: Path, device: str,
                  max_new_tokens: int, max_samples: int | None = None,
                  append_answer_prefix: bool = True) -> dict:
    """Run generation over one RULER JSONL file and return score + predictions.

    Following RULER's official call_api.py, the model sees `input` as the
    prompt. We optionally append `answer_prefix` — for base-LM evaluation this
    helps the model emit the answer directly (mirrors RULER's design where the
    prefix was carved out of the input and stored separately)."""
    preds: list[str] = []
    refs: list[list[str]] = []
    idxs: list[int] = []

    with jsonl_path.open() as f:
        for line in f:
            if max_samples is not None and len(preds) >= max_samples:
                break
            d = json.loads(line)
            prompt = d["input"]
            if append_answer_prefix and d.get("answer_prefix"):
                prompt = prompt + d["answer_prefix"]
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            try:
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
                )
            except Exception as e:
                print(f"  [warn] generate failed (idx={d.get('index')}): {e}")
                preds.append(""); refs.append(d["outputs"]); idxs.append(d.get("index", -1))
                continue
            new_ids = out[0, ids.shape[1]:].tolist()
            pred = tokenizer.decode(new_ids, skip_special_tokens=True)
            preds.append(pred); refs.append(d["outputs"]); idxs.append(d.get("index", -1))
            if len(preds) <= 2:
                pt = pred.replace("\n", "↵")[:160]
                print(f"  [idx={idxs[-1]}]  gt={d['outputs']}   pred='{pt}'", flush=True)

    score = string_match_all(preds, refs) if preds else 0.0
    return {"n": len(preds), "accuracy": score, "predictions": preds,
            "references": refs, "indices": idxs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="e.g. data/niah/qwen3")
    ap.add_argument("--tasks", nargs="+",
                    default=["niah_single_1", "niah_single_2", "niah_single_3",
                             "niah_multikey_1", "niah_multivalue", "niah_multiquery"])
    ap.add_argument("--ctx_lens", nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="RULER niah default is 128")
    ap.add_argument("--no_answer_prefix", action="store_true",
                    help="skip appending answer_prefix (default: append, base-LM friendly)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--hf_model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_preds", action="store_true",
                    help="include raw predictions/references in output json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rope_scaling_type", default=None,
                    help="HF model only: e.g. 'yarn' to enable YaRN RoPE scaling")
    ap.add_argument("--rope_scaling_factor", type=float, default=None)
    ap.add_argument("--rope_scaling_original_max_pos", type=int, default=None)
    args = ap.parse_args()

    use_hf = args.hf_model is not None
    if use_hf:
        if args.config or args.checkpoint:
            ap.error("--hf_model is mutually exclusive with --config/--checkpoint")
    else:
        if not (args.config and args.checkpoint):
            ap.error("Either --hf_model or --config+--checkpoint required")

    rope_scaling = None
    if args.rope_scaling_type is not None:
        if not use_hf:
            ap.error("--rope_scaling_* only for --hf_model path")
        if args.rope_scaling_factor is None or args.rope_scaling_original_max_pos is None:
            ap.error("--rope_scaling_type requires --rope_scaling_factor and --rope_scaling_original_max_pos")
        rope_scaling = {
            "type": args.rope_scaling_type,
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(args.rope_scaling_original_max_pos),
        }

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    max_ctx = max(args.ctx_lens)
    if use_hf:
        model = load_hf_model(args.hf_model, args.device, max_pos=max_ctx + 256,
                              rope_scaling=rope_scaling)
    else:
        model = load_dcp_model(args.config, args.checkpoint, args.device)

    # Incremental save: if `--output` already exists from a prior (timed-out)
    # run, load it and skip task/ctx combinations we've already evaluated.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        if "per_task" not in results:
            results = {"per_task": {}, "config": {k: v for k, v in vars(args).items()}}
        print(f"[resume] loaded {len(results['per_task'])} existing entries from {args.output}")
    else:
        results = {"per_task": {}, "config": {k: v for k, v in vars(args).items()}}

    for task in args.tasks:
        for ctx_len in args.ctx_lens:
            key = f"{task}_ctx{ctx_len}"
            if key in results["per_task"]:
                print(f"  [skip already done] {key}")
                continue
            path = Path(args.data_dir) / f"ctx{ctx_len}" / task / "validation.jsonl"
            if not path.exists():
                print(f"  [skip missing] {path}")
                continue
            print(f"\n=== {task} @ ctx={ctx_len} ===", flush=True)
            r = eval_one_file(model, tokenizer, path, args.device,
                              max_new_tokens=args.max_new_tokens,
                              max_samples=args.max_samples,
                              append_answer_prefix=not args.no_answer_prefix)
            entry = {"task": task, "ctx_len": ctx_len,
                     "n": r["n"], "accuracy": r["accuracy"]}
            if args.save_preds:
                entry["predictions"] = r["predictions"]
                entry["references"] = r["references"]
                entry["indices"] = r["indices"]
            results["per_task"][key] = entry
            print(f"  -> n={r['n']}  accuracy={r['accuracy']:.2f}", flush=True)
            # Persist after every (task, ctx_len) so a timeout keeps progress.
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
