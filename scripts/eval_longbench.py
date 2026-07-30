#!/usr/bin/env python3
"""LongBench (English subset) evaluation for HF or flame-DCP models.

Pipeline (per task):
  1. Load examples from data/longbench/raw/data/<task>.jsonl (downloaded via
     `huggingface_hub.hf_hub_download('THUDM/LongBench', 'data.zip')`).
  2. Apply prompt template from data/longbench/dataset2prompt.json.
  3. Truncate to --max_length tokens, keeping first half + last half (the
     LongBench convention; preserves both context start and the trailing
     "Question/Answer:" prompt).
  4. Greedy-generate up to dataset2maxlen[task] tokens.
  5. Score with task-specific metric (verbatim from LongBench's metrics.py /
     eval.py — see scripts/longbench_metrics.py).

Output:
  - {output_pred_dir}/{task}.jsonl     — one JSONL per task with predictions
  - {output_scores_path}              — single JSON with per-task scores

Models supported (same loaders as eval_token_ppl.py):
  - HF:  --hf_model <path/to/dir>
  - DCP: --config <flame-config.json> --checkpoint <DCP step dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import custom_models  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.distributed.checkpoint as DCP

from longbench_metrics import DATASET2METRIC, scorer  # noqa: E402

ENGLISH_TASKS = list(DATASET2METRIC.keys())


def load_hf_model(model_path: str, device: str = "cuda"):
    print(f"Loading HF model from {model_path}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    model = model.to(device).eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def load_model_from_dcp(config_path: str, ckpt_path: str, device: str = "cuda"):
    print(f"Loading DCP: cfg={config_path} ckpt={ckpt_path}")
    cfg = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    cfg.fuse_cross_entropy = False
    cfg.fuse_linear_cross_entropy = False
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    sr = DCP.filesystem.FileSystemReader(ckpt_path)
    md = sr.read_metadata()
    dcp_keys = set(md.state_dict_metadata.keys())
    hf_state = model.state_dict()
    sample_hf_key = next(iter(hf_state))
    if f"model.{sample_hf_key}" in dcp_keys:
        wrapped = {f"model.{k}": v for k, v in hf_state.items() if f"model.{k}" in dcp_keys}
        DCP.load(wrapped, storage_reader=sr)
        loaded = {k[len("model."):]: v for k, v in wrapped.items()}
    else:
        wrapped = {k: v for k, v in hf_state.items() if k in dcp_keys}
        DCP.load(wrapped, storage_reader=sr)
        loaded = dict(wrapped)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    hard = [k for k in missing if "lm_head" not in k and "embeddings" not in k]
    if hard:
        raise RuntimeError(f"Missing DCP keys: {hard[:5]}")
    if unexpected:
        print(f"  warning: unexpected keys (ignored): {sorted(unexpected)[:3]} ...")

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


@torch.no_grad()
def custom_greedy_decode(model, input_ids: torch.Tensor, max_new_tokens: int,
                         eos_token_id) -> torch.Tensor:
    """Greedy decode without HF generate.

    HF generate() always builds and forwards an attention_mask, which trips a
    CUDA gather-out-of-bounds inside fla.layers.attn's unpad_input path during
    decoding (q_len=1 with a full-length mask). We sidestep by calling the
    model directly with input_ids and past_key_values, never passing a mask —
    this hits the no-mask branch (`flash_attn_func` straight, no unpad).
    """
    if isinstance(eos_token_id, int):
        eos_set = {eos_token_id}
    elif eos_token_id is None:
        eos_set = set()
    else:
        eos_set = set(eos_token_id)

    # Prefill.
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if next_tok.item() in eos_set:
        return next_tok

    generated = [next_tok]
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_tok, past_key_values=past,
                    use_cache=True, return_dict=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)
        if next_tok.item() in eos_set:
            break
    return torch.cat(generated, dim=1)


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_length: int,
                 max_new_tokens: int, device: str = "cuda") -> str:
    """Encode prompt, truncate keep-both-ends to max_length, greedy decode."""
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) > max_length:
        half = max_length // 2
        prompt = (
            tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True)
        )
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

    input_ids = ids.unsqueeze(0).to(device)
    new_ids = custom_greedy_decode(
        model, input_ids, max_new_tokens, tokenizer.eos_token_id,
    )
    return tokenizer.decode(new_ids[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data_dir", default="data/longbench/raw/data",
                   help="Dir holding <task>.jsonl files.")
    p.add_argument("--prompt_config",
                   default="data/longbench/dataset2prompt.json")
    p.add_argument("--maxlen_config",
                   default="data/longbench/dataset2maxlen.json")
    p.add_argument("--tasks", nargs="+", default=ENGLISH_TASKS)
    p.add_argument("--max_length", type=int, default=31500,
                   help="Truncate prompt to this many tokens (LongBench keeps "
                        "first/last half). 31500 ≈ 32k-context default; for "
                        "64k models you can push to ~63000.")
    p.add_argument("--num_per_task", type=int, default=0,
                   help="If >0, only eval first N examples per task (debug).")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Slice: skip the first N examples (per task).")
    p.add_argument("--end_idx", type=int, default=0,
                   help="Slice: stop at this index (per task). 0 = end of file.")
    p.add_argument("--pred_filename", default="",
                   help="Override pred filename (default '<task>.jsonl'). "
                        "Useful for sharded runs writing to '<task>_shard_i.jsonl'.")
    p.add_argument("--output_pred_dir", required=True,
                   help="Dir to write per-task .jsonl predictions.")
    p.add_argument("--output_scores", required=True,
                   help="JSON file aggregating per-task scores.")
    args = p.parse_args()

    if args.hf_model is None and (args.config is None or args.checkpoint is None):
        p.error("Either --hf_model OR (--config AND --checkpoint) is required.")

    device = "cuda"
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.hf_model:
        model = load_hf_model(args.hf_model, device)
        model_id = args.hf_model
    else:
        model = load_model_from_dcp(args.config, args.checkpoint, device)
        model_id = f"{args.config}::{args.checkpoint}"

    with open(args.prompt_config) as f:
        dataset2prompt = json.load(f)
    with open(args.maxlen_config) as f:
        dataset2maxlen = json.load(f)

    os.makedirs(args.output_pred_dir, exist_ok=True)
    Path(args.output_scores).parent.mkdir(parents=True, exist_ok=True)

    results = {"model": model_id, "max_length": args.max_length, "tasks": {}}
    for task in args.tasks:
        if task not in dataset2prompt:
            print(f"[skip] {task}: not in prompt config")
            continue
        path_in = os.path.join(args.data_dir, f"{task}.jsonl")
        if not os.path.exists(path_in):
            print(f"[skip] {task}: no file {path_in}")
            continue
        prompt_tpl = dataset2prompt[task]
        max_new = dataset2maxlen[task]
        if args.pred_filename:
            path_out = os.path.join(args.output_pred_dir, args.pred_filename)
        else:
            path_out = os.path.join(args.output_pred_dir, f"{task}.jsonl")

        print(f"\n=== {task} (max_new={max_new}) ===")
        examples = [json.loads(line) for line in open(path_in)]
        if args.num_per_task > 0:
            examples = examples[: args.num_per_task]
        # Apply slice [start_idx, end_idx) for sharded runs.
        if args.end_idx > 0:
            examples = examples[args.start_idx: args.end_idx]
        elif args.start_idx > 0:
            examples = examples[args.start_idx:]
        print(f"  examples: {len(examples)} (slice [{args.start_idx}:"
              f"{args.end_idx if args.end_idx > 0 else 'end'}])")

        preds, answers, all_classes = [], [], None
        with open(path_out, "w") as f_out:
            for i, ex in enumerate(examples):
                prompt = prompt_tpl.format(**ex)
                pred = generate_one(model, tokenizer, prompt,
                                    args.max_length, max_new, device)
                preds.append(pred)
                answers.append(ex["answers"])
                all_classes = ex.get("all_classes")
                row = {
                    "id": ex.get("_id", i),
                    "pred": pred,
                    "answers": ex["answers"],
                    "length": ex.get("length"),
                    "dataset": task,
                }
                if all_classes is not None:
                    row["all_classes"] = all_classes
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(examples)}", flush=True)

        score = scorer(task, preds, answers, all_classes)
        results["tasks"][task] = {
            "score": score,
            "n": len(examples),
        }
        print(f"  → {task}: {score:.2f}  (n={len(examples)})")

    if results["tasks"]:
        results["macro_avg"] = round(
            sum(t["score"] for t in results["tasks"].values()) / len(results["tasks"]),
            2,
        )
        print(f"\nMacro avg over {len(results['tasks'])} tasks: {results['macro_avg']:.2f}")

    with open(args.output_scores, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote scores -> {args.output_scores}")
    print(f"Wrote predictions -> {args.output_pred_dir}/")


if __name__ == "__main__":
    main()
