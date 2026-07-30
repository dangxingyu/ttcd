#!/usr/bin/env python3
"""Evaluate a model on the full RULER synthetic suite (13 tasks).

Same data layout as eval_niah.py: {data_dir}/ctx{N}/{task}/validation.jsonl
produced by RULER's `prepare.py`.

Differences vs eval_niah.py:
  - Per-task max_new_tokens (NIAH 128, VT 30, CWE 120, FWE 50, QA 32),
    matching RULER's official tokens_to_generate.
  - Per-task metric (string_match_all for NIAH/VT/CWE/FWE; string_match_part
    for QA), copied verbatim from RULER's eval/synthetic/constants.py.
  - Writes both:
      (a) {output} — same JSON schema as eval_niah.py for our analysis;
      (b) {output_dir}/<task>_ctx<N>.jsonl — RULER's official prediction
          format (one JSONL per task per ctx) so RULER's own evaluate.py
          can be re-run end-to-end.

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


# Verbatim from 3rdparty/RULER/scripts/eval/synthetic/constants.py
def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
         for pred, ref in zip(preds, refs)]
    ) / len(preds) * 100
    return round(score, 2)


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
         for pred, ref in zip(preds, refs)]
    ) / len(preds) * 100
    return round(score, 2)


# Map task name → RULER category (from synthetic.yaml)
TASK_TO_CATEGORY = {
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa",
    "qa_2": "qa",
}

TOKENS_TO_GENERATE = {
    "niah": 128,
    "variable_tracking": 30,
    "common_words_extraction": 120,
    "freq_words_extraction": 50,
    "qa": 32,
}

CATEGORY_METRIC = {
    "niah": string_match_all,
    "variable_tracking": string_match_all,
    "common_words_extraction": string_match_all,
    "freq_words_extraction": string_match_all,
    "qa": string_match_part,
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
    m = m.to(device=device).eval()
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
    m = m.to(device=device, dtype=torch.bfloat16).eval()
    return m


@torch.no_grad()
def _custom_greedy_decode(model, input_ids: torch.Tensor, max_new_tokens: int,
                          eos_token_id) -> torch.Tensor:
    """Greedy decode bypassing HF generate's attention_mask path.

    fla.layers.attn.Attention's KV-cache decoding path runs unpad_input with
    q_len=1 and a full-length mask, which triggers a CUDA gather-out-of-bounds
    on H100. By calling the model directly with input_ids + past_key_values
    (no attention_mask) we land on the no-mask branch (`flash_attn_func`,
    no unpad).
    """
    if isinstance(eos_token_id, int):
        eos_set = {eos_token_id}
    elif eos_token_id is None:
        eos_set = set()
    else:
        eos_set = set(eos_token_id)

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
def eval_one_file(model, tokenizer, jsonl_path: Path, device: str,
                  max_new_tokens: int, max_samples: int | None = None,
                  append_answer_prefix: bool = True) -> dict:
    """Run generation over one RULER JSONL file. Returns predictions and the
    raw RULER lines so we can write both our analysis JSON and the official
    prediction JSONL format."""
    preds: list[str] = []
    refs: list[list[str]] = []
    raw_lines: list[dict] = []

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
                new_ids_t = _custom_greedy_decode(
                    model, ids, max_new_tokens, tokenizer.eos_token_id,
                )
                # Reshape to look like model.generate output: [B, prompt+new]
                out = torch.cat([ids, new_ids_t], dim=1)
            except Exception as e:
                print(f"  [warn] generate failed (idx={d.get('index')}): {e}")
                preds.append(""); refs.append(d["outputs"]); raw_lines.append(d)
                continue
            new_ids = out[0, ids.shape[1]:].tolist()
            pred = tokenizer.decode(new_ids, skip_special_tokens=True)
            preds.append(pred); refs.append(d["outputs"]); raw_lines.append(d)
            if len(preds) <= 2:
                pt = pred.replace("\n", "↵")[:160]
                print(f"  [idx={d.get('index')}]  gt={d['outputs']}   pred='{pt}'", flush=True)

    return {"n": len(preds), "preds": preds, "refs": refs, "raw": raw_lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tasks", nargs="+",
                    default=["niah_single_1", "niah_single_2", "niah_single_3",
                             "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                             "niah_multivalue", "niah_multiquery",
                             "vt", "cwe", "fwe", "qa_1", "qa_2"])
    ap.add_argument("--ctx_lens", nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--no_answer_prefix", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--hf_model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True,
                    help="JSON output path (eval_niah-style schema)")
    ap.add_argument("--pred_dir", default=None,
                    help="If set, write RULER-format predictions to "
                         "{pred_dir}/<task>_ctx<N>.jsonl alongside the JSON output")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rope_scaling_type", default=None)
    ap.add_argument("--rope_scaling_factor", type=float, default=None)
    ap.add_argument("--rope_scaling_original_max_pos", type=int, default=None)
    args = ap.parse_args()

    use_hf = args.hf_model is not None
    if use_hf and (args.config or args.checkpoint):
        ap.error("--hf_model is mutually exclusive with --config/--checkpoint")
    if not use_hf and not (args.config and args.checkpoint):
        ap.error("Either --hf_model or --config+--checkpoint required")

    rope_scaling = None
    if args.rope_scaling_type is not None:
        if not use_hf:
            ap.error("--rope_scaling_* only for --hf_model path")
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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.pred_dir:
        os.makedirs(args.pred_dir, exist_ok=True)

    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        if "per_task" not in results:
            results = {"per_task": {}, "config": {k: v for k, v in vars(args).items()}}
        print(f"[resume] loaded {len(results['per_task'])} entries from {args.output}")
    else:
        results = {"per_task": {}, "config": {k: v for k, v in vars(args).items()}}

    for task in args.tasks:
        category = TASK_TO_CATEGORY.get(task)
        if category is None:
            print(f"[skip unknown task] {task}")
            continue
        max_new = TOKENS_TO_GENERATE[category]
        metric = CATEGORY_METRIC[category]
        for ctx_len in args.ctx_lens:
            key = f"{task}_ctx{ctx_len}"
            if key in results["per_task"]:
                print(f"  [skip already done] {key}")
                continue
            path = Path(args.data_dir) / f"ctx{ctx_len}" / task / "validation.jsonl"
            if not path.exists():
                print(f"  [skip missing] {path}")
                continue
            print(f"\n=== {task} @ ctx={ctx_len}  cat={category}  max_new={max_new} ===", flush=True)
            r = eval_one_file(model, tokenizer, path, args.device,
                              max_new_tokens=max_new, max_samples=args.max_samples,
                              append_answer_prefix=not args.no_answer_prefix)
            score = metric(r["preds"], r["refs"]) if r["preds"] else 0.0
            results["per_task"][key] = {
                "task": task, "ctx_len": ctx_len, "category": category,
                "n": r["n"], "accuracy": score,
            }
            print(f"  -> n={r['n']}  accuracy={score:.2f}  ({metric.__name__})", flush=True)

            if args.pred_dir:
                pred_jsonl = Path(args.pred_dir) / f"{task}_ctx{ctx_len}.jsonl"
                with pred_jsonl.open("w") as fp:
                    for raw, pred in zip(r["raw"], r["preds"]):
                        line = dict(raw)
                        line["pred"] = pred
                        line.setdefault("others", {"id": raw.get("index", -1)})
                        fp.write(json.dumps(line) + "\n")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
