#!/usr/bin/env python3
"""Evaluate fixed-final-block perplexity using single-document contexts only."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_dir in sys.path:
    sys.path.remove(_project_dir)
sys.path.insert(0, _project_dir)
try:
    import fla  # noqa: F401
except ImportError:
    pass

import custom_models  # noqa: F401

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flame.utils.legacy_state_dict import strip_legacy_runtime_ttt_keys


DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3-0324"
DEFAULT_DATASET_PATH = "/nfs/ridgerzhu/data/emozilla/Long-Data-Collections-Pretrain-Without-Books"
DEFAULT_DATASET_DATA_DIR = os.path.join(DEFAULT_DATASET_PATH, "data")
TOTAL_PARQUET_FILES = 474
LEAVEOUT_FILES = 50

_IPTTCD_EXTRA_MODEL_TYPES = {
    "deltanet_baseline",
}
_IPTTCD_OPTIONAL_MODEL_TYPES = set()
_IPTTCD_TRITON_MODEL_TYPES = set()


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _import_or_reload(module_name: str) -> None:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)


def _ipttcd_registration_overrides(model_type: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if model_type in _IPTTCD_EXTRA_MODEL_TYPES:
        overrides["IPTTCD_REGISTER_EXTRA_VARIANTS"] = "1"
    if model_type in _IPTTCD_OPTIONAL_MODEL_TYPES:
        overrides["IPTTCD_REGISTER_OPTIONAL_VARIANTS"] = "1"
    if model_type in _IPTTCD_TRITON_MODEL_TYPES:
        overrides["IPTTCD_REGISTER_TRITON_VARIANTS"] = "1"
    return overrides


def _ensure_custom_model_registration(config_ref: str) -> None:
    config_path = os.path.abspath(config_ref)
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type")
    except Exception:
        return

    if isinstance(model_type, str) and model_type.startswith("ipttcd"):
        overrides = _ipttcd_registration_overrides(model_type)
        if overrides:
            with _temporary_env(overrides):
                _import_or_reload("custom_models.ipttcd")
        else:
            importlib.import_module("custom_models.ipttcd")
    elif isinstance(model_type, str) and model_type.startswith("ipttt"):
        importlib.import_module("custom_models.ipttt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-final-block perplexity using same-document prefixes only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Model spec format:\n'
               '  DCP: "Name|checkpoint_dir|step|config_path"\n'
               '  HF:  "Name|hf|hf_model_path"',
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", required=True)
    p.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    p.add_argument("--seq_len", type=int, default=32768,
                   help="Maximum per-document suffix length kept for eval.")
    p.add_argument("--final_block_size", type=int, default=1024)
    p.add_argument("--context_lens", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 24576, 30720])
    p.add_argument("--token_budget", type=int, default=20_000_000,
                   help="Target total token budget across eligible documents.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--min_doc_tokens_override", type=int, default=None,
                   help="If set, force the minimum document length used for document selection.")
    p.add_argument("--docs_manifest_in", type=str, default=None,
                   help="Optional .npz manifest containing preselected document suffixes.")
    p.add_argument("--docs_manifest_out", type=str, default=None,
                   help="Optional output path to save the selected document suffixes as a .npz manifest.")
    p.add_argument("--data_files", type=str, nargs="*", default=None,
                   help="Optional parquet files/globs. If omitted, use leaveout split.")
    p.add_argument("--stats_only", action="store_true",
                   help="Only scan eligible documents and write stats, without model eval.")
    p.add_argument("--skip_plot", action="store_true",
                   help="Skip plot generation and only write json/csv outputs.")
    return p.parse_args()


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def parse_model_spec(spec: str) -> tuple[str, dict]:
    parts = spec.split("|")
    if len(parts) == 3 and parts[1] == "hf":
        name, _, hf_path = parts
        return name, {"hf": hf_path}
    if len(parts) != 4:
        raise ValueError(
            f"Invalid model spec: {spec}\n"
            f'  DCP: "Name|checkpoint_dir|step|config_path"\n'
            f'  HF:  "Name|hf|hf_model_path"'
        )
    name, ckpt_dir, step, config = parts
    return name, {"checkpoint_dir": ckpt_dir, "step": int(step), "config": config}


def convert_dcp_to_hf(checkpoint_dir: str, step: int, config_path: str,
                      tokenizer_path: str, output_dir: str):
    print(f"[CONVERT] DCP step-{step} -> HF at {output_dir}")
    if os.path.isfile(config_path) and config_path.endswith(".json"):
        os.makedirs(output_dir, exist_ok=True)
        dst = os.path.join(output_dir, "config.json")
        if not os.path.exists(dst):
            shutil.copy2(config_path, dst)
        _ensure_custom_model_registration(output_dir)
        config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    else:
        _ensure_custom_model_registration(config_path)
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    dcp_path = os.path.join(checkpoint_dir, f"checkpoint/step-{step}")
    if not os.path.exists(dcp_path):
        raise FileNotFoundError(f"DCP checkpoint not found: {dcp_path}")

    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        pt_path = os.path.join(tmpdir, "checkpoint.pt")
        dcp_to_torch_save(dcp_path, pt_path)
        model = AutoModelForCausalLM.from_config(config)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        state_dict = torch.load(pt_path, map_location="cpu")["model"]
        state_dict, _ = strip_legacy_runtime_ttt_keys(model, state_dict)
        model.load_state_dict(state_dict)
        model.save_pretrained(output_dir)
        del model
    print(f"[CONVERT] Done: {output_dir}")


def ensure_hf_model(name: str, model_def: dict, output_dir: str, tokenizer_path: str) -> str:
    if "hf" in model_def:
        hf_path = model_def["hf"]
        if not os.path.exists(os.path.join(hf_path, "config.json")):
            raise FileNotFoundError(f"HF model not found: {hf_path}")
        return hf_path

    hf_dir = os.path.join(output_dir, "hf_models", f"{name}_step{model_def['step']}")
    has_config = os.path.exists(os.path.join(hf_dir, "config.json"))
    has_weights = (
        os.path.exists(os.path.join(hf_dir, "model.safetensors"))
        or os.path.exists(os.path.join(hf_dir, "pytorch_model.bin"))
    )
    if has_config and has_weights:
        print(f"[INFO] HF model exists: {hf_dir}")
        return hf_dir
    if has_config and not has_weights:
        print(f"[INFO] Incomplete HF model found, reconverting: {hf_dir}")
    os.makedirs(hf_dir, exist_ok=True)
    convert_dcp_to_hf(
        model_def["checkpoint_dir"],
        model_def["step"],
        model_def["config"],
        tokenizer_path,
        hf_dir,
    )
    return hf_dir


def resolve_data_files(data_files: Optional[List[str]]) -> List[str]:
    import glob as _glob

    if data_files:
        resolved = []
        for pattern in data_files:
            matches = sorted(_glob.glob(pattern))
            if matches:
                resolved.extend(matches)
            elif os.path.exists(pattern):
                resolved.append(pattern)
    else:
        start_idx = TOTAL_PARQUET_FILES - LEAVEOUT_FILES
        patterns = [
            os.path.join(DEFAULT_DATASET_DATA_DIR,
                         f"train-{i:05d}-of-{TOTAL_PARQUET_FILES:05d}-*.parquet")
            for i in range(start_idx, TOTAL_PARQUET_FILES)
        ]
        resolved = sorted(f for p in patterns for f in _glob.glob(p))
    if not resolved:
        raise FileNotFoundError("No parquet files matched the requested data selection.")
    return resolved


def _quantiles(values: List[int]) -> Dict[str, int]:
    if not values:
        return {}
    arr = sorted(values)

    def at(q: float) -> int:
        idx = min(len(arr) - 1, int((len(arr) - 1) * q))
        return int(arr[idx])

    return {
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": int(arr[-1]),
    }


def load_document_suffixes(
    tokenizer,
    seq_len: int,
    token_budget: int,
    min_doc_tokens: int,
    data_files: Optional[List[str]],
) -> tuple[List[np.ndarray], Dict]:
    files = resolve_data_files(data_files)
    print(f"[DATA] Scanning {len(files)} parquet files (document mode)")

    suffixes: List[np.ndarray] = []
    eligible_doc_lengths: List[int] = []
    sampled_doc_lengths: List[int] = []
    selected_total_tokens = 0
    sampled_docs = 0

    saved_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            col = table.column("text")
            texts = []
            for i in range(len(col)):
                t = col[i].as_py()
                if t and t.strip():
                    texts.append(t)
            if not texts:
                continue

            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=True,
                max_length=seq_len,
            )["input_ids"]

            for ids in encoded:
                doc_len = len(ids)
                sampled_docs += 1
                sampled_doc_lengths.append(doc_len)
                if doc_len < min_doc_tokens:
                    continue
                eligible_doc_lengths.append(doc_len)
                take = min(doc_len, seq_len)
                suffixes.append(np.array(ids[-take:], dtype=np.int64))
                selected_total_tokens += take
                if len(suffixes) % 200 == 0:
                    print(
                        f"  [DATA] docs={len(suffixes)} "
                        f"selected_tokens={selected_total_tokens / 1e6:.2f}M"
                    )
                if selected_total_tokens >= token_budget:
                    break
            if selected_total_tokens >= token_budget:
                break
        if selected_total_tokens >= token_budget:
            break

    tokenizer.truncation_side = saved_truncation_side
    stats = {
        "mode": "single_document",
        "num_files": len(files),
        "sampled_docs": sampled_docs,
        "sampled_doc_length_quantiles": _quantiles(sampled_doc_lengths),
        "eligible_docs": len(suffixes),
        "eligible_total_tokens": int(sum(eligible_doc_lengths)),
        "selected_total_tokens": int(selected_total_tokens),
        "min_doc_tokens": int(min_doc_tokens),
        "selected_doc_length_quantiles": _quantiles([len(x) for x in suffixes]),
    }
    for th in (2048, 4096, 8192, 16384, 32768):
        stats[f"sampled_docs_ge_{th}"] = int(sum(x >= th for x in sampled_doc_lengths))
        stats[f"eligible_docs_ge_{th}"] = int(sum(x >= th for x in eligible_doc_lengths))
        stats[f"eligible_tokens_ge_{th}"] = int(sum(x for x in eligible_doc_lengths if x >= th))
    return suffixes, stats


def save_document_manifest(path: str, docs: List[np.ndarray], stats: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lengths = np.array([len(x) for x in docs], dtype=np.int32)
    total = int(lengths.sum())
    flat = np.empty(total, dtype=np.int32)
    pos = 0
    for arr in docs:
        n = len(arr)
        flat[pos:pos + n] = arr.astype(np.int32, copy=False)
        pos += n
    np.savez_compressed(
        path,
        flat_tokens=flat,
        lengths=lengths,
        document_stats_json=np.array(json.dumps(stats)),
    )
    print(f"[INFO] Saved docs manifest: {path}")


def load_document_manifest(path: str) -> tuple[List[np.ndarray], Dict]:
    data = np.load(path, allow_pickle=False)
    flat = data["flat_tokens"]
    lengths = data["lengths"]
    stats = json.loads(str(data["document_stats_json"]))
    docs: List[np.ndarray] = []
    pos = 0
    for n in lengths.tolist():
        n = int(n)
        docs.append(flat[pos:pos + n].copy())
        pos += n
    if pos != len(flat):
        raise ValueError(f"Malformed docs manifest: {path}")
    print(f"[INFO] Loaded docs manifest: {path} ({len(docs)} docs)")
    return docs, stats


def evaluate_fixed_final_block_curve(
    model,
    docs: List[np.ndarray],
    context_lens: List[int],
    final_block_size: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> List[Dict]:
    autocast_enabled = device.startswith("cuda")
    autocast_dtype = dtype if autocast_enabled else torch.float32
    results = []

    with torch.inference_mode():
        for ctx in context_lens:
            eval_len = ctx + final_block_size
            eligible = [x for x in docs if len(x) >= eval_len]
            if not eligible:
                results.append({
                    "context_len": ctx,
                    "final_block_size": final_block_size,
                    "eval_len": eval_len,
                    "num_sequences": 0,
                    "total_pred_tokens": 0,
                    "avg_nll": None,
                    "perplexity": None,
                })
                print(f"[EVAL] ctx={ctx}, eval_len={eval_len} -> no eligible documents")
                continue

            total_nll = 0.0
            total_tok = 0
            n_seq = 0
            print(f"[EVAL] ctx={ctx}, eval_len={eval_len}, eligible_docs={len(eligible)}")

            for start in range(0, len(eligible), batch_size):
                batch = eligible[start:start + batch_size]
                arr = np.stack([x[-eval_len:] for x in batch], axis=0)
                x = torch.from_numpy(arr).long().to(device)
                labels = x.clone()
                labels[:, :ctx] = -100
                labels[:, 0] = -100

                with torch.autocast(
                    device_type="cuda" if autocast_enabled else "cpu",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    logits = model(input_ids=x, use_cache=False).logits
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                per_tok = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(shift_labels.shape)
                valid_mask = shift_labels.ne(-100)
                total_nll += float(per_tok[valid_mask].sum().item())
                total_tok += int(valid_mask.sum().item())
                n_seq += x.size(0)

            avg_nll = total_nll / max(total_tok, 1)
            ppl = math.exp(avg_nll)
            results.append({
                "context_len": ctx,
                "final_block_size": final_block_size,
                "eval_len": eval_len,
                "num_sequences": n_seq,
                "total_pred_tokens": total_tok,
                "avg_nll": avg_nll,
                "perplexity": ppl,
            })
            print(f"  -> ppl={ppl:.4f} sequences={n_seq} pred_tokens={total_tok}")
    return results


def plot_curve(all_results: Dict[str, List[Dict]], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, (name, rows) in enumerate(all_results.items()):
        valid = [r for r in rows if r["perplexity"] is not None]
        xs = [r["context_len"] for r in valid]
        ys = [r["perplexity"] for r in valid]
        ax.plot(xs, ys, marker="o", lw=1.8, color=colors[i % len(colors)], label=name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prefix Context Length")
    ax.set_ylabel("Perplexity on Fixed Final Block")
    ax.set_title("Fixed Final Block PPL vs Extended Context (Single Document)")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"fixed_final_block_doc_ppl_curve.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] fixed_final_block_doc_ppl_curve -> {fig_dir}/")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _parse_dtype(args.dtype)

    context_lens = sorted(set(int(x) for x in args.context_lens))
    max_required_len = max(context_lens) + args.final_block_size
    min_doc_tokens = (
        int(args.min_doc_tokens_override)
        if args.min_doc_tokens_override is not None
        else max_required_len
    )
    if min(context_lens) < 1:
        raise ValueError("context_lens must be >= 1")
    if max_required_len > args.seq_len:
        raise ValueError(
            f"max(context_lens)+final_block_size must be <= seq_len, got "
            f"{max(context_lens)}+{args.final_block_size}>{args.seq_len}"
        )

    print(f"[INFO] Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.docs_manifest_in:
        docs, doc_stats = load_document_manifest(args.docs_manifest_in)
    else:
        docs, doc_stats = load_document_suffixes(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            token_budget=args.token_budget,
            min_doc_tokens=min_doc_tokens,
            data_files=args.data_files,
        )
        if args.docs_manifest_out:
            save_document_manifest(args.docs_manifest_out, docs, doc_stats)
    if not docs:
        raise RuntimeError("No eligible single-document sequences prepared from dataset.")
    print(
        f"[INFO] Prepared {len(docs)} eligible docs, "
        f"selected_tokens={doc_stats['selected_total_tokens'] / 1e6:.2f}M"
    )
    Path(os.path.join(args.output_dir, "document_stats.json")).write_text(
        json.dumps(doc_stats, indent=2) + "\n"
    )
    if args.stats_only:
        print(f"[INFO] Stats only written to: {args.output_dir}")
        return

    try:
        import custom_models  # noqa: F401
    except ImportError:
        pass

    models = OrderedDict(parse_model_spec(s) for s in args.models)
    all_results = OrderedDict()

    for name, mdef in models.items():
        print(f"\n{'=' * 68}\nMODEL: {name}\n{'=' * 68}")
        hf_path = ensure_hf_model(name, mdef, args.output_dir, args.tokenizer)
        _ensure_custom_model_registration(hf_path)
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=dtype if device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
        ).to(device)
        # Fixed-block eval needs explicit logits. Training-oriented fused linear
        # cross entropy skips materializing logits when labels are absent.
        if getattr(model.config, "fuse_linear_cross_entropy", False):
            model.config.fuse_linear_cross_entropy = False
        ttt_enabled = bool(getattr(getattr(model, "config", None), "ttt_mode", False))
        if ttt_enabled:
            model.train()
            print("[INFO] eval mode: train() for TTT online updates (no grad)")
        else:
            model.eval()
            print("[INFO] eval mode: eval()")
        rows = evaluate_fixed_final_block_curve(
            model=model,
            docs=docs,
            context_lens=context_lens,
            final_block_size=args.final_block_size,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )
        all_results[name] = rows
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_json = {
        "meta": {
            "tokenizer": args.tokenizer,
            "seq_len": args.seq_len,
            "final_block_size": args.final_block_size,
            "context_lens": context_lens,
            "token_budget": args.token_budget,
            "num_documents": len(docs),
            "selected_total_tokens": doc_stats["selected_total_tokens"],
            "min_doc_tokens": doc_stats["min_doc_tokens"],
            "device": device,
            "dtype": args.dtype,
        },
        "document_stats": doc_stats,
        "results": all_results,
    }
    Path(os.path.join(args.output_dir, "fixed_final_block_doc_ppl.json")).write_text(
        json.dumps(out_json, indent=2) + "\n"
    )

    with open(os.path.join(args.output_dir, "fixed_final_block_doc_ppl.csv"), "w", encoding="utf-8") as f:
        names = list(all_results.keys())
        f.write("context_len," + ",".join(names) + "\n")
        for i, ctx in enumerate(context_lens):
            vals = []
            for n in names:
                v = all_results[n][i]["perplexity"]
                vals.append("" if v is None else f"{v:.8f}")
            f.write(f"{ctx}," + ",".join(vals) + "\n")

    if not args.skip_plot:
        plot_curve(all_results, args.output_dir)
    print(f"[INFO] Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
