#!/usr/bin/env python3
"""Evaluate per-token-position perplexity on books data.

Produces a curve of NLL vs token index, following TTT-E2E methodology.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import custom_models  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.distributed.checkpoint as DCP


def load_hf_model(model_path: str, device: str = "cuda",
                  max_position_embeddings: int | None = None,
                  rope_scaling: dict | None = None):
    """Load a stock HuggingFace model (e.g., Qwen3-0.6B) from safetensors.

    rope_scaling: optional dict like
        {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}
    passed into HF config so the model uses YaRN-extended RoPE instead of raw
    extrapolation beyond the pretrain context.
    """
    print(f"Loading HF model from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if max_position_embeddings is not None:
        # Allow eval beyond the trained context (RoPE extrapolation).
        config.max_position_embeddings = max_position_embeddings
    if rope_scaling is not None:
        config.rope_scaling = rope_scaling
        print(f"  [rope_scaling] {rope_scaling}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = model.to(device=device)
    model.eval()
    print(f"HF model loaded. {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def load_model_from_dcp(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load IPTTCDv9 model from DCP checkpoint."""
    print(f"Loading config from {config_path}")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    # For eval, disable fused losses
    config.fuse_cross_entropy = False
    config.fuse_linear_cross_entropy = False

    print(f"Creating model...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print(f"Loading DCP checkpoint from {checkpoint_path}")
    # Two key conventions exist in DCP files here:
    #   1. flame/torchtitan training checkpoints wrap weights under `model.`
    #      (so HF `model.embeddings.weight` becomes `model.model.embeddings.weight`).
    #   2. convert_qwen_to_ipttcd_dcp.py writes step-0 files using HF keys directly.
    # Detect by peeking at one key in the DCP metadata.
    storage_reader = DCP.filesystem.FileSystemReader(checkpoint_path)
    md = storage_reader.read_metadata()
    dcp_keys = set(md.state_dict_metadata.keys())
    hf_state = model.state_dict()
    # tied weights: if the DCP has no lm_head key at all, the convert step did not
    # write it (tie_word_embeddings=true), so we drop it from the request and let
    # model.tie_weights() / load_state_dict(strict=False) handle the share.
    sample_hf_key = next(iter(hf_state))
    if f"model.{sample_hf_key}" in dcp_keys:
        wrapped_state = {f"model.{k}": v for k, v in hf_state.items() if f"model.{k}" in dcp_keys}
        DCP.load(wrapped_state, storage_reader=storage_reader)
        loaded = {k[len("model."):]: v for k, v in wrapped_state.items()}
    else:
        wrapped_state = {k: v for k, v in hf_state.items() if k in dcp_keys}
        DCP.load(wrapped_state, storage_reader=storage_reader)
        loaded = dict(wrapped_state)
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    # For tied embeddings, the lm_head.weight is a view of embeddings.weight so
    # missing lm_head is fine after tie_weights (already done during from_config).
    tolerable = {k for k in missing if "lm_head" in k or "embeddings" in k}
    hard_missing = set(missing) - tolerable
    if hard_missing:
        raise RuntimeError(f"Missing keys in DCP load: {sorted(hard_missing)[:5]} ...")
    if unexpected:
        print(f"  warning: unexpected DCP keys (ignored): {sorted(unexpected)[:5]} ...")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def load_books_data(data_dir: str, tokenizer, seq_len: int, num_samples: int,
                    offset: int = 0, offset_max: int = 0, seed: int = 0):
    """Load books from parquet and tokenize to fixed-length sequences.

    offset behavior:
      - if offset_max > 0: per-book offset is sampled uniformly in [0, offset_max];
        books with len < offset_max + seq_len are skipped. Used to decouple the
        per-token NLL curve from absolute book position.
      - elif offset > 0: fixed offset, take tokens[offset:offset+seq_len].
      - else: tokens[:seq_len] (original behaviour).
    """
    import pyarrow.parquet as pq
    import random

    parquet_files = sorted(Path(data_dir).glob("shard_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    rng = random.Random(seed)
    use_random = offset_max > 0
    min_len = (offset_max if use_random else offset) + seq_len
    sequences = []
    offsets_used = []
    skipped = 0
    for pf in parquet_files:
        if len(sequences) >= num_samples:
            break
        table = pq.read_table(pf, columns=["text"])
        for text in table["text"].to_pylist():
            if len(sequences) >= num_samples:
                break
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < min_len:
                skipped += 1
                continue
            if use_random:
                off = rng.randint(0, offset_max)
            else:
                off = offset
            sequences.append(tokens[off:off + seq_len])
            offsets_used.append(off)

    desc = f"random offset in [0,{offset_max}]" if use_random else f"offset={offset}"
    print(f"Loaded {len(sequences)} sequences of length {seq_len} ({desc}, skipped {skipped} short docs)")
    if use_random and offsets_used:
        import statistics
        print(f"  offset stats: min={min(offsets_used)}, max={max(offsets_used)}, "
              f"mean={statistics.mean(offsets_used):.0f}")
    return sequences


@torch.no_grad()
def compute_per_token_nll(model, sequences, device="cuda", batch_size=1):
    """Compute NLL at each token position, averaged over sequences."""
    seq_len = len(sequences[0])
    # Accumulate per-position NLL
    total_nll = np.zeros(seq_len - 1)  # positions 1..seq_len-1
    count = 0

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        input_ids = torch.tensor(batch_seqs, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [B, T, V]

        # Compute per-token NLL chunkwise to avoid OOM at long context.
        # At T=65k, V≈152k, materializing fp32 log_softmax for the full
        # sequence would need ~38 GiB. Chunk along the seq dim instead.
        shift_logits = logits[:, :-1, :]  # bf16 view, [B, T-1, V]
        shift_labels = input_ids[:, 1:]  # [B, T-1]
        B, Tm1, V = shift_logits.shape
        chunk_size = 512
        token_nll_buf = torch.empty(
            (B, Tm1), dtype=torch.float32, device=shift_logits.device
        )
        for c0 in range(0, Tm1, chunk_size):
            c1 = min(c0 + chunk_size, Tm1)
            chunk = shift_logits[:, c0:c1, :].float()
            log_probs = torch.nn.functional.log_softmax(chunk, dim=-1)
            labels_c = shift_labels[:, c0:c1]
            token_lp = log_probs.gather(2, labels_c.unsqueeze(-1)).squeeze(-1)
            token_nll_buf[:, c0:c1] = -token_lp
            del chunk, log_probs, token_lp
        token_nll = token_nll_buf.cpu().numpy()  # [B, T-1]
        del logits, outputs, shift_logits, token_nll_buf

        total_nll += token_nll.sum(axis=0)
        count += len(batch_seqs)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {count}/{len(sequences)} sequences", flush=True)

    avg_nll = total_nll / count
    return avg_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Model config JSON path (DCP mode)")
    parser.add_argument("--checkpoint", type=str, default=None, help="DCP checkpoint dir (DCP mode)")
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HF model dir/repo (e.g., a Qwen3 safetensors directory). Mutually exclusive with --config/--checkpoint.",
    )
    parser.add_argument("--tokenizer", type=str, default="tokenizers/Qwen3-0.6B-Base")
    parser.add_argument("--data_dir", type=str, default="data/books_65k")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0,
                        help="Fixed start offset within each book (default 0). Ignored if --offset_max > 0.")
    parser.add_argument("--offset_max", type=int, default=0,
                        help="If > 0, sample per-book offset uniformly in [0, offset_max] "
                             "(requires books of length >= offset_max + seq_len).")
    parser.add_argument("--offset_seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="eval_results/token_ppl.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rope_scaling_type", type=str, default=None,
                        help="e.g. 'yarn' to enable YaRN RoPE scaling on HF models only")
    parser.add_argument("--rope_scaling_factor", type=float, default=None)
    parser.add_argument("--rope_scaling_original_max_pos", type=int, default=None,
                        help="Pretraining max_position_embeddings (e.g. 32768 for Qwen3, 8192 for SmolLM2)")
    args = parser.parse_args()

    use_hf = args.hf_model is not None
    if use_hf:
        if args.config or args.checkpoint:
            parser.error("--hf_model is mutually exclusive with --config/--checkpoint")
    else:
        if not (args.config and args.checkpoint):
            parser.error("Either --hf_model or both --config and --checkpoint must be provided")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Optional YaRN RoPE scaling — HF path only (DCP flame transformer doesn't
    # accept rope_scaling; those models are already trained at 65k).
    rope_scaling = None
    if args.rope_scaling_type is not None:
        if not use_hf:
            parser.error("--rope_scaling_* options are only supported for --hf_model path")
        if args.rope_scaling_factor is None or args.rope_scaling_original_max_pos is None:
            parser.error("--rope_scaling_type requires --rope_scaling_factor and --rope_scaling_original_max_pos")
        rope_scaling = {
            "type": args.rope_scaling_type,
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(args.rope_scaling_original_max_pos),
        }

    # Load model
    if use_hf:
        model = load_hf_model(args.hf_model, args.device,
                              max_position_embeddings=args.seq_len,
                              rope_scaling=rope_scaling)
    else:
        model = load_model_from_dcp(args.config, args.checkpoint, args.device)

    # Load data
    sequences = load_books_data(args.data_dir, tokenizer, args.seq_len, args.num_samples,
                                args.offset, args.offset_max, args.offset_seed)
    if len(sequences) == 0:
        print("ERROR: No sequences loaded!")
        return

    # Compute per-token NLL
    print(f"Computing per-token NLL over {len(sequences)} sequences...")
    avg_nll = compute_per_token_nll(model, sequences, args.device, args.batch_size)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "seq_len": args.seq_len,
        "num_samples": len(sequences),
        "offset": args.offset,
        "offset_max": args.offset_max,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "hf_model": args.hf_model,
        "avg_nll": avg_nll.tolist(),
        "avg_ppl": float(np.exp(avg_nll.mean())),
        "positions": list(range(1, args.seq_len)),
    }
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {args.output}")
    print(f"Overall avg PPL: {results['avg_ppl']:.2f}")
    print(f"NLL at pos 100: {avg_nll[99]:.4f}, pos 1000: {avg_nll[999]:.4f}, pos 8000: {avg_nll[7999]:.4f}, pos 16000: {avg_nll[-1]:.4f}")


if __name__ == "__main__":
    main()
