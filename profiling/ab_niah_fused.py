#!/usr/bin/env python3
"""A/B NIAH sanity: integrated baseline vs fused dual-window attention kernel
on the real step-10000 checkpoint (12 samples, niah_single_1 @ 64K)."""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "profiling")

from pathlib import Path

from eval_niah import load_dcp_model, eval_one_file
from transformers import AutoTokenizer

import patches

tok = AutoTokenizer.from_pretrained("tokenizers/Qwen3-0.6B-Base", trust_remote_code=True)
m = load_dcp_model("configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json",
                   "exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/run/checkpoint/step-10000", "cuda")
p = Path("data/niah/qwen3/ctx65536/niah_single_1/validation.jsonl")

r0 = eval_one_file(m, tok, p, "cuda", max_new_tokens=64, max_samples=12)
print(f"BASELINE      acc={r0['accuracy']:.1f} n={r0['n']}")

patches.patch_fused_attn(m)
r1 = eval_one_file(m, tok, p, "cuda", max_new_tokens=64, max_samples=12)
print(f"FUSED-KERNEL  acc={r1['accuracy']:.1f} n={r1['n']}")

same = sum(a.strip() == b.strip() for a, b in zip(r0["predictions"], r1["predictions"]))
print(f"identical predictions: {same}/{r0['n']}")
