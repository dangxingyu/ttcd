#!/usr/bin/env python3
"""Collect per-token KL(p_base||p_fused) AND TV distance on books docs @64K
(real checkpoint, baseline vs fused-kernel paths). Saves both tensors."""
import glob
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "profiling")

import pyarrow.parquet as pq
import torch
from eval_niah import load_dcp_model

import patches

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n-docs", type=int, default=8)
ap.add_argument("--skip-docs", type=int, default=0)
ap.add_argument("--suffix", default="")
args = ap.parse_args()

CFG = "configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json"
CKPT = "exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/run/checkpoint/step-10000"
N_DOCS = args.n_docs
SKIP = args.skip_docs

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizers/Qwen3-0.6B-Base", trust_remote_code=True)
docs = []
seen_long = 0
for f in sorted(glob.glob("data/books_65k/*.parquet")):
    tbl = pq.read_table(f, columns=["text"])
    for row in tbl.column("text"):
        txt = row.as_py()
        if len(txt) < 65536 * 2:   # cheap pre-filter by chars
            continue
        ids = tok(txt, add_special_tokens=False).input_ids
        if len(ids) >= 65536:
            seen_long += 1
            if seen_long > SKIP:
                docs.append(ids[:65536])
                if len(docs) % 10 == 0:
                    print(f"tokenized doc {len(docs)}/{N_DOCS}", flush=True)
        if len(docs) >= N_DOCS:
            break
    if len(docs) >= N_DOCS:
        break

m_base = load_dcp_model(CFG, CKPT, "cuda")
m_fused = load_dcp_model(CFG, CKPT, "cuda")
patches.patch_fused_attn(m_fused)

kls, tvs, flips, total = [], [], 0, 0
for di, ids in enumerate(docs):
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    with torch.no_grad():
        lb = m_base(input_ids=x, use_cache=False).logits
        lf = m_fused(input_ids=x, use_cache=False).logits
    T = lb.shape[1]
    doc_kl = torch.empty(T, dtype=torch.float32)
    doc_tv = torch.empty(T, dtype=torch.float32)
    for c0 in range(0, T, 2048):
        c1 = min(c0 + 2048, T)
        logpb = torch.log_softmax(lb[0, c0:c1].float(), -1)
        logpf = torch.log_softmax(lf[0, c0:c1].float(), -1)
        pb = logpb.exp()
        doc_kl[c0:c1] = (pb * (logpb - logpf)).sum(-1).cpu()
        doc_tv[c0:c1] = (0.5 * (pb - logpf.exp()).abs().sum(-1)).cpu()
        flips += (logpb.argmax(-1) != logpf.argmax(-1)).sum().item()
        total += c1 - c0
        del logpb, logpf, pb
    del lb, lf
    kls.append(doc_kl)
    tvs.append(doc_tv)
    print(f"doc {di + 1}/{len(docs)} done", flush=True)
    torch.cuda.empty_cache()

kl = torch.cat(kls)
tv = torch.cat(tvs)
torch.save(kl, f"profiling/results/kl_per_token{args.suffix}.pt")
torch.save(tv, f"profiling/results/tv_per_token{args.suffix}.pt")
q = lambda t, p: torch.quantile(t.double(), p).item()
for name, t in [("KL", kl), ("TV", tv)]:
    print(f"{name}: n={t.numel()} p50={q(t, .5):.2e} p90={q(t, .9):.2e} p99={q(t, .99):.2e} "
          f"p99.9={q(t, .999):.2e} max={t.max().item():.2e}", flush=True)
print(f"top-1 flip rate: {flips}/{total} = {flips / total * 100:.4f}%", flush=True)
