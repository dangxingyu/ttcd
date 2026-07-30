#!/usr/bin/env python3
"""Extended A/B signoff: baseline vs fused-kernel on the real checkpoint.

NIAH: 4 tasks x {32K, 64K} x 25 samples, accuracy + prediction-identity.
PPL: mean NLL on 16 books_65k docs @ 64K, both paths.
"""
import json
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "profiling")

from pathlib import Path

import torch
from eval_niah import load_dcp_model, eval_one_file
from transformers import AutoTokenizer

import patches

CFG = "configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json"
CKPT = "exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/run/checkpoint/step-10000"
TASKS = ["niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multiquery"]
CTXS = [32768, 65536]
N = 25

tok = AutoTokenizer.from_pretrained("tokenizers/Qwen3-0.6B-Base", trust_remote_code=True)


def run_suite(model, tag):
    out = {}
    for ctx in CTXS:
        for task in TASKS:
            p = Path(f"data/niah/qwen3/ctx{ctx}/{task}/validation.jsonl")
            t0 = time.time()
            r = eval_one_file(model, tok, p, "cuda", max_new_tokens=128, max_samples=N)
            out[f"{task}@{ctx}"] = r
            print(f"[{tag}] {task}@{ctx//1024}K acc={r['accuracy']:.1f} n={r['n']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def ppl_books(model, tag, n_docs=16):
    import pyarrow.parquet as pq
    import glob
    files = sorted(glob.glob("data/books_65k/*.parquet"))
    tokens = []
    for f in files:
        tbl = pq.read_table(f, columns=["input_ids"])
        for row in tbl.column("input_ids"):
            ids = row.as_py()
            if len(ids) >= 65536:
                tokens.append(ids[:65536])
            if len(tokens) >= n_docs:
                break
        if len(tokens) >= n_docs:
            break
    nlls = []
    for ids in tokens:
        x = torch.tensor(ids, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=x, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        nll = -lp.gather(-1, x[:, 1:, None]).squeeze(-1).mean().item()
        nlls.append(nll)
        del logits, lp
    mean_nll = sum(nlls) / len(nlls)
    print(f"[{tag}] books PPL: mean NLL={mean_nll:.6f} over {len(nlls)} docs", flush=True)
    return mean_nll, nlls


def kl_pass(model_base, model_fused, n_docs=8):
    """Per-token KL(p_base || p_fused) + NLL both ways on books docs @64K."""
    import glob
    import pyarrow.parquet as pq
    docs = []
    for f in sorted(glob.glob("data/books_65k/*.parquet")):
        tbl = pq.read_table(f, columns=["text"])
        for row in tbl.column("text"):
            ids = tok(row.as_py(), add_special_tokens=False).input_ids
            if len(ids) >= 65536:
                docs.append(ids[:65536])
            if len(docs) >= n_docs:
                break
        if len(docs) >= n_docs:
            break
    kls, nll_b, nll_f, flips = [], [], [], 0
    total_pos = 0
    for di, ids in enumerate(docs):
        x = torch.tensor(ids, device="cuda").unsqueeze(0)
        with torch.no_grad():
            lb = model_base(input_ids=x, use_cache=False).logits
            lf = model_fused(input_ids=x, use_cache=False).logits
        doc_kl = torch.empty(lb.shape[1], dtype=torch.float32)
        for c0 in range(0, lb.shape[1], 2048):
            c1 = min(c0 + 2048, lb.shape[1])
            pb = torch.log_softmax(lb[0, c0:c1].float(), -1)
            pf = torch.log_softmax(lf[0, c0:c1].float(), -1)
            doc_kl[c0:c1] = (pb.exp() * (pb - pf)).sum(-1).cpu()
            flips += (pb.argmax(-1) != pf.argmax(-1)).sum().item()
            total_pos += c1 - c0
            if c0 == 0:
                tgt = x[0, 1:c1 + 1] if c1 < lb.shape[1] else x[0, 1:]
            del pb, pf
        lpb = torch.log_softmax(lb[0, :-1].float(), -1)
        nll_b.append(-lpb.gather(-1, x[0, 1:, None]).squeeze(-1).mean().item())
        del lpb
        lpf = torch.log_softmax(lf[0, :-1].float(), -1)
        nll_f.append(-lpf.gather(-1, x[0, 1:, None]).squeeze(-1).mean().item())
        del lpf, lb, lf
        kls.append(doc_kl)
        print(f"[KL] doc {di + 1}/{len(docs)} done", flush=True)
        torch.cuda.empty_cache()
    kl = torch.cat(kls)
    torch.save(kl, "profiling/results/kl_per_token.pt")
    q = lambda p: torch.quantile(kl.double(), p).item()
    print(f"[KL] n={kl.numel()}  p50={q(0.5):.2e} p90={q(0.9):.2e} p99={q(0.99):.2e} "
          f"p99.9={q(0.999):.2e} max={kl.max().item():.2e}", flush=True)
    print(f"[KL] top-1 flip rate: {flips}/{total_pos} = {flips / total_pos * 100:.4f}%", flush=True)
    print(f"[KL] mean NLL base={sum(nll_b) / len(nll_b):.6f} fused={sum(nll_f) / len(nll_f):.6f} "
          f"Δ={(sum(nll_f) - sum(nll_b)) / len(nll_b):+.2e}", flush=True)
    return kl


print("loading model...", flush=True)
m = load_dcp_model(CFG, CKPT, "cuda")

base = run_suite(m, "BASE")

print("applying fused_attn patch...", flush=True)
patches.patch_fused_attn(m)

fused = run_suite(m, "FUSED")

print("loading second (unpatched) model for KL pass...", flush=True)
m_base = load_dcp_model(CFG, CKPT, "cuda")
kl_pass(m_base, m)

print("\n===== SUMMARY =====", flush=True)
tot_same, tot_n = 0, 0
for key in base:
    b, f = base[key], fused[key]
    same = sum(a.strip() == c.strip() for a, c in zip(b["predictions"], f["predictions"]))
    tot_same += same
    tot_n += b["n"]
    flag = "" if b["accuracy"] == f["accuracy"] else "  <-- DIFF"
    print(f"{key:26s} base={b['accuracy']:5.1f} fused={f['accuracy']:5.1f} identical_preds={same}/{b['n']}{flag}", flush=True)
print(f"identical predictions overall: {tot_same}/{tot_n}", flush=True)
json.dump({"base": {k: v["accuracy"] for k, v in base.items()},
           "fused": {k: v["accuracy"] for k, v in fused.items()}},
          open("profiling/results/eval_ab_extended.json", "w"), indent=2)
print("DONE", flush=True)
