#!/usr/bin/env python3
"""Attention-only speed across batch sizes incl. FlashAttention-3:
FA2 single/pair vs FA3 single/pair vs our fused Triton kernel."""
import sys
import time

sys.path.insert(0, "profiling")

import torch
from flash_attn import flash_attn_func as fa2

import flash_attn_interface as fa3mod

from fused_dual_attn import fused_dual_window_attn


def fa3(q, k, v, window):
    out = fa3mod.flash_attn_func(q, k, v, causal=True, window_size=window)
    return out[0] if isinstance(out, tuple) else out


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


T, H, HK, D, WT, WS = 65536, 16, 8, 128, 8192, 4096

# correctness spot check vs FA2
torch.manual_seed(0)
q = torch.randn(1, T, H, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, T, HK, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, T, HK, D, dtype=torch.bfloat16, device="cuda")
r2 = fa2(q, k, v, causal=True, window_size=(WT - 1, 0))
r3 = fa3(q, k, v, (WT - 1, 0))
print(f"FA3 vs FA2 teacher output: max|Δ|={(r2.float() - r3.float()).abs().max().item():.2e}")
del q, k, v, r2, r3

print(f"\nattention-only @64K, w{WT}/{WS} (ms):")
print(f"{'B':>3} {'FA2-1x':>8} {'FA2-2x':>8} {'FA3-1x':>8} {'FA3-2x':>8} {'fused':>8}   fused vs FA3-1x / FA3-2x")
for B in (1, 2, 4, 8):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    t2a = bench(lambda: fa2(q, k, v, causal=True, window_size=(WT - 1, 0)))
    t2b = bench(lambda: (fa2(q, k, v, causal=True, window_size=(WT - 1, 0)),
                         fa2(q, k, v, causal=True, window_size=(WS - 1, 0))))
    t3a = bench(lambda: fa3(q, k, v, (WT - 1, 0)))
    t3b = bench(lambda: (fa3(q, k, v, (WT - 1, 0)), fa3(q, k, v, (WS - 1, 0))))
    tf_ = bench(lambda: fused_dual_window_attn(q, k, v, WT, WS))
    print(f"{B:>3} {t2a:>8.2f} {t2b:>8.2f} {t3a:>8.2f} {t3b:>8.2f} {tf_:>8.2f}   "
          f"{(tf_ / t3a - 1) * 100:+.0f}% / {(tf_ / t3b - 1) * 100:+.0f}%")
    del q, k, v
    torch.cuda.empty_cache()
