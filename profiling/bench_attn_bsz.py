#!/usr/bin/env python3
"""Attention-only speed across batch sizes: FA2 single/pair vs fused vs LSE."""
import sys
import time

sys.path.insert(0, "profiling")

import torch
from flash_attn import flash_attn_func

from fused_dual_attn import fused_dual_window_attn
from lse_dual_attn import lse_dual_window_attn


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
print(f"attention-only @64K, w{WT}/{WS} (ms):")
print(f"{'B':>3} {'FA2-teacher':>12} {'FA2-pair':>10} {'fused':>8} {'LSE':>8}   fused vs FA2-single")
for B in (1, 2, 4, 8):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    t1 = bench(lambda: flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0)))
    t2 = bench(lambda: (flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0)),
                        flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0))))
    tf_ = bench(lambda: fused_dual_window_attn(q, k, v, WT, WS))
    tl = bench(lambda: lse_dual_window_attn(q, k, v, WT, WS))
    print(f"{B:>3} {t1:>12.2f} {t2:>10.2f} {tf_:>8.2f} {tl:>8.2f}   {(tf_ / t1 - 1) * 100:+.0f}%")
    del q, k, v
    torch.cuda.empty_cache()
