"""Narrow teacher-student window gaps (WT - WS < BLOCK_M + BLOCK_N) must not
double-accumulate the teacher in the fused kernel's boundary segments.
Requires a GPU + flash-attn (run on a compute node).

Run: pytest test/test_fused_attn_narrow_gap.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parent.parent
for p in (str(_root), str(_root / "profiling")):
    if p not in sys.path:
        sys.path.insert(0, p)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _case(WT, WS, T=8192, B=1, H=8, HK=4, D=128):
    from flash_attn import flash_attn_func
    from fused_dual_attn import fused_dual_window_attn

    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
    rt = flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0))
    rs = flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0))
    ot, os_ = fused_dual_window_attn(q, k, v, WT, WS)
    dt = (ot.float() - rt.float()).abs().max().item()
    ds = (os_.float() - rs.float()).abs().max().item()
    return dt, ds


@requires_gpu
@pytest.mark.parametrize("WT,WS", [
    (4128, 4096),   # gap 32  << BLOCK_M + BLOCK_N  (shared boundary block)
    (4160, 4096),   # gap 64
    (4224, 4096),   # gap 128
    (1216, 1024),   # gap 192, small windows
    (8192, 4096),   # wide gap (recipe) -- regression guard
])
def test_no_double_accumulation(WT, WS):
    dt, ds = _case(WT, WS)
    # double-accumulation errors are O(1); rounding differences are ~1e-3
    assert dt < 5e-3, f"teacher diff too large (double count?): {dt}"
    assert ds < 5e-3, f"student diff too large: {ds}"


if __name__ == "__main__":
    for WT, WS in [(4128, 4096), (4160, 4096), (4224, 4096), (1216, 1024), (8192, 4096)]:
        dt, ds = _case(WT, WS)
        status = "OK  " if max(dt, ds) < 5e-3 else "FAIL"
        print(f"{status} WT={WT} WS={WS}: teacher absΔ={dt:.2e} student absΔ={ds:.2e}")
