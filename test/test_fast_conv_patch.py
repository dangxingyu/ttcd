"""The shifted-add causal depthwise conv in profiling/patches.py must match
the reference Conv1d path of IPTTCDMLP._apply_ttt_conv exactly (fp32, CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent.parent
for p in (str(_root), str(_root / "profiling")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _reference_apply(x: torch.Tensor, conv: nn.Conv1d, causal: bool = True) -> torch.Tensor:
    # verbatim from custom_models/ipttcd/modeling_ipttcdv9.py IPTTCDMLP._apply_ttt_conv
    from einops import rearrange
    import torch.nn.functional as F
    b, t, c, d = x.shape
    x_flat = rearrange(x, "b t c d -> (b t) d c")
    if causal:
        x_flat = F.pad(x_flat, (conv.kernel_size[0] - 1, 0))
    x_flat = conv(x_flat)
    return rearrange(x_flat, "(b t) d c -> b t c d", b=b, t=t, d=d)


def _shifted_add(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    # verbatim math from profiling/patches.py patch_fast_conv
    k = conv.kernel_size[0]
    w = conv.weight.view(-1, k).to(x.dtype)
    y = x * w[:, k - 1]
    for i in range(1, k):
        y[:, :, i:, :] += x[:, :, :-i, :] * w[:, k - 1 - i]
    return y


def test_shifted_add_matches_conv1d_fp32():
    torch.manual_seed(0)
    b, t, c, d, k = 2, 3, 17, 8, 5
    x = torch.randn(b, t, c, d, dtype=torch.float32)
    conv = nn.Conv1d(d, d, kernel_size=k, padding=0, groups=d, bias=False)
    ref = _reference_apply(x, conv)
    fast = _shifted_add(x, conv)
    assert ref.shape == fast.shape
    diff = (ref - fast).abs().max().item()
    assert diff < 1e-5, f"max diff {diff}"


def test_shifted_add_matches_conv1d_bf16_tolerance():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 33, 16, dtype=torch.float32)
    conv = nn.Conv1d(16, 16, kernel_size=5, padding=0, groups=16, bias=False)
    ref = _reference_apply(x.bfloat16().float(), conv).float()
    fast = _shifted_add(x.bfloat16(), conv).float()
    rel = (ref - fast).abs().max().item() / ref.abs().max().clamp_min(1e-6).item()
    assert rel < 0.05, f"bf16 rel diff too large: {rel}"


if __name__ == "__main__":
    test_shifted_add_matches_conv1d_fp32()
    test_shifted_add_matches_conv1d_bf16_tolerance()
    print("fast_conv patch: PASS")
