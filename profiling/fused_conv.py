"""Triton causal depthwise conv1d over the chunk axis, computed directly on
the (N=b*T, C, D) layout used by the TTT MLP (no rearrange copies, no cuDNN).

y[n, c, d] = sum_{j=0..K-1} w[d, j] * x[n, c-(K-1)+j, d]   (zero-padded left)

fp32 accumulation over the K taps (matches cuDNN's accumulate behavior).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_D": 128}, num_warps=8),
    ],
    key=["C", "D", "K"],
)
@triton.jit
def _causal_dwconv_fwd(
    X, W, Y,
    C, D,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    base = X + pid_n * C * D
    mask_cd = (offs_c[:, None] < C) & (offs_d[None, :] < D)

    acc = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    for j in tl.static_range(K):
        src = offs_c - (K - 1) + j
        x = tl.load(base + src[:, None] * D + offs_d[None, :],
                    mask=(src[:, None] >= 0) & mask_cd, other=0.0)
        w = tl.load(W + offs_d * K + j, mask=offs_d < D, other=0.0)
        acc += x.to(tl.float32) * w.to(tl.float32)[None, :]

    tl.store(Y + pid_n * C * D + offs_c[:, None] * D + offs_d[None, :],
             acc.to(Y.dtype.element_ty), mask=mask_cd)


def causal_dwconv_chunked(x: torch.Tensor, conv: torch.nn.Conv1d) -> torch.Tensor:
    """x: (b, T, C, D); conv: depthwise Conv1d(D, D, K, groups=D, padding=0).
    Equivalent to IPTTCDMLP._apply_ttt_conv with ttt_conv_causal=True."""
    b, T, C, D = x.shape
    K = conv.kernel_size[0]
    x = x.contiguous()
    y = torch.empty_like(x)
    w = conv.weight.view(D, K).contiguous()
    grid = lambda meta: (b * T, triton.cdiv(C, meta["BLOCK_C"]), triton.cdiv(D, meta["BLOCK_D"]))
    _causal_dwconv_fwd[grid](x.view(-1, C, D), w, y.view(-1, C, D), C, D, K=K)
    return y


if __name__ == "__main__":
    import time
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange

    def ref(x, conv):
        b, t, c, d = x.shape
        xf = rearrange(x, "b t c d -> (b t) d c")
        xf = F.pad(xf, (conv.kernel_size[0] - 1, 0))
        return rearrange(conv(xf), "(b t) d c -> b t c d", b=b, t=t, d=d)

    torch.manual_seed(0)
    b, T, C, D, K = 1, 16, 4096, 3072, 5
    x = torch.randn(b, T, C, D, dtype=torch.bfloat16, device="cuda")
    conv = nn.Conv1d(D, D, K, padding=0, groups=D, bias=False).cuda().bfloat16()

    y_ref = ref(x, conv)
    y = causal_dwconv_chunked(x, conv)
    d_abs = (y.float() - y_ref.float()).abs()
    bit_eq = torch.equal(y, y_ref)
    print(f"activation diff: max|Δ|={d_abs.max().item():.3e}  bitwise-equal={bit_eq}  "
          f"frac|Δ|>0: {(d_abs > 0).float().mean().item() * 100:.3f}%")

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t_ref = bench(lambda: ref(x, conv))
    t_new = bench(lambda: causal_dwconv_chunked(x, conv))
    print(f"cuDNN path (incl. rearranges): {t_ref:.2f}ms | triton: {t_new:.2f}ms")
