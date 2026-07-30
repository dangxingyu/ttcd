"""LSE-combine implementation of dual-window attention using stock flash-attn
kernels only (no custom kernel):

  student = flash(q, k, v, window=WS)                      # bitwise = baseline
  far     = flash(q[WS:], k[:-WS], v[:-WS], window=WT-WS)  # the [WS, WT) band
  teacher = lse_merge(student, far)                        # ring-attention identity

Teacher differs from a single flash call only by the split-merge rounding.
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_func


def lse_dual_window_attn(q, k, v, window_teacher: int, window_student: int):
    B, T, H, D = q.shape
    WT = T if (window_teacher is None or window_teacher < 0) else window_teacher
    WS = window_student
    assert WS < WT

    o_s, lse_s, _ = flash_attn_func(q, k, v, causal=True,
                                    window_size=(WS - 1, 0), return_attn_probs=True)
    if T <= WS:
        return o_s, o_s

    o_f, lse_f, _ = flash_attn_func(
        q[:, WS:], k[:, :-WS], v[:, :-WS], causal=True,
        window_size=(WT - WS - 1, 0), return_attn_probs=True)

    # merge (positions >= WS); lse shape (B, H, Tq)
    ls = lse_s[:, :, WS:].float()
    lf = lse_f.float()
    lt = torch.logaddexp(ls, lf)
    w_s = torch.exp(ls - lt).transpose(1, 2).unsqueeze(-1)   # (B, Tq, H, 1)
    w_f = torch.exp(lf - lt).transpose(1, 2).unsqueeze(-1)
    o_t_tail = (o_s[:, WS:].float() * w_s + o_f.float() * w_f).to(q.dtype)
    o_t = torch.cat([o_s[:, :WS], o_t_tail], dim=1)
    return o_t, o_s


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B, T, H, HK, D = 1, 65536, 16, 8, 128
    for WT, WS in [(8192, 4096), (2048, 1024), (16384, 8192)]:
        q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
        rt = flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0))
        rs = flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0))
        ot, os_ = lse_dual_window_attn(q, k, v, WT, WS)
        dt = (ot.float() - rt.float()).abs().max().item()
        ds = (os_.float() - rs.float()).abs().max().item()
        bit_s = torch.equal(os_, rs)

        def bench(fn, iters=10):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

        t_two = bench(lambda: (flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0)),
                               flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0))))
        t_lse = bench(lambda: lse_dual_window_attn(q, k, v, WT, WS))
        print(f"WT={WT} WS={WS}: teacher absΔ={dt:.2e} student bitwise={bit_s} | "
              f"two-cutlass {t_two:.2f}ms  lse-combine {t_lse:.2f}ms")


def lse_dual_window_attn_fa3(q, k, v, window_teacher: int, window_student: int):
    """FA3 flavor: flash_attn_interface returns (out, lse) natively.
    student = FA3(near, WS) bitwise; teacher = lse-merge(near, far-band)."""
    import flash_attn_interface as fa3

    B, T, H, D = q.shape
    WT = T if (window_teacher is None or window_teacher < 0) else window_teacher
    WS = window_student
    assert WS < WT

    o_s, lse_s = fa3.flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0),
                                     return_attn_probs=True)[:2]
    if T <= WS:
        return o_s, o_s
    o_f, lse_f = fa3.flash_attn_func(
        q[:, WS:].contiguous(), k[:, :-WS].contiguous(), v[:, :-WS].contiguous(),
        causal=True, window_size=(WT - WS - 1, 0), return_attn_probs=True)[:2]

    ls = lse_s[:, :, WS:].float()
    lf = lse_f.float()
    lt = torch.logaddexp(ls, lf)
    w_s = torch.exp(ls - lt).transpose(1, 2).unsqueeze(-1)
    w_f = torch.exp(lf - lt).transpose(1, 2).unsqueeze(-1)
    o_t_tail = (o_s[:, WS:].float() * w_s + o_f.float() * w_f).to(q.dtype)
    o_t = torch.cat([o_s[:, :WS], o_t_tail], dim=1)
    return o_t, o_s


import triton
import triton.language as tl


@triton.jit
def _lse_merge_kernel(OS, OF, LS, LF, OT,
                      Tq, stride_ot_b,
                      H: tl.constexpr, D: tl.constexpr,
                      BLOCK_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    tmask = offs_t < Tq

    ls = tl.load(LS + (b * H + h) * Tq + offs_t, mask=tmask, other=0.0)
    lf = tl.load(LF + (b * H + h) * Tq + offs_t, mask=tmask, other=0.0)
    m = tl.maximum(ls, lf)
    lt = m + tl.log(tl.exp(ls - m) + tl.exp(lf - m))
    ws = tl.exp(ls - lt)
    wf = tl.exp(lf - lt)

    base = b * Tq * H * D + offs_t[:, None] * H * D + h * D + offs_d[None, :]
    os_ = tl.load(OS + base, mask=tmask[:, None], other=0.0).to(tl.float32)
    of_ = tl.load(OF + base, mask=tmask[:, None], other=0.0).to(tl.float32)
    ot = os_ * ws[:, None] + of_ * wf[:, None]
    base_ot = b * stride_ot_b + offs_t[:, None] * H * D + h * D + offs_d[None, :]
    tl.store(OT + base_ot, ot.to(OT.dtype.element_ty), mask=tmask[:, None])


def lse_dual_window_attn_fa3_fast(q, k, v, window_teacher: int, window_student: int):
    """FA3-LSE with a one-pass Triton merge (student output bitwise = FA3)."""
    import flash_attn_interface as fa3

    B, T, H, D = q.shape
    WT = T if (window_teacher is None or window_teacher < 0) else window_teacher
    WS = window_student
    assert WS < WT

    o_s, lse_s = fa3.flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0),
                                     return_attn_probs=True)[:2]
    if T <= WS:
        return o_s, o_s
    o_f, lse_f = fa3.flash_attn_func(
        q[:, WS:], k[:, :-WS], v[:, :-WS],
        causal=True, window_size=(WT - WS - 1, 0), return_attn_probs=True)[:2]

    Tq = T - WS
    o_t = torch.empty_like(q)
    o_t[:, :WS] = o_s[:, :WS]
    os_tail = o_s[:, WS:].contiguous()
    ls_tail = lse_s[:, :, WS:].contiguous()
    BLOCK_T = 128
    grid = (triton.cdiv(Tq, BLOCK_T), B * H)
    _lse_merge_kernel[grid](os_tail, o_f, ls_tail, lse_f, o_t[:, WS:],
                            Tq, o_t.stride(0), H=H, D=D, BLOCK_T=BLOCK_T)
    return o_t, o_s
