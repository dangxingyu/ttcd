"""Fused dual-window causal attention (Triton, v3): one pass over K/V produces
BOTH the teacher (window Wt) and student (window Ws <= Wt) outputs.

Numerics are structured to mirror CUTLASS FlashAttention-2 step by step:
  - row max kept in the RAW score domain; p = exp2(s*c - m*c) (two-multiply
    form, matching flash-attn's apply_exp2 with scale_max=true);
  - KV blocks processed BACKWARD from the diagonal (flash-attn's order), so
    the online-softmax rescale sequence matches;
  - masking with true -inf (backward order guarantees the first block of every
    row contains its diagonal, so running maxima are finite afterwards);
  - epilogue multiplies by 1/l (reciprocal), not division.
Remaining difference vs CUTLASS is only intra-mma reduction order (last-bit).

Loop structure (per q-block, backward):
  A: diagonal blocks   — causal mask; both outputs (masks identical -> student
                         P@V recovered from teacher's via row rescale);
  B: shared interior   — NO masks; both outputs (rescale reuse);
  C: student boundary  — teacher unmasked, student window-masked (full path);
  D: teacher interior  — NO masks; teacher only;
  E: teacher boundary  — teacher window mask; teacher only.

Layout: q (B, T, H, D), k/v (B, T, HK, D), GQA, bf16/fp16. Forward only.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
NEG_INF = tl.constexpr(float("-inf"))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=3),
    ],
    key=["T", "WT", "WS"],
)
@triton.jit
def _fused_dual_swa_fwd(
    Q, K, V, OT, OS,
    sm_scale,
    T,
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_ob, stride_ot_, stride_oh, stride_od,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    WT: tl.constexpr, WS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // (H // HK)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + b * stride_qb + offs_m[:, None] * stride_qt + h * stride_qh + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < T, other=0.0)

    acc_t = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    acc_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_t = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    m_s = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_t = tl.zeros([BLOCK_M], dtype=tl.float32)
    l_s = tl.zeros([BLOCK_M], dtype=tl.float32)

    c = sm_scale * LOG2E                                # log2-domain scale
    m_start = pid_m * BLOCK_M
    m_end = m_start + BLOCK_M - 1                       # last row index

    # block-index boundaries (all inclusive-exclusive on block index nb)
    nb_last = m_end // BLOCK_N                          # last diagonal block
    nb_diag = m_start // BLOCK_N                        # first causal-masked block
    nb_s_full = tl.maximum((m_start + BLOCK_M - WS + BLOCK_N - 1) // BLOCK_N, 0)   # ceil
    nb_s_any = tl.maximum((m_start - WS + 1) // BLOCK_N, 0)
    nb_t_full = tl.maximum((m_start + BLOCK_M - WT + BLOCK_N - 1) // BLOCK_N, 0)
    nb_t_any = tl.maximum((m_start - WT + 1) // BLOCK_N, 0)

    kv_base_k = K + b * stride_kb + hk * stride_kh
    kv_base_v = V + b * stride_kb + hk * stride_kh

    # ---- A: diagonal blocks (backward), causal mask, both outputs ----
    for nb in range(nb_last, nb_diag - 1, -1):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < T
        k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=kv_mask, other=0.0)
        v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd, mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k))
        dist = offs_m[:, None] - offs_n[None, :]
        s = tl.where((dist >= 0) & (offs_n[None, :] < T), s, NEG_INF)

        m_t_new = tl.maximum(m_t, tl.max(s, 1))
        mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)   # CUTLASS guard
        alpha_t = tl.math.exp2(m_t * c - mtc)
        p_t = tl.math.exp2(s * c - mtc[:, None])
        pv = tl.dot(p_t.to(v.dtype), v)
        acc_t = acc_t * alpha_t[:, None] + pv
        l_t = l_t * alpha_t + tl.sum(p_t, 1)
        m_t = m_t_new

        # identical mask for student in diagonal blocks -> rescale reuse
        m_s_new = tl.maximum(m_s, m_t_new)
        msc = tl.where(m_s_new == NEG_INF, 0.0, m_s_new * c)
        alpha_s = tl.math.exp2(m_s * c - msc)
        scale = tl.math.exp2(mtc - msc)
        acc_s = acc_s * alpha_s[:, None] + pv * scale[:, None]
        l_s = l_s * alpha_s + tl.sum(p_t, 1) * scale
        m_s = m_s_new

    # ---- B: shared interior blocks (backward), no masks, both outputs ----
    for nb in range(nb_diag - 1, nb_s_full - 1, -1):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
        v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
        s = tl.dot(q, tl.trans(k))

        m_t_new = tl.maximum(m_t, tl.max(s, 1))
        mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)   # CUTLASS guard
        alpha_t = tl.math.exp2(m_t * c - mtc)
        p_t = tl.math.exp2(s * c - mtc[:, None])
        pv = tl.dot(p_t.to(v.dtype), v)
        acc_t = acc_t * alpha_t[:, None] + pv
        l_t = l_t * alpha_t + tl.sum(p_t, 1)
        m_t = m_t_new

        m_s_new = tl.maximum(m_s, m_t_new)
        msc = tl.where(m_s_new == NEG_INF, 0.0, m_s_new * c)
        alpha_s = tl.math.exp2(m_s * c - msc)
        scale = tl.math.exp2(mtc - msc)
        acc_s = acc_s * alpha_s[:, None] + pv * scale[:, None]
        l_s = l_s * alpha_s + tl.sum(p_t, 1) * scale
        m_s = m_s_new

    # Wide-gap configs (all recipe settings, WT >= WS + BLOCK_M + BLOCK_N or
    # full attention): boundary regions are disjoint -> keep the specialized
    # C/D/E segments (bitwise-identical to the signed-off kernel).
    # Narrow-gap configs (WT - WS < BLOCK_M + BLOCK_N): teacher/student
    # boundaries can share KV blocks; use one combined dual-masked segment to
    # avoid double-accumulating the teacher (constexpr branch, zero runtime
    # cost for the wide case).
    if WT >= WS + BLOCK_M + BLOCK_N:
        # ---- C: student boundary blocks; teacher unmasked, student masked ----
        for nb in range(nb_s_full - 1, nb_s_any - 1, -1):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            s = tl.dot(q, tl.trans(k))

            m_t_new = tl.maximum(m_t, tl.max(s, 1))
            mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)   # CUTLASS guard
            alpha_t = tl.math.exp2(m_t * c - mtc)
            p_t = tl.math.exp2(s * c - mtc[:, None])
            acc_t = acc_t * alpha_t[:, None] + tl.dot(p_t.to(v.dtype), v)
            l_t = l_t * alpha_t + tl.sum(p_t, 1)
            m_t = m_t_new

            dist = offs_m[:, None] - offs_n[None, :]
            s_s = tl.where(dist < WS, s, NEG_INF)
            m_s_new = tl.maximum(m_s, tl.max(s_s, 1))
            msc = tl.where(m_s_new == NEG_INF, 0.0, m_s_new * c)
            alpha_s = tl.math.exp2(m_s * c - msc)
            p_s = tl.math.exp2(s_s * c - msc[:, None])
            acc_s = acc_s * alpha_s[:, None] + tl.dot(p_s.to(v.dtype), v)
            l_s = l_s * alpha_s + tl.sum(p_s, 1)
            m_s = m_s_new

        # ---- D: teacher interior blocks, no masks, teacher only ----
        for nb in range(nb_s_any - 1, nb_t_full - 1, -1):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            s = tl.dot(q, tl.trans(k))

            m_t_new = tl.maximum(m_t, tl.max(s, 1))
            mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)   # CUTLASS guard
            alpha_t = tl.math.exp2(m_t * c - mtc)
            p_t = tl.math.exp2(s * c - mtc[:, None])
            acc_t = acc_t * alpha_t[:, None] + tl.dot(p_t.to(v.dtype), v)
            l_t = l_t * alpha_t + tl.sum(p_t, 1)
            m_t = m_t_new

        # ---- E: teacher boundary blocks, window mask, teacher only ----
        for nb in range(nb_t_full - 1, nb_t_any - 1, -1):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            s = tl.dot(q, tl.trans(k))
            dist = offs_m[:, None] - offs_n[None, :]
            s = tl.where(dist < WT, s, NEG_INF)

            m_t_new = tl.maximum(m_t, tl.max(s, 1))
            mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)   # CUTLASS guard
            alpha_t = tl.math.exp2(m_t * c - mtc)
            p_t = tl.math.exp2(s * c - mtc[:, None])
            acc_t = acc_t * alpha_t[:, None] + tl.dot(p_t.to(v.dtype), v)
            l_t = l_t * alpha_t + tl.sum(p_t, 1)
            m_t = m_t_new

    else:
        # ---- combined boundary segment (narrow gap): both masks, full updates ----
        for nb in range(nb_s_full - 1, nb_t_any - 1, -1):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            k = tl.load(kv_base_k + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            v = tl.load(kv_base_v + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd)
            s = tl.dot(q, tl.trans(k))
            dist = offs_m[:, None] - offs_n[None, :]

            s_t = tl.where(dist < WT, s, NEG_INF)
            m_t_new = tl.maximum(m_t, tl.max(s_t, 1))
            mtc = tl.where(m_t_new == NEG_INF, 0.0, m_t_new * c)
            alpha_t = tl.math.exp2(m_t * c - mtc)
            p_t = tl.math.exp2(s_t * c - mtc[:, None])
            acc_t = acc_t * alpha_t[:, None] + tl.dot(p_t.to(v.dtype), v)
            l_t = l_t * alpha_t + tl.sum(p_t, 1)
            m_t = m_t_new

            s_s = tl.where(dist < WS, s, NEG_INF)
            m_s_new = tl.maximum(m_s, tl.max(s_s, 1))
            msc = tl.where(m_s_new == NEG_INF, 0.0, m_s_new * c)
            alpha_s = tl.math.exp2(m_s * c - msc)
            p_s = tl.math.exp2(s_s * c - msc[:, None])
            acc_s = acc_s * alpha_s[:, None] + tl.dot(p_s.to(v.dtype), v)
            l_s = l_s * alpha_s + tl.sum(p_s, 1)
            m_s = m_s_new

    inv_t = 1.0 / l_t
    inv_s = 1.0 / l_s
    ot = acc_t * inv_t[:, None]
    os_ = acc_s * inv_s[:, None]
    tl.store(OT + b * stride_ob + offs_m[:, None] * stride_ot_ + h * stride_oh + offs_d[None, :] * stride_od,
             ot.to(OT.dtype.element_ty), mask=offs_m[:, None] < T)
    tl.store(OS + b * stride_ob + offs_m[:, None] * stride_ot_ + h * stride_oh + offs_d[None, :] * stride_od,
             os_.to(OS.dtype.element_ty), mask=offs_m[:, None] < T)


def fused_dual_window_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           window_teacher: int, window_student: int,
                           block_n: int = 64):
    """Returns (o_teacher, o_student); windows follow flash_attn semantics of
    window_size=(W-1, 0). window_teacher None/-1 = full causal.

    block_n affects the online-softmax block sequence and hence last-bit
    rounding; 64 empirically matches CUTLASS FA2 most closely on H100."""
    B, T, H, D = q.shape
    HK = k.shape[2]
    if window_teacher is None or window_teacher < 0:
        window_teacher = T
    assert window_student <= window_teacher
    # loop-structure assumptions (hold for all recipe configs):
    assert window_student >= 128 + block_n, "WS too small for diagonal-block assumption"
    # narrow teacher-student gaps are handled by the combined boundary segment
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    ot = torch.empty_like(q)
    os_ = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_M"]), B * H)
    _fused_dual_swa_fwd[grid](
        q, k, v, ot, os_,
        D ** -0.5, T,
        *q.stride(), *k.stride(), *ot.stride(),
        H=H, HK=HK, D=D,
        WT=window_teacher, WS=window_student,
        BLOCK_N=block_n,
    )
    return ot, os_


def _ulp_report(name, a, ref):
    ai = a.view(torch.int16).to(torch.int32)
    ri = ref.view(torch.int16).to(torch.int32)
    # map bf16 bit patterns to a monotonic integer scale (sign-magnitude -> offset)
    ai = torch.where(ai < 0, -32768 - ai, ai)
    ri = torch.where(ri < 0, -32768 - ri, ri)
    d = (ai - ri).abs()
    n = d.numel()
    eq = (d == 0).sum().item() / n * 100
    le1 = (d <= 1).sum().item() / n * 100
    le2 = (d <= 2).sum().item() / n * 100
    print(f"  {name}: bit-equal {eq:.2f}%  |Δ|<=1ulp {le1:.3f}%  <=2ulp {le2:.4f}%  max {d.max().item()} ulp")


if __name__ == "__main__":
    import time
    from flash_attn import flash_attn_func

    torch.manual_seed(0)
    B, T, H, HK, D = 1, 65536, 16, 8, 128
    for WT, WS in [(8192, 4096), (2048, 1024), (16384, 8192)]:
        q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, T, HK, D, dtype=torch.bfloat16, device="cuda")

        ot_ref = flash_attn_func(q, k, v, causal=True, window_size=(WT - 1, 0))
        os_ref = flash_attn_func(q, k, v, causal=True, window_size=(WS - 1, 0))
        ot, os_ = fused_dual_window_attn(q, k, v, WT, WS)

        print(f"WT={WT} WS={WS}")
        _ulp_report("teacher", ot, ot_ref)
        _ulp_report("student", os_, os_ref)

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
        t_fused = bench(lambda: fused_dual_window_attn(q, k, v, WT, WS))
        print(f"  two-cutlass {t_two:.2f}ms  fused-triton {t_fused:.2f}ms")
