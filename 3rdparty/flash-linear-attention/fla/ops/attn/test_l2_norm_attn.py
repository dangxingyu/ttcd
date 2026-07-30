import math
import os
import sys
import torch

from fla.ops.attn import parallel_attn

RCP_LN2 = 1.4426950216

def expand_kv_to_hq(k: torch.Tensor, v: torch.Tensor, HQ: int):
    """
    k: [B, T, H, K]
    v: [B, T, H, V]
    => repeat_interleave along head dim to [B, T, HQ, ...]
    mapping matches i_h = i_hq // G (heads grouped contiguously)
    """
    H = k.shape[2]
    assert HQ % H == 0
    G = HQ // H
    k_rep = k.repeat_interleave(G, dim=2)
    v_rep = v.repeat_interleave(G, dim=2)
    return k_rep, v_rep


def g_cumsum_torch(g: torch.Tensor, cu_seqlens: torch.Tensor | None):
    """
    A torch reference for chunk_global_cumsum(g, scale=RCP_LN2) under the common convention:
    inclusive cumsum per sequence segment.
    g: [B, T, H] (or [1, total_T, H] for varlen)
    returns: [B, T, H] float32
    """
    g = g.float() * RCP_LN2
    if cu_seqlens is None:
        return torch.cumsum(g, dim=1)

    # varlen case: batch must be 1 in your op
    assert g.shape[0] == 1
    out = torch.empty_like(g, dtype=torch.float32)
    cu = cu_seqlens.to(torch.int64).cpu()
    for i in range(cu.numel() - 1):
        bos = int(cu[i].item())
        eos = int(cu[i + 1].item())
        if eos > bos:
            out[:, bos:eos, :] = torch.cumsum(g[:, bos:eos, :], dim=1)
    return out


def attn_ref_torch(
    q: torch.Tensor,            # [B, T, HQ, K]
    k: torch.Tensor,            # [B, T, H,  K]
    v: torch.Tensor,            # [B, T, H,  V]
    g: torch.Tensor | None,     # [B, T, H] or None
    scale: float,
    use_l2: bool,
    cu_seqlens: torch.Tensor | None = None,
):
    """
    Pure PyTorch reference:
      - causal mask
      - GQA by repeating k/v heads
      - scores in log2 domain (match kernel: dot*scale*RCP_LN2)
      - L1 or L2 normalization over keys
    """
    B, T, HQ, Kq = q.shape
    assert Kq == k.shape[-1]
    device = q.device

    k_rep, v_rep = expand_kv_to_hq(k, v, HQ)  # [B, T, HQ, K], [B, T, HQ, V]

    # scores_log2: [B, HQ, T, T]
    #   einsum: q[b,t,h,k] * k[b,s,h,k] -> score[b,h,t,s]
    scores_log2 = torch.einsum(
        "bthk,bshk->bhts",
        q.float(),
        k_rep.float(),
    ) * (scale * RCP_LN2)

    if g is not None:
        # compute cumsum in log2 domain, then bias = gc[t] - gc[s]
        gc = g_cumsum_torch(g, cu_seqlens=cu_seqlens)  # [B, T, H]
        # expand to HQ
        H = g.shape[2]
        assert HQ % H == 0
        G = HQ // H
        gc_rep = gc.repeat_interleave(G, dim=2)  # [B, T, HQ]
        gc_bht = gc_rep.permute(0, 2, 1).contiguous()  # [B, HQ, T]
        bias = gc_bht[:, :, :, None] - gc_bht[:, :, None, :]  # [B, HQ, T, T]
        scores_log2 = scores_log2 + bias

    # causal mask (and varlen masking if cu_seqlens provided)
    # for fixed-length: mask lower-triangular
    i = torch.arange(T, device=device)
    causal = (i[:, None] >= i[None, :])  # [T, T]

    if cu_seqlens is None:
        scores_log2 = scores_log2.masked_fill(~causal[None, None, :, :], float("-inf"))
    else:
        # varlen: for each segment [bos:eos), only allow attention within segment and causal inside it
        assert B == 1
        full_mask = torch.zeros((T, T), device=device, dtype=torch.bool)
        cu = cu_seqlens.to(torch.int64).cpu()
        for idx in range(cu.numel() - 1):
            bos = int(cu[idx].item())
            eos = int(cu[idx + 1].item())
            if eos > bos:
                seg_len = eos - bos
                ii = torch.arange(seg_len, device=device)
                seg_causal = (ii[:, None] >= ii[None, :])  # [L,L]
                full_mask[bos:eos, bos:eos] = seg_causal
        scores_log2 = scores_log2.masked_fill(~full_mask[None, None, :, :], float("-inf"))

    a = torch.exp2(scores_log2)  # [B, HQ, T, T], nonnegative

    if use_l2:
        denom = torch.sqrt(torch.sum(a * a, dim=-1, keepdim=True) + 1e-20)
    else:
        denom = torch.sum(a, dim=-1, keepdim=True)

    p = a / denom  # [B, HQ, T, T]

    # output: o[b,t,h,v] = sum_s p[b,h,t,s] * v[b,s,h,v]
    o = torch.einsum("bhts,bshv->bthv", p, v_rep.float())
    return o.to(q.dtype)


def compare(name, a, b, atol, rtol):
    diff = (a - b).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (b.abs() + 1e-6)).max().item()
    ok = (max_abs <= atol) or (max_rel <= rtol)
    print(f"{name:>10s} | max_abs={max_abs:.4e} max_rel={max_rel:.4e} | atol={atol} rtol={rtol} | {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name} mismatch: max_abs={max_abs}, max_rel={max_rel}")


@torch.no_grad()
def warmup():
    # trigger compilation once
    device = "cuda"
    B, T, H, HQ, K, V = 1, 128, 4, 8, 32, 64
    dtype = torch.float16
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, device=device, dtype=dtype)
    scale = K ** -0.5
    _ = parallel_attn(q, k, v, g=g, scale=scale, cu_seqlens=None, use_l2=False)
    _ = parallel_attn(q, k, v, g=g, scale=scale, cu_seqlens=None, use_l2=True)
    torch.cuda.synchronize()


def run_case(
    *,
    use_l2: bool,
    with_g: bool,
    varlen: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    device = "cuda"

    # small but nontrivial
    if not varlen:
        B, T, H, HQ, K, V = 2, 256, 4, 8, 32, 64
        cu_seqlens = None
        total_T = T
    else:
        # varlen requires B==1 per your op
        B, H, HQ, K, V = 1, 4, 8, 32, 64
        lens = [64, 96, 80]   # total 240
        total_T = sum(lens)
        # build cu_seqlens
        cu = [0]
        s = 0
        for L in lens:
            s += L
            cu.append(s)
        cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
        T = total_T

    scale = K ** -0.5

    # ---- triton path ----
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True) if with_g else None

    o = parallel_attn(q, k, v, g=g, scale=scale, cu_seqlens=cu_seqlens, use_l2=use_l2)
    do = torch.randn_like(o)

    loss = (o * do).sum()
    loss.backward()

    dq, dk, dv = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()
    dg = g.grad.detach().clone() if with_g else None
    o_out = o.detach().clone()

    # ---- torch reference path ----
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    g2 = g.detach().clone().requires_grad_(True) if with_g else None

    o_ref = attn_ref_torch(q2, k2, v2, g2, scale=scale, use_l2=use_l2, cu_seqlens=cu_seqlens)
    loss2 = (o_ref * do).sum()
    loss2.backward()

    # ---- compare ----
    atol, rtol = 2e-2, 2e-2

    tag = f"use_l2={use_l2} g={with_g} varlen={varlen} dtype={dtype}"
    print(f"\n=== {tag} ===")
    compare("o", o_out.float(), o_ref.detach().float(), atol=atol, rtol=rtol)
    compare("dq", dq.float(), q2.grad.detach().float(), atol=atol * 2, rtol=rtol * 2)
    compare("dk", dk.float(), k2.grad.detach().float(), atol=atol * 2, rtol=rtol * 2)
    compare("dv", dv.float(), v2.grad.detach().float(), atol=atol * 2, rtol=rtol * 2)
    if with_g:
        compare("dg", dg.float(), g2.grad.detach().float(), atol=atol * 2, rtol=rtol * 2)


def main():
    assert torch.cuda.is_available(), "CUDA is required for this test."
    warmup()

    for dtype in [torch.float16, torch.float32]:
        for use_l2 in [False, True]:
            for with_g in [False]:
                run_case(use_l2=use_l2, with_g=with_g, varlen=False, dtype=dtype)

    # optional varlen coverage
    for dtype in [torch.float16, torch.float32]:
        for use_l2 in [False, True]:
            for with_g in [False]:
                run_case(use_l2=use_l2, with_g=with_g, varlen=True, dtype=dtype)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
