"""Numerical tests for the fused teacher/student Triton attention kernel.

The default suite keeps sequence lengths modest so it can run during normal
development. Set IPTTCD_RUN_LARGE_KERNEL_TESTS=1 to include the 64K recipe
case used by the profiling benchmarks.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("triton")
_flash_attn = pytest.importorskip("flash_attn")
flash_attn_func = _flash_attn.flash_attn_func

_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_ROOT), str(_ROOT / "profiling")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fused_dual_attn import fused_dual_window_attn  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_qkv(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
    qk_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        batch,
        seqlen_q,
        heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    return q * qk_scale, k * qk_scale, v


def _flash_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int | None,
) -> torch.Tensor:
    flash_window = (-1, -1) if window is None or window < 0 else (window - 1, 0)
    return flash_attn_func(q, k, v, causal=True, window_size=flash_window)


def _dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Independent fp32 reference with FlashAttention bottom-right alignment."""
    batch, seqlen_q, heads, head_dim = q.shape
    del batch
    seqlen_k = k.shape[1]
    kv_heads = k.shape[2]
    assert heads % kv_heads == 0
    k_repeated = k.repeat_interleave(heads // kv_heads, dim=2)
    v_repeated = v.repeat_interleave(heads // kv_heads, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k_repeated.float())
    scores = scores / math.sqrt(head_dim)
    query_position = torch.arange(seqlen_q, device=q.device)[:, None] + seqlen_k - seqlen_q
    key_position = torch.arange(seqlen_k, device=q.device)[None, :]
    mask = (key_position <= query_position) & ((query_position - key_position) < window)
    probabilities = torch.softmax(scores.masked_fill(~mask[None, None], -torch.inf), dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_repeated.float())
    return output.to(q.dtype)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def _assert_fa2_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual.float() - expected.float()).abs()
    max_error = difference.max().item()
    mean_error = difference.mean().item()
    if actual.dtype == torch.bfloat16:
        atol, rtol, mean_limit = 8e-3, 1e-2, 2.5e-4
    else:
        atol, rtol, mean_limit = 2e-3, 5e-3, 6e-5
    pointwise_limit = atol + rtol * expected.float().abs()
    violating = difference > pointwise_limit
    assert not violating.any(), (
        f"{violating.sum().item()} values exceed atol={atol} + rtol={rtol}; "
        f"max error is {max_error:.6g}"
    )
    assert mean_error <= mean_limit, f"mean error {mean_error:.6g} exceeds {mean_limit:.6g}"


@pytest.mark.parametrize(
    ("batch", "seqlen", "heads", "kv_heads", "head_dim", "dtype", "teacher_window", "student_window"),
    [
        pytest.param(1, 257, 4, 2, 64, torch.bfloat16, -1, 192, id="unaligned-bf16-gqa-d64"),
        pytest.param(2, 257, 4, 4, 64, torch.float16, -1, 192, id="unaligned-fp16-mha-d64"),
        pytest.param(1, 513, 15, 5, 64, torch.bfloat16, -1, 256, id="smollm-gqa"),
        pytest.param(1, 769, 8, 1, 128, torch.bfloat16, 512, 256, id="local-bf16-mqa-d128"),
        pytest.param(1, 769, 8, 4, 128, torch.float16, 512, 256, id="local-fp16-gqa-d128"),
        pytest.param(1, 4097, 8, 8, 128, torch.bfloat16, 2048, 1024, id="block-boundary-mha"),
    ],
)
@torch.inference_mode()
def test_matches_flash_attention_2(
    batch: int,
    seqlen: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    teacher_window: int,
    student_window: int,
) -> None:
    q, k, v = _make_qkv(
        batch=batch,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=1000 + seqlen + heads,
    )
    teacher, student = fused_dual_window_attn(q, k, v, teacher_window, student_window)
    teacher_reference = _flash_reference(q, k, v, teacher_window)
    student_reference = _flash_reference(q, k, v, student_window)

    assert teacher.shape == q.shape
    assert student.shape == q.shape
    assert teacher.isfinite().all()
    assert student.isfinite().all()
    _assert_fa2_close(teacher, teacher_reference)
    _assert_fa2_close(student, student_reference)


@pytest.mark.parametrize("qk_scale", [0.0, 0.125, 1.0, 4.0, 16.0])
@torch.inference_mode()
def test_score_scale_stress_matches_flash_attention_2(qk_scale: float) -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=257,
        seqlen_k=257,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=2026,
        qk_scale=qk_scale,
    )
    teacher, student = fused_dual_window_attn(q, k, v, -1, 192)
    _assert_fa2_close(teacher, _flash_reference(q, k, v, -1))
    _assert_fa2_close(student, _flash_reference(q, k, v, 192))


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 19, 42, 73, 2026])
@torch.inference_mode()
def test_random_seed_sweep_matches_flash_attention_2(seed: int) -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=769,
        seqlen_k=769,
        heads=8,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=seed,
    )
    teacher, student = fused_dual_window_attn(q, k, v, 512, 256)
    _assert_fa2_close(teacher, _flash_reference(q, k, v, 512))
    _assert_fa2_close(student, _flash_reference(q, k, v, 256))


@torch.inference_mode()
def test_window_masks_match_independent_dense_reference() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=257,
        seqlen_k=257,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=73,
    )
    teacher, student = fused_dual_window_attn(q, k, v, -1, 192)
    teacher_max, teacher_mean = _error_stats(teacher, _dense_reference(q, k, v, 257))
    student_max, student_mean = _error_stats(student, _dense_reference(q, k, v, 192))
    assert teacher_max <= 2e-2 and teacher_mean <= 4e-4
    assert student_max <= 2e-2 and student_mean <= 4e-4


@torch.inference_mode()
def test_block_aligned_equal_windows_produce_identical_outputs() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=91,
    )
    teacher, student = fused_dual_window_attn(q, k, v, 256, 256)
    reference = _flash_reference(q, k, v, 256)
    assert torch.equal(teacher, student)
    _assert_fa2_close(teacher, reference)


@pytest.mark.xfail(
    strict=True,
    reason="A non-aligned shared left-boundary block is accumulated twice in the teacher path",
)
@torch.inference_mode()
def test_non_aligned_equal_windows_match_flash_attention_2() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=257,
        seqlen_k=257,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=91,
    )
    teacher, student = fused_dual_window_attn(q, k, v, 257, 257)
    reference = _flash_reference(q, k, v, 257)
    _assert_fa2_close(teacher, reference)
    _assert_fa2_close(student, reference)


@pytest.mark.xfail(
    strict=True,
    reason="The current kernel assumes Tq == Tk and does not apply the cached-query position offset",
)
@torch.inference_mode()
def test_cached_prefill_matches_flash_attention_bottom_right_alignment() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=256,
        seqlen_k=384,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=19,
    )
    teacher, student = fused_dual_window_attn(q, k, v, -1, 192)
    _assert_fa2_close(teacher, _flash_reference(q, k, v, -1))
    _assert_fa2_close(student, _flash_reference(q, k, v, 192))


class _ZeroMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor, t: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        del t, kwargs
        return torch.zeros_like(hidden_states)


class _GatedAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_norm = nn.Identity()
        self.gate_fn = torch.sigmoid
        self.qk_norm = False
        self.use_nope = True
        self.use_output_gate = True
        self.fuse_norm_and_gate = False
        self.window_size = None
        self.max_position_embeddings = None
        self.layer_idx = 0
        self.heads = heads


class _GatedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_ttt_layer = True
        self.ttt_chunk = 192
        self.ttt_visible_chunks = 1
        self.config = SimpleNamespace(fuse_norm=False)
        self.attn_norm = nn.Identity()
        self.mlp_norm = nn.Identity()
        self.attn = _GatedAttention(hidden_size=128, heads=2, head_dim=64)
        self.mlp = _ZeroMLP()

    def forward(self, *args, **kwargs):
        raise AssertionError("The stock fallback should not run in this test")


@pytest.mark.xfail(strict=True, reason="patch_fused_attn currently bypasses the attention output gate")
@torch.inference_mode()
def test_fused_patch_preserves_output_gate_semantics() -> None:
    from einops import rearrange
    from patches import patch_fused_attn

    torch.manual_seed(37)
    block = _GatedBlock().cuda().to(torch.bfloat16).eval()
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    hidden_states = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)

    normalized = block.attn_norm(hidden_states)
    q = rearrange(block.attn.q_proj(normalized), "... (h d) -> ... h d", d=block.attn.head_dim)
    k = rearrange(block.attn.k_proj(normalized), "... (h d) -> ... h d", d=block.attn.head_dim)
    v = rearrange(block.attn.v_proj(normalized), "... (h d) -> ... h d", d=block.attn.head_dim)
    teacher = _flash_reference(q, k, v, -1).reshape_as(hidden_states)
    gate = block.attn.gate_fn(block.attn.g_proj(normalized))
    expected = hidden_states + block.attn.o_proj(teacher * gate)

    assert patch_fused_attn(model) == 1
    actual = block(hidden_states)[0]
    _assert_fa2_close(actual, expected)


@pytest.mark.xfail(strict=True, reason="The forward-only wrapper currently accepts grad-enabled inputs silently")
def test_forward_only_wrapper_rejects_autograd_inputs() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        seed=101,
    )
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    with pytest.raises(RuntimeError, match="forward.only|autograd|gradient"):
        fused_dual_window_attn(q, k, v, -1, 192)


@pytest.mark.skipif(
    os.environ.get("IPTTCD_RUN_LARGE_KERNEL_TESTS") != "1",
    reason="set IPTTCD_RUN_LARGE_KERNEL_TESTS=1 to run the 64K recipe test",
)
@torch.inference_mode()
def test_64k_recipe_matches_flash_attention_2() -> None:
    q, k, v = _make_qkv(
        batch=1,
        seqlen_q=65536,
        seqlen_k=65536,
        heads=16,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=64,
    )
    teacher, student = fused_dual_window_attn(q, k, v, 8192, 4096)
    teacher_max, teacher_mean = _error_stats(teacher, _flash_reference(q, k, v, 8192))
    student_max, student_mean = _error_stats(student, _flash_reference(q, k, v, 4096))
    assert teacher_max <= 2e-3 and teacher_mean <= 2e-6
    assert student_max <= 2e-3 and student_mean <= 2e-6
