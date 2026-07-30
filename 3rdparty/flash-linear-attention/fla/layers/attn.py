# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding, FusedRMSNormGated
from fla.modules.activations import ACT2FN
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

def _should_return_full_flash_output(kwargs) -> bool:
    return bool(
        kwargs.get("return_attn_probs", False)
        or kwargs.get("return_lse", False)
        or kwargs.get("return_aux", False)
    )


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    try:
        from flash_attn.cute import (
            flash_attn_func as _cute_flash_attn_func,
            flash_attn_varlen_func as _cute_flash_attn_varlen_func,
        )

        def flash_attn_func(*args, **kwargs):
            out = _cute_flash_attn_func(*args, **kwargs)
            if _should_return_full_flash_output(kwargs):
                return out
            return out[0] if isinstance(out, tuple) else out

        def flash_attn_varlen_func(*args, **kwargs):
            out = _cute_flash_attn_varlen_func(*args, **kwargs)
            if _should_return_full_flash_output(kwargs):
                return out
            return out[0] if isinstance(out, tuple) else out
    except ImportError:
        warnings.warn(
            "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation` "
            "or `pip install --pre flash-attn-4` for the CuTeDSL FA4 backend.",
            category=ImportWarning,
        )
        flash_attn_func = None
        flash_attn_varlen_func = None
    
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn import parallel_attn

logger = logging.get_logger(__name__)


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        # l2_softmax
        use_l2_softmax: bool = False,
        # gated output
        use_output_gate: bool = False,
        gate_fn: str = 'sigmoid',
        elementwise_affine: bool | None = True,
        gate_logit_normalizer: int = 16,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        # nope
        use_nope: bool = False,
        # Optional rope_scaling dict (HF-style); supports type="yarn" with
        # factor + original_max_position_embeddings (+ optional beta_fast/slow,
        # mscale). When provided, the rotary embedding uses YaRN-aware inv_freq
        # and bakes mscale into cos/sin.
        rope_scaling: dict | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.use_nope = use_nope
        if not self.use_nope:
            rotary_kwargs = {}
            if rope_scaling is not None:
                rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
                if rope_type == "yarn":
                    rotary_kwargs.update(
                        yarn_factor=float(rope_scaling.get("factor", 1.0)),
                        yarn_original_max_position=int(
                            rope_scaling.get("original_max_position_embeddings",
                                             self.max_position_embeddings or 0)
                        ),
                        yarn_beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
                        yarn_beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
                        yarn_mscale=float(rope_scaling.get("mscale", 1.0)),
                        yarn_mscale_all_dim=float(rope_scaling.get("mscale_all_dim", 0.0)),
                    )
                else:
                    raise NotImplementedError(
                        f"FLA Attention rope_scaling: only type='yarn' is supported, got {rope_type!r}"
                    )
            self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta, **rotary_kwargs)

        self.use_l2_softmax = use_l2_softmax

        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
            if gate_fn == 'swish' and fuse_norm:
                self.g_norm_swish_gate = FusedRMSNormGated(
                    hidden_size=self.head_dim,
                    elementwise_affine=elementwise_affine,
                    eps=norm_eps,
                )
                self.fuse_norm_and_gate = True
            else:
                self.fuse_norm_and_gate = False
                self.g_norm = RMSNorm(
                    hidden_size=self.head_dim,
                    elementwise_affine=elementwise_affine,
                    eps=norm_eps,
                )
                self.gate_fn = ACT2FN[gate_fn]

            self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        if not self.use_nope:
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            if not self.use_l2_softmax:
                o = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                )
            else:
                if torch.equal(cu_seqlens_q, cu_seqlens_k):
                    o = parallel_attn(
                        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), cu_seqlens=cu_seqlens_k, use_l2=True,
                    ).squeeze(0)
                else:
                    o = attn_decoding_one_step(
                        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), cu_seqlens=cu_seqlens_k, use_l2=True,
                    ).squeeze(0)
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            if self.use_l2_softmax:
                # TODO (alex): haven't been tested yet.
                o = parallel_attn(
                    q, k, v, cu_seqlens=cu_seqlens, use_l2=True,
                ).squeeze(0)
            else:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                ).unsqueeze(0)
        else:
            if self.use_l2_softmax:
                o = parallel_attn(
                    q, k, v,
                    use_l2=True
                )
            else:
                o = flash_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
                )
        
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
                o = o * self.gate_fn(g)
        else:
            o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
