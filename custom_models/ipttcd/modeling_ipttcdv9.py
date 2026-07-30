# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from custom_models.ipttcd.configuration_ipttcdv9 import IPTTCDv9Config
import fla.layers.attn as _fla_attn
from fla.layers.attn import Attention
from fla.models.transformer.modeling_transformer import TransformerPreTrainedModel
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.activations import swiglu, swiglu_linear
from fla.modules.l2warp import l2_warp

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except Exception:
    DTensor, distribute_tensor = None, None

try:
    from flash_attn_interface import (
        flash_attn_func as _fa3_flash_attn_func,
        flash_attn_varlen_func as _fa3_flash_attn_varlen_func,
    )

    _FA3_AVAILABLE = True
except Exception:
    _FA3_AVAILABLE = False
    _fa3_flash_attn_func = None
    _fa3_flash_attn_varlen_func = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


@torch.compile()
def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """
    Muon-style Newton-Schulz orthogonalization for a batch of matrices.

    Args:
        G: [B, D1, D2]
    Returns:
        [B, D1, D2]
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    # Ensure spectral norm is bounded before NS iterations.
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.transpose(1, 2)
    return X


class IPTTCDMLP(nn.Module):

    def __init__(self, config: IPTTCDv9Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_ratio = config.hidden_ratio
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        self.fuse_swiglu = config.fuse_swiglu

        if self.hidden_ratio is None:
            self.hidden_ratio = 4
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * self.hidden_ratio * 2 / 3)
            self.intermediate_size = 256 * ((self.intermediate_size + 256 - 1) // 256)
        if self.hidden_act != "swish":
            raise ValueError(f"Unsupported hidden_act: {self.hidden_act}")

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.is_ttt_layer = bool(config.ttt_mode and layer_idx in set(config.ttt_layers))
        if self.is_ttt_layer:
            self.ttt_chunk = config.ttt_chunk
            self.ttt_lr = float(config.ttt_lr)
            self.ttt_update_every = max(1, int(getattr(config, "ttt_update_every", 1)))
            self.ttt_weight_renorm = bool(getattr(config, "ttt_weight_renorm", False))
            self.ttt_norm_log_freq = max(1, int(getattr(config, "ttt_norm_log_freq", 50)))
            self.ttt_proj = (
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                if config.ttt_proj
                else None
            )
            if self.ttt_proj is not None:
                self.ttt_proj._is_ttt_proj = True
            self.ttt_conv_kernel_size = int(getattr(config, "ttt_conv_kernel_size", 5))
            self.ttt_conv_causal = bool(getattr(config, "ttt_conv_causal", True))
            self.ttt_output_use_teacher_conv = bool(
                getattr(config, "ttt_output_use_teacher_conv", True)
            )
            self.ttt_output_source = str(getattr(config, "ttt_output_source", "teacher")).lower()
            conv_padding = 0 if self.ttt_conv_causal else self.ttt_conv_kernel_size // 2
            self.ttt_teacher_conv = (
                nn.Conv1d(
                    self.intermediate_size,
                    self.intermediate_size,
                    kernel_size=self.ttt_conv_kernel_size,
                    padding=conv_padding,
                    groups=self.intermediate_size,
                    bias=False,
                )
                if bool(getattr(config, "ttt_teacher_conv", False))
                else None
            )
            if self.ttt_teacher_conv is not None:
                self.ttt_teacher_conv._is_ttt_conv = True
                self.ttt_teacher_conv._ttt_conv_role = "teacher"
            self.ttt_student_conv = (
                nn.Conv1d(
                    self.intermediate_size,
                    self.intermediate_size,
                    kernel_size=self.ttt_conv_kernel_size,
                    padding=conv_padding,
                    groups=self.intermediate_size,
                    bias=False,
                )
                if bool(getattr(config, "ttt_student_conv", False))
                else None
            )
            if self.ttt_student_conv is not None:
                self.ttt_student_conv._is_ttt_conv = True
                self.ttt_student_conv._ttt_conv_role = "student"
            # Token-wise dynamic learning rate: β_t = sigmoid(w_β · h'_t)
            # Use a 1D parameter (hidden_size,) instead of nn.Linear(hidden_size, 1)
            # to avoid FSDP sharding issues (dim-0 size 1 is not divisible by num_ranks).
            self.dynamic_lr = config.dynamic_lr
            if self.dynamic_lr:
                self.b_proj = nn.Parameter(torch.empty(self.hidden_size))
            # Optional Muon-style update preconditioner over d_W.
            self.use_muon = bool(config.use_muon)
            self.ttt_norm_proj_diff = bool(getattr(config, "ttt_norm_proj_diff", False))
            self.ttt_normalize_student_features = bool(
                getattr(config, "ttt_normalize_student_features", True)
            )
            self.ttt_fuse_proj_into_update = bool(
                getattr(config, "ttt_fuse_proj_into_update", True)
            )
            self.ttt_force_grouped_scan = bool(
                getattr(config, "ttt_force_grouped_scan", False)
            )
            self.ttt_scan_group_size = max(1, int(getattr(config, "ttt_scan_group_size", 4)))
            self.ttt_scan_state_dtype = str(
                getattr(config, "ttt_scan_state_dtype", "compute")
            ).lower()

    def _padding(self, x: torch.Tensor) -> torch.Tensor:
        """Pad sequence length to a multiple of ttt_chunk and reshape to (b, T, C, d)."""
        if not self.is_ttt_layer:
            return x
        if x.shape[1] % self.ttt_chunk != 0:
            pad_tokens = self.ttt_chunk - x.shape[1] % self.ttt_chunk
            x = F.pad(x, (0, 0, 0, pad_tokens))
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    @staticmethod
    def _renorm_weight_states(
        weight_states: torch.Tensor,
        target_frob_norm: torch.Tensor,
    ) -> torch.Tensor:
        cur_frob_norm = (
            weight_states.flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
        )
        return weight_states * (target_frob_norm / cur_frob_norm)

    def _apply_ttt_conv(self, x: torch.Tensor, conv: nn.Conv1d | None) -> torch.Tensor:
        if conv is None:
            return x
        b, t, c, d = x.shape
        x_flat = rearrange(x, "b t c d -> (b t) d c")
        if self.ttt_conv_causal:
            x_flat = F.pad(x_flat, (conv.kernel_size[0] - 1, 0))
        x_flat = conv(x_flat)
        return rearrange(x_flat, "(b t) d c -> b t c d", b=b, t=t, d=d)

    def _select_ttt_output_padded(
        self,
        teacher_output_padded: torch.Tensor,
        student_output_padded: torch.Tensor,
    ) -> torch.Tensor:
        if self.ttt_output_source == "teacher":
            return teacher_output_padded
        if self.ttt_output_source == "student":
            return student_output_padded
        raise ValueError(
            f"Unsupported `ttt_output_source`: {self.ttt_output_source!r}. "
            "Expected 'teacher' or 'student'."
        )

    def _get_ttt_scan_state_dtype(self, compute_dtype: torch.dtype) -> torch.dtype:
        if self.ttt_scan_state_dtype == "compute":
            return compute_dtype
        if self.ttt_scan_state_dtype == "bf16":
            return torch.bfloat16
        if self.ttt_scan_state_dtype == "fp32":
            return torch.float32
        raise ValueError(
            f"Unsupported `ttt_scan_state_dtype`: {self.ttt_scan_state_dtype!r}."
        )

    def _get_ttt_scan_group_size(self, num_chunks: int) -> int:
        return min(num_chunks, self.ttt_update_every)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        """
        Args:
            x: Teacher hidden states (full-context attention output), (b, S, d_h).
            t: Student hidden states (truncated-context attention output), (b, S, d_h).
               None when not training or not a TTT layer.
        """
        with record_function(f"ipttcdv9.mlp.teacher_gateup.layer{self.layer_idx}"):
            gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu and (t is None or not self.is_ttt_layer):
            return swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)

        # Teacher swiglu output: z = phi(W_gate x) * W_up x
        with record_function(f"ipttcdv9.mlp.teacher_swiglu.layer{self.layer_idx}"):
            z = swiglu(gate, up)                                     # (b, S, d_i)
        if t is None or not self.is_ttt_layer:
            return self.down_proj(z)

        # ==== In-Place Context Distillation (IPTTCD v9) ====
        #
        # v9 makes the fast-weight update off-policy on purpose:
        #   - Update path always projects through the fixed base weight W_base.
        #   - Output path still uses the running adapted weight W_cur.
        #
        # This means per-chunk updates no longer depend on the evolving W_cur, so
        # we can compute all chunk updates in parallel and recover the running
        # output weights with a single cumsum scan, just like IPTTT.
        #
        # Notation (d_h = hidden_size, d_i = intermediate_size, C = chunk_size):
        #   z        = swiglu(W_gate x, W_up x)     -- teacher features (full context)
        #   z_hat    = swiglu(W_gate t, W_up t)     -- student features (truncated context)
        #   z_target = conv(z) if conv else z       -- distillation target
        #
        # Per-chunk update (fixed W_base):
        #   diff_i      = z_target_i - z_hat_i                       (b, C, d_i)
        #   proj_diff_i = diff_i @ W_base^T                          (b, C, d_h)
        #   beta_i      = sigmoid(W_beta · t_i)  [if dynamic_lr]     (b, C, 1)
        #   ΔW_i        = (beta_i * proj_diff_i)^T @ norm(z_hat_i)   (b, d_h, d_i)
        #
        # Running output state:
        #   W_cur^(0)   = W_base
        #   W_cur^(i+1) = W_cur^(i) + η · ΔW_i
        #   Y_i         = z_output_i @ (W_cur^(i))^T

        seq_len = x.shape[1]

        # Teacher/student Z paths with optional chunk-local causal short conv.
        with record_function(f"ipttcdv9.mlp.teacher_ttt_prepare.layer{self.layer_idx}"):
            z_padded = self._padding(z)                              # (b, T, C, d_i)
            b, T, C, _ = z_padded.shape
            compute_dtype = z_padded.dtype
            z_target_padded = self._apply_ttt_conv(z_padded, self.ttt_teacher_conv)
            teacher_output_padded = (
                z_target_padded if self.ttt_output_use_teacher_conv else z_padded
            )

        # Student: pad & chunk, then compute swiglu (no conv)
        with record_function(f"ipttcdv9.mlp.student_gateup.layer{self.layer_idx}"):
            t_padded = self._padding(t)                              # (b, T, C, d_h)
            t_flat = t_padded.reshape(b, T * C, -1)                 # (b, T*C, d_h)
            z_hat = swiglu(self.gate_proj(t_flat), self.up_proj(t_flat))
            z_hat_padded = z_hat.reshape(b, T, C, -1)               # (b, T, C, d_i)
            z_hat_padded = self._apply_ttt_conv(z_hat_padded, self.ttt_student_conv)
        z_output_padded = self._select_ttt_output_padded(
            teacher_output_padded=teacher_output_padded,
            student_output_padded=z_hat_padded,
        )
        beta_padded = None
        if self.dynamic_lr:
            with record_function(f"ipttcdv9.mlp.student_beta.layer{self.layer_idx}"):
                # Compute token-wise beta for all chunks once.
                beta_padded = (t_padded @ self.b_proj).unsqueeze(-1).sigmoid()  # (b, T, C, 1)

        # Keep a fp32 master copy for numerically stable fast-weight adaptation.
        # W_base stays fixed for the update rule; W_cur only affects the output path.
        W_base = self.down_proj.weight.float().unsqueeze(0).expand(b, -1, -1)
        W_base_unsq = W_base.unsqueeze(1)
        W_base_frob_norm = None
        if self.ttt_weight_renorm:
            W_base_frob_norm = W_base.flatten(1).norm(dim=-1, keepdim=True).view(b, 1, 1)
        W_base_t = W_base.to(dtype=compute_dtype).transpose(1, 2).unsqueeze(1)

        ttt_proj_weight_unsq = None
        if self.ttt_proj is not None:
            ttt_proj_weight_unsq = (
                self.ttt_proj.weight.unsqueeze(0).expand(b, -1, -1).to(dtype=compute_dtype).unsqueeze(1)
            )
        ttt_lr = W_base.new_tensor(self.ttt_lr)
        update_every = self._get_ttt_scan_group_size(T)
        scan_state_dtype = self._get_ttt_scan_state_dtype(compute_dtype)

        compute_norm_metrics = False
        if self.training:
            self._ttt_norm_forward_count = getattr(self, "_ttt_norm_forward_count", 0) + 1
            compute_norm_metrics = (
                self._ttt_norm_forward_count % self.ttt_norm_log_freq == 0
            )

        if compute_norm_metrics:
            z_hat_avg_sum = 0.0
            proj_diff_avg_sum = 0.0
            d_w_avg_sum = 0.0
            z_hat_max_norm = 0.0
            proj_diff_max_norm = 0.0
            d_w_max_norm = 0.0
            norm_metric_count = 0
        else:
            self._last_ttt_norm_metrics = {}

        fuse_ttt_proj_into_update = (
            self.ttt_fuse_proj_into_update
            and
            ttt_proj_weight_unsq is not None
            and not self.ttt_norm_proj_diff
            and not compute_norm_metrics
        )
        if fuse_ttt_proj_into_update:
            with record_function(f"ipttcdv9.mlp.fuse_base_proj.layer{self.layer_idx}"):
                W_update_t = torch.matmul(W_base_t, ttt_proj_weight_unsq)
        else:
            W_update_t = W_base_t

        if not self.ttt_weight_renorm:
            with record_function(f"ipttcdv9.mlp.update_scan.layer{self.layer_idx}"):
                # Exact IPTTT-style full scan: compute all updates against fixed W_base
                # once, prepend the base weight, then recover all prefix states with
                # one cumsum along the chunk axis.
                if T > 1:
                    z_target_upd = z_target_padded[:, :-1]            # (b, T-1, C, d_i)
                    z_hat_upd = z_hat_padded[:, :-1]                  # (b, T-1, C, d_i)
                    diff = z_target_upd - z_hat_upd
                    proj_diff = torch.matmul(diff, W_update_t)
                    if self.ttt_norm_proj_diff:
                        proj_diff = F.normalize(proj_diff, p=2, dim=-1, eps=1e-6)

                    if compute_norm_metrics:
                        with torch.no_grad():
                            z_hat_token_norm = z_hat_upd.detach().float().norm(dim=-1)
                            norm_count = T - 1
                            z_hat_avg_sum += float(z_hat_token_norm.mean().item()) * norm_count
                            z_hat_max_norm = max(z_hat_max_norm, float(z_hat_token_norm.max().item()))

                    z_hat_norm = (
                        F.normalize(z_hat_upd, p=2, dim=-1, eps=1e-6)
                        if self.ttt_normalize_student_features
                        else z_hat_upd
                    )

                    if self.dynamic_lr:
                        proj_diff = proj_diff * beta_padded[:, :-1]

                    if compute_norm_metrics:
                        with torch.no_grad():
                            proj_diff_token_norm = proj_diff.detach().float().norm(dim=-1)
                            proj_diff_avg_sum += float(proj_diff_token_norm.mean().item()) * norm_count
                            proj_diff_max_norm = max(
                                proj_diff_max_norm,
                                float(proj_diff_token_norm.max().item()),
                            )

                    if ttt_proj_weight_unsq is not None and not fuse_ttt_proj_into_update:
                        proj_diff = torch.matmul(proj_diff, ttt_proj_weight_unsq)

                    d_w_chunks = torch.matmul(
                        proj_diff.float().transpose(-1, -2),
                        z_hat_norm.float(),
                    )

                    if compute_norm_metrics:
                        with torch.no_grad():
                            d_w_batch_norm = d_w_chunks.detach().float().flatten(2).norm(dim=-1)
                            d_w_avg_sum += float(d_w_batch_norm.mean().item()) * norm_count
                            d_w_max_norm = max(d_w_max_norm, float(d_w_batch_norm.max().item()))
                            norm_metric_count += norm_count

                    if self.training:
                        self._last_delta_w_norm = d_w_chunks[:, -1].detach().norm().item()

                    if self.use_muon:
                        d_w_shape = d_w_chunks.shape
                        d_w_chunks = zeropower_via_newtonschulz5(
                            d_w_chunks.reshape(-1, d_w_shape[-2], d_w_shape[-1])
                        ).reshape(d_w_shape)

                    w_states = torch.cat([W_base_unsq, ttt_lr * d_w_chunks], dim=1)
                    w_run = torch.cumsum(w_states, dim=1)
                else:
                    w_run = W_base_unsq

            with record_function(f"ipttcdv9.mlp.output_proj.layer{self.layer_idx}"):
                output = torch.matmul(
                    z_output_padded,
                    w_run.to(dtype=compute_dtype).transpose(-1, -2),
                )
        else:
            with record_function(f"ipttcdv9.mlp.grouped_scan.layer{self.layer_idx}"):
                W_cur = W_base
                output = z_output_padded.new_empty(b, T, C, self.hidden_size)
                for gs in range(0, T, update_every):
                    ge = min(gs + update_every, T)
                    Kg = ge - gs
                    upd_end = min(ge, T - 1)
                    K_upd = upd_end - gs

                    if K_upd > 0:
                        z_target_g = z_target_padded[:, gs:upd_end]      # (b, K_upd, C, d_i)
                        z_hat_g = z_hat_padded[:, gs:upd_end]            # (b, K_upd, C, d_i)
                        diff_g = z_target_g - z_hat_g
                        proj_diff_g = torch.matmul(diff_g, W_update_t)
                        if self.ttt_norm_proj_diff:
                            proj_diff_g = F.normalize(proj_diff_g, p=2, dim=-1, eps=1e-6)

                        if compute_norm_metrics:
                            with torch.no_grad():
                                z_hat_token_norm = z_hat_g.detach().float().norm(dim=-1)
                                z_hat_avg_sum += float(z_hat_token_norm.mean().item()) * K_upd
                                z_hat_max_norm = max(z_hat_max_norm, float(z_hat_token_norm.max().item()))

                        z_hat_g_norm = (
                            F.normalize(z_hat_g, p=2, dim=-1, eps=1e-6)
                            if self.ttt_normalize_student_features
                            else z_hat_g
                        )

                        if self.dynamic_lr:
                            proj_diff_g = proj_diff_g * beta_padded[:, gs:upd_end]

                        if compute_norm_metrics:
                            with torch.no_grad():
                                proj_diff_token_norm = proj_diff_g.detach().float().norm(dim=-1)
                                proj_diff_avg_sum += float(proj_diff_token_norm.mean().item()) * K_upd
                                proj_diff_max_norm = max(
                                    proj_diff_max_norm,
                                    float(proj_diff_token_norm.max().item()),
                                )

                        if ttt_proj_weight_unsq is not None and not fuse_ttt_proj_into_update:
                            proj_diff_g = torch.matmul(proj_diff_g, ttt_proj_weight_unsq)

                        d_w_chunks = torch.matmul(
                            proj_diff_g.float().transpose(-1, -2),
                            z_hat_g_norm.float(),
                        )

                        if compute_norm_metrics:
                            with torch.no_grad():
                                d_w_batch_norm = d_w_chunks.detach().float().flatten(2).norm(dim=-1)
                                d_w_avg_sum += float(d_w_batch_norm.mean().item()) * K_upd
                                d_w_max_norm = max(d_w_max_norm, float(d_w_batch_norm.max().item()))
                                norm_metric_count += K_upd

                        if self.training:
                            self._last_delta_w_norm = d_w_chunks[:, -1].detach().norm().item()

                        if self.use_muon:
                            d_w_shape = d_w_chunks.shape
                            d_w_chunks = zeropower_via_newtonschulz5(
                                d_w_chunks.reshape(-1, d_w_shape[-2], d_w_shape[-1])
                            ).reshape(d_w_shape)

                        cs = ttt_lr * torch.cumsum(d_w_chunks, dim=1)
                    else:
                        cs = None

                    w_run = W_cur.to(dtype=scan_state_dtype).unsqueeze(1).expand(-1, Kg, -1, -1).clone()
                    if K_upd > 0:
                        cs_state = cs.to(dtype=scan_state_dtype)
                        fill = min(Kg, K_upd + 1)
                        if fill > 1:
                            w_run[:, 1:fill] = w_run[:, 1:fill] + cs_state[:, :fill - 1]
                        if fill < Kg:
                            w_run[:, fill:] = w_run[:, fill:] + cs_state[:, -1:].expand(-1, Kg - fill, -1, -1)
                        if self.ttt_weight_renorm:
                            w_run = self._renorm_weight_states(
                                w_run.float(),
                                W_base_frob_norm.unsqueeze(1),
                            ).to(dtype=scan_state_dtype)
                            W_cur = self._renorm_weight_states(W_cur + cs[:, -1], W_base_frob_norm)
                        else:
                            W_cur = W_cur + cs[:, -1]

                    output[:, gs:ge] = torch.matmul(
                        z_output_padded[:, gs:ge],
                        w_run.to(dtype=compute_dtype).transpose(-1, -2),
                    )

        if compute_norm_metrics and norm_metric_count > 0:
            self._last_ttt_norm_metrics = {
                "z_hat_avg_norm": z_hat_avg_sum / norm_metric_count,
                "z_hat_max_norm": z_hat_max_norm,
                "proj_diff_avg_norm": proj_diff_avg_sum / norm_metric_count,
                "proj_diff_max_norm": proj_diff_max_norm,
                "d_w_avg_norm": d_w_avg_sum / norm_metric_count,
                "d_w_max_norm": d_w_max_norm,
            }
        elif compute_norm_metrics:
            self._last_ttt_norm_metrics = {}

        return rearrange(output, "b t c d -> b (t c) d")[:, :seq_len, :]


_INFER_SIDE_STREAM = None


def _get_infer_side_stream():
    """Lazy per-process side stream for overlapping the student attention tail
    with the teacher attention during inference prefill. Returns None (fully
    sequential fast path) when CUDA is absent or overlap is disabled."""
    global _INFER_SIDE_STREAM
    if not torch.cuda.is_available() or os.environ.get("IPTTCD_DISABLE_PREFILL_STREAM") == "1":
        return None
    if _INFER_SIDE_STREAM is None:
        _INFER_SIDE_STREAM = torch.cuda.Stream()
    return _INFER_SIDE_STREAM


def _use_fast_prefill_path(blk, hidden_states, attention_mask, kwargs) -> bool:
    """Shared-QKV prefill fast path applies when there is no padding and no
    varlen packing; otherwise the dual-path fallback (which routes the teacher
    through the unpad-aware Attention module) is used."""
    if os.environ.get("IPTTCD_DISABLE_FAST_PREFILL") == "1":
        return False
    if hidden_states.shape[1] < blk.ttt_chunk:
        return False
    if kwargs.get("cu_seqlens") is not None:
        return False
    if attention_mask is not None and not bool(attention_mask.all()):
        return False
    return True


class IPTTCDBlock(GradientCheckpointingLayer):

    def __init__(self, config: IPTTCDv9Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.is_ttt_layer = bool(config.ttt_mode and layer_idx in set(config.ttt_layers))
        if self.is_ttt_layer:
            self.ttt_chunk = config.ttt_chunk
            self.ttt_visible_chunks = config.ttt_visible_chunks

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            use_output_gate=config.use_output_gate,
            gate_fn=config.gate_fn,
            elementwise_affine=config.elementwise_affine,
            gate_logit_normalizer=config.gate_logit_normalizer,
            norm_eps=config.norm_eps,
            fuse_norm=config.fuse_norm,
            use_l2_softmax=config.use_l2_softmax,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = IPTTCDMLP(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        student_output = None
        attentions = None
        
        # # Teacher attention, always full context
        # teacher_attn_output, attentions, past_key_values = self.attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     **kwargs,
        # )
        
        # # Student attention, chunk-local attention
        # student_output = None
        # if self.is_ttt_layer and self.training:
        #     saved_window = self.attn.window_size
        #     self.attn.window_size = self.ttt_visible_chunks * self.ttt_chunk
        #     try:
        #         student_attn_output, _, _ = self.attn(
        #             hidden_states=hidden_states,
        #             attention_mask=attention_mask,
        #             past_key_values=None,
        #             use_cache=False,
        #             output_attentions=False,
        #         )
        #     finally:
        #         self.attn.window_size = saved_window
            
        #     if self.config.fuse_norm:
        #         student_output, _ = self.mlp_norm(student_attn_output, residual, True)
        #     else:
        #         student_output = self.mlp_norm(residual + student_attn_output)

        if self.is_ttt_layer and self.training:
            # === Training: Shared-QKV fast path (no KV cache) ===
            # Compute Q, K, V projections and RoPE once; run flash_attn twice
            # with different window sizes for teacher (full/SWA) and student (chunk-local).
            attn = self.attn
            batch_size, q_len, _ = hidden_states.size()

            q = rearrange(attn.q_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            k = rearrange(attn.k_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            v = rearrange(attn.v_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)

            if attn.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)

            seqlen_offset, max_seqlen = 0, q_len
            if attn.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, attn.max_position_embeddings)
            if not attn.use_nope:
                q, k = attn.rotary(q, k, seqlen_offset=seqlen_offset,
                                   max_seqlen=max_seqlen,
                                   cu_seqlens=kwargs.get('cu_seqlens'))

            teacher_window = (-1, -1) if attn.window_size is None else (attn.window_size - 1, 0)
            student_window_size = self.ttt_visible_chunks * self.ttt_chunk

            teacher_o = _fla_attn.flash_attn_func(q, k, v, causal=True, window_size=teacher_window)
            student_o = _fla_attn.flash_attn_func(q, k, v, causal=True,
                                                   window_size=(student_window_size - 1, 0))

            if attn.use_output_gate:
                g = attn.g_proj(hidden_states)
                if attn.fuse_norm_and_gate:
                    g = rearrange(g, '... (h d) -> ... h d', d=attn.head_dim)
                    teacher_o = rearrange(attn.g_norm_swish_gate(teacher_o, g), '... h d -> ... (h d)')
                    student_o = rearrange(attn.g_norm_swish_gate(student_o, g), '... h d -> ... (h d)')
                else:
                    teacher_o = rearrange(attn.g_norm(teacher_o), '... h d -> ... (h d)')
                    student_o = rearrange(attn.g_norm(student_o), '... h d -> ... (h d)')
                    gate = attn.gate_fn(g)
                    teacher_o = teacher_o * gate
                    student_o = student_o * gate
            else:
                teacher_o = teacher_o.reshape(batch_size, q_len, -1)
                student_o = student_o.reshape(batch_size, q_len, -1)

            teacher_attn_output = attn.o_proj(teacher_o)
            student_attn_output = attn.o_proj(student_o)

            if self.config.fuse_norm:
                student_output, _ = self.mlp_norm(student_attn_output, residual, True)
            else:
                student_output = self.mlp_norm(residual + student_attn_output)
        elif self.is_ttt_layer and _use_fast_prefill_path(self, hidden_states, attention_mask, kwargs):
            # === Inference prefill fast path: shared-QKV teacher+student ===
            # Q/K/V and RoPE are computed once; the teacher (windowed, cached)
            # and student (chunk-local) attentions share them, with the student
            # tail overlapped on a side CUDA stream. Bit-exact vs the fallback
            # dual path below (verified against step-10000 ckpt logits).
            # Disable with IPTTCD_DISABLE_FAST_PREFILL=1.
            attn = self.attn
            batch_size, q_len, _ = hidden_states.size()

            q = rearrange(attn.q_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            k = rearrange(attn.k_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            v = rearrange(attn.v_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            if attn.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)

            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(attn.layer_idx)
                max_seqlen = q_len + seqlen_offset
            if attn.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, attn.max_position_embeddings)
            if not attn.use_nope:
                q, k = attn.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen,
                                   cu_seqlens=kwargs.get('cu_seqlens'))

            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(attn.layer_idx) > 0
                k_c, v_c = past_key_values.update(
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    layer_idx=attn.layer_idx, offset=q_len,
                    cache_kwargs=dict(window_size=attn.window_size),
                )['attn_state']
                if cache_has_content:
                    k = rearrange(k_c, '... (h d) -> ... h d', d=attn.head_dim)
                    v = rearrange(v_c, '... (h d) -> ... h d', d=attn.head_dim)

            teacher_window = (-1, -1) if attn.window_size is None else (attn.window_size - 1, 0)
            student_window = self.ttt_visible_chunks * self.ttt_chunk
            attentions = None

            def _postprocess(o):
                if attn.use_output_gate:
                    g = attn.g_proj(hidden_states)
                    if attn.fuse_norm_and_gate:
                        gg = rearrange(g, '... (h d) -> ... h d', d=attn.head_dim)
                        return rearrange(attn.g_norm_swish_gate(o, gg), '... h d -> ... (h d)')
                    o = rearrange(attn.g_norm(o), '... h d -> ... (h d)')
                    return o * attn.gate_fn(g)
                return o.reshape(batch_size, q_len, -1)

            def _student_tail(residual_in):
                s_o = _fla_attn.flash_attn_func(q, k, v, causal=True,
                                                window_size=(student_window - 1, 0))
                s_attn = attn.o_proj(_postprocess(s_o))
                if self.config.fuse_norm:
                    out, _ = self.mlp_norm(s_attn, residual_in, True)
                else:
                    out = self.mlp_norm(residual_in + s_attn)
                return out

            side = _get_infer_side_stream()
            if side is not None:
                ev_qkv = torch.cuda.Event()
                ev_qkv.record()
                with torch.cuda.stream(side):
                    side.wait_event(ev_qkv)
                    for tsr in (q, k, v, residual):
                        tsr.record_stream(side)
                    student_output = _student_tail(residual)
                    ev_student = torch.cuda.Event()
                    ev_student.record(side)
                teacher_attn_output = attn.o_proj(_postprocess(
                    _fla_attn.flash_attn_func(q, k, v, causal=True, window_size=teacher_window)))
                torch.cuda.current_stream().wait_event(ev_student)
            else:
                teacher_attn_output = attn.o_proj(_postprocess(
                    _fla_attn.flash_attn_func(q, k, v, causal=True, window_size=teacher_window)))
                student_output = _student_tail(residual)
        elif self.is_ttt_layer:
            # === Inference on TTT layer (fallback: padded batches / cu_seqlens
            # / fast path disabled): teacher via cached self.attn, student
            # separately via chunk-local flash_attn. Matches paper's design
            # (Theorem 1 in ipttcd.tex) where the fast-weight update W_cur is
            # computed at test time from (teacher - student) signals.
            attn = self.attn
            batch_size, q_len, _ = hidden_states.size()

            # Teacher path: full (or SWA) attention with KV cache handling.
            teacher_attn_output, attentions, past_key_values = attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            # Student path: chunk-local attention, no cache. TTT is only
            # meaningful when q_len >= ttt_chunk (i.e. during prefill); at
            # single-token decode steps we emit a zero-sized student signal so
            # the MLP falls back to the plain SwiGLU path (it already checks
            # seq_len < ttt_chunk).
            if q_len >= self.ttt_chunk:
                q = rearrange(attn.q_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
                k = rearrange(attn.k_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
                v = rearrange(attn.v_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
                if attn.qk_norm:
                    q, k = attn.q_norm(q), attn.k_norm(k)
                max_seqlen = q_len
                if attn.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, attn.max_position_embeddings)
                if not attn.use_nope:
                    q, k = attn.rotary(q, k, seqlen_offset=0,
                                       max_seqlen=max_seqlen,
                                       cu_seqlens=kwargs.get('cu_seqlens'))
                student_window_size = self.ttt_visible_chunks * self.ttt_chunk
                student_o = _fla_attn.flash_attn_func(
                    q, k, v, causal=True, window_size=(student_window_size - 1, 0),
                )
                if attn.use_output_gate:
                    g = attn.g_proj(hidden_states)
                    if attn.fuse_norm_and_gate:
                        g = rearrange(g, '... (h d) -> ... h d', d=attn.head_dim)
                        student_o = rearrange(attn.g_norm_swish_gate(student_o, g), '... h d -> ... (h d)')
                    else:
                        student_o = rearrange(attn.g_norm(student_o), '... h d -> ... (h d)')
                        student_o = student_o * attn.gate_fn(g)
                else:
                    student_o = student_o.reshape(batch_size, q_len, -1)
                student_attn_output = attn.o_proj(student_o)
                if self.config.fuse_norm:
                    student_output, _ = self.mlp_norm(student_attn_output, residual, True)
                else:
                    student_output = self.mlp_norm(residual + student_attn_output)
            # else: student_output stays None; MLP will skip TTT path.
        else:
            # Non-TTT layer: standard attention path
            teacher_attn_output, attentions, past_key_values = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(teacher_attn_output, residual, True)
        else:
            hidden_states = residual + teacher_attn_output
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        truncated_t = student_output if self.is_ttt_layer else None
        hidden_states = self.mlp(hidden_states, t=truncated_t, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


class IPTTCDPreTrainedModel(TransformerPreTrainedModel):
    config_class = IPTTCDv9Config
    _no_split_modules = ["IPTTCDBlock"]

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        super()._init_weights(
            module,
            rescale_prenorm_residual=rescale_prenorm_residual,
            num_residuals_per_layer=num_residuals_per_layer,
        )
        if isinstance(module, nn.Linear) and getattr(module, "_is_ttt_proj", False):
            ttt_proj_init = getattr(self.config, "ttt_proj_init", "diagonal")
            if ttt_proj_init == "diagonal":
                self._init_ttt_proj_as_diagonal(module.weight, self.config.initializer_range)
            elif ttt_proj_init == "diagonal_gaussian":
                self._init_ttt_proj_as_diagonal_gaussian(module.weight, self.config.initializer_range)
            elif ttt_proj_init == "zero":
                with torch.no_grad():
                    module.weight.zero_()
        if isinstance(module, nn.Conv1d) and getattr(module, "_is_ttt_conv", False):
            role = getattr(module, "_ttt_conv_role", "ttt")
            init = getattr(self.config, f"ttt_{role}_conv_init", getattr(self.config, "ttt_conv_init", "zero"))
            if init == "identity":
                self._init_ttt_conv_as_identity(module)
            elif init == "zero":
                with torch.no_grad():
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
            elif init == "random":
                pass
            else:
                raise ValueError(
                    f"Unsupported TTT conv init {init!r} for role {role!r}; expected 'identity', 'zero', or 'random'."
                )
        if isinstance(module, IPTTCDMLP) and module.is_ttt_layer:
            with torch.no_grad():
                if module.dynamic_lr:
                    nn.init.normal_(module.b_proj, mean=0.0, std=self.config.initializer_range)

    @staticmethod
    def _init_ttt_proj_as_diagonal(weight: torch.Tensor, scale: float = 1.0) -> None:
        with torch.no_grad():
            if DTensor is not None and distribute_tensor is not None and isinstance(weight, DTensor):
                local_weight = weight.to_local()
                eye = torch.eye(
                    weight.shape[0],
                    weight.shape[1],
                    device=local_weight.device,
                    dtype=local_weight.dtype,
                )
                if scale != 1.0:
                    eye = eye * scale
                weight.copy_(distribute_tensor(eye, weight.device_mesh, weight.placements))
            else:
                nn.init.eye_(weight)
                if scale != 1.0:
                    weight.mul_(scale)

    @staticmethod
    def _init_ttt_proj_as_diagonal_gaussian(weight: torch.Tensor, scale: float = 1.0) -> None:
        with torch.no_grad():
            diag_size = min(weight.shape[0], weight.shape[1])
            if DTensor is not None and distribute_tensor is not None and isinstance(weight, DTensor):
                local_weight = weight.to_local()
                diagonal_weight = torch.zeros(
                    weight.shape[0],
                    weight.shape[1],
                    device=local_weight.device,
                    dtype=local_weight.dtype,
                )
                diag_values = torch.randn(
                    diag_size,
                    device=local_weight.device,
                    dtype=local_weight.dtype,
                )
                diag_indices = torch.arange(diag_size, device=local_weight.device)
                diagonal_weight[diag_indices, diag_indices] = diag_values * scale
                weight.copy_(
                    distribute_tensor(diagonal_weight, weight.device_mesh, weight.placements)
                )
            else:
                weight.zero_()
                diag_values = torch.randn(
                    diag_size,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                diag_indices = torch.arange(diag_size, device=weight.device)
                weight[diag_indices, diag_indices] = diag_values * scale

    @staticmethod
    def _init_ttt_conv_as_identity(module: nn.Conv1d) -> None:
        with torch.no_grad():
            kernel_idx = module.kernel_size[0] - 1 if module.padding[0] == 0 else module.kernel_size[0] // 2
            if DTensor is not None and distribute_tensor is not None and isinstance(module.weight, DTensor):
                local_weight = module.weight.to_local()
                identity_weight = torch.zeros(
                    module.weight.shape,
                    device=local_weight.device,
                    dtype=local_weight.dtype,
                )
                identity_weight[:, 0, kernel_idx] = 1.0
                module.weight.copy_(
                    distribute_tensor(identity_weight, module.weight.device_mesh, module.weight.placements)
                )
            else:
                module.weight.zero_()
                module.weight[:, 0, kernel_idx] = 1.0
            if module.bias is not None:
                module.bias.zero_()


class IPTTCDModel(IPTTCDPreTrainedModel):

    def __init__(
        self,
        config: IPTTCDv9Config,
    ) -> IPTTCDModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Patch fla.layers.attn to use Flash Attention 3 (Hopper) if requested
        if getattr(config, "use_flash_attn_3", False):
            if not _FA3_AVAILABLE:
                raise ImportError(
                    "Flash Attention 3 is not available. "
                    "Please install it: cd flash-attention/hopper && python setup.py install"
                )
            import fla.layers.attn as _attn_module

            _attn_module.flash_attn_func = _fa3_flash_attn_func
            _attn_module.flash_attn_varlen_func = _fa3_flash_attn_varlen_func
            logger.warning_once(
                "Flash Attention 3 (Hopper) enabled — patched fla.layers.attn"
            )

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [IPTTCDBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`IPTTCDModel` does not support output attention weights now, so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class IPTTCDForCausalLM(IPTTCDPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = IPTTCDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_ttt_norm_metrics(self):
        layer_metrics = []
        per_layer_out = {}
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            metrics = getattr(mlp, "_last_ttt_norm_metrics", None)
            if metrics:
                layer_metrics.append(metrics)
                layer_idx = getattr(layer, "layer_idx", None)
                if layer_idx is None:
                    layer_idx = len(layer_metrics) - 1
                for k, v in metrics.items():
                    per_layer_out[f"ttt_norm/layer_{layer_idx}/{k}"] = v

        if not layer_metrics:
            return {}

        out = dict(per_layer_out)
        avg_keys = ("z_hat_avg_norm", "proj_diff_avg_norm", "d_w_avg_norm")
        max_keys = ("z_hat_max_norm", "proj_diff_max_norm", "d_w_max_norm")

        for key in avg_keys:
            vals = [m[key] for m in layer_metrics if key in m]
            if vals:
                out[f"ttt_norm/{key}"] = sum(vals) / len(vals)
        for key in max_keys:
            vals = [m[key] for m in layer_metrics if key in m]
            if vals:
                out[f"ttt_norm/{key}"] = max(vals)
        return out

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = (
            None
            if self.config.fuse_linear_cross_entropy
            else self.lm_head(hidden_states[:, -logits_to_keep:])
        )

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)),
                1,
            )
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["IPTTCDForCausalLM", "IPTTCDModel", "IPTTCDPreTrainedModel"]
