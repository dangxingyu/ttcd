# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function

from custom_models.ipttcd.configuration_ipttcdv9_lowcopy import IPTTCDv9LowCopyConfig
from custom_models.ipttcd.modeling_ipttcdv9 import (
    _FA3_AVAILABLE,
    GradientCheckpointingLayer,
    IPTTCDBlock as _BaseIPTTCDBlock,
    IPTTCDForCausalLM as _BaseIPTTCDForCausalLM,
    IPTTCDMLP as _BaseIPTTCDMLP,
    IPTTCDModel as _BaseIPTTCDModel,
    IPTTCDPreTrainedModel as _BaseIPTTCDPreTrainedModel,
    RMSNorm,
    TransformerPreTrainedModel,
    logger,
    swiglu,
    swiglu_linear,
    zeropower_via_newtonschulz5,
)
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.deprecation import deprecate_kwarg

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from flash_attn_interface import (
        flash_attn_func as _fa3_flash_attn_func,
        flash_attn_varlen_func as _fa3_flash_attn_varlen_func,
    )
except Exception:
    _fa3_flash_attn_func = None
    _fa3_flash_attn_varlen_func = None


class IPTTCDMLP(_BaseIPTTCDMLP):

    def __init__(self, config: IPTTCDv9LowCopyConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)

    @staticmethod
    def _cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if tensor.dtype == dtype:
            return tensor
        return tensor.to(dtype=dtype)

    def _get_base_weight_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_weight = self._cast_if_needed(self.down_proj.weight, dtype)
        return base_weight.unsqueeze(0).expand(batch_size, -1, -1)

    def _get_base_weight_transposed(self, dtype: torch.dtype) -> torch.Tensor:
        base_weight_t = self.down_proj.weight.transpose(0, 1)
        base_weight_t = self._cast_if_needed(base_weight_t, dtype)
        return base_weight_t.unsqueeze(0).unsqueeze(0)

    def _get_ttt_proj_weight(self, dtype: torch.dtype) -> torch.Tensor | None:
        if self.ttt_proj is None:
            return None
        proj_weight = self._cast_if_needed(self.ttt_proj.weight, dtype)
        return proj_weight.unsqueeze(0).unsqueeze(0)

    def _prepare_output_weight(
        self,
        w_run: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._cast_if_needed(w_run, compute_dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        with record_function(f"ipttcdv9_lowcopy.mlp.teacher_gateup.layer{self.layer_idx}"):
            gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu and (t is None or not self.is_ttt_layer):
            return swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)

        with record_function(f"ipttcdv9_lowcopy.mlp.teacher_swiglu.layer{self.layer_idx}"):
            z = swiglu(gate, up)
        if t is None or not self.is_ttt_layer:
            return self.down_proj(z)

        seq_len = x.shape[1]

        with record_function(f"ipttcdv9_lowcopy.mlp.teacher_ttt_prepare.layer{self.layer_idx}"):
            z_padded = self._padding(z)
            b, T, C, _ = z_padded.shape
            compute_dtype = z_padded.dtype
            z_target_padded = self._apply_ttt_conv(z_padded, self.ttt_teacher_conv)
            teacher_output_padded = (
                z_target_padded if self.ttt_output_use_teacher_conv else z_padded
            )

        with record_function(f"ipttcdv9_lowcopy.mlp.student_gateup.layer{self.layer_idx}"):
            t_padded = self._padding(t)
            t_flat = t_padded.reshape(b, T * C, -1)
            z_hat = swiglu(self.gate_proj(t_flat), self.up_proj(t_flat))
            z_hat_padded = z_hat.reshape(b, T, C, -1)
            z_hat_padded = self._apply_ttt_conv(z_hat_padded, self.ttt_student_conv)

        z_output_padded = self._select_ttt_output_padded(
            teacher_output_padded=teacher_output_padded,
            student_output_padded=z_hat_padded,
        )

        beta_padded = None
        if self.dynamic_lr:
            with record_function(f"ipttcdv9_lowcopy.mlp.student_beta.layer{self.layer_idx}"):
                beta_padded = (t_padded @ self.b_proj).unsqueeze(-1).sigmoid()

        scan_state_dtype = self._get_ttt_scan_state_dtype(compute_dtype)
        W_base_state = self._get_base_weight_state(b, scan_state_dtype)
        W_base_t = self._get_base_weight_transposed(compute_dtype)

        W_base_frob_norm = None
        if self.ttt_weight_renorm:
            W_base_frob_norm = (
                self.down_proj.weight.float()
                .unsqueeze(0)
                .expand(b, -1, -1)
                .flatten(1)
                .norm(dim=-1, keepdim=True)
                .view(b, 1, 1)
            )

        ttt_proj_weight_unsq = self._get_ttt_proj_weight(compute_dtype)
        ttt_lr_state = W_base_state.new_tensor(self.ttt_lr)
        update_every = self._get_ttt_scan_group_size(T)

        # NOTE: compute_norm_metrics must be False when using activation
        # checkpointing, because the forward is re-executed during backward
        # and the counter-based check produces a different value on the second
        # pass, causing tensor shape mismatches in the autograd graph.
        compute_norm_metrics = False

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
            and ttt_proj_weight_unsq is not None
            and not self.ttt_norm_proj_diff
            and not compute_norm_metrics
        )
        if fuse_ttt_proj_into_update:
            with record_function(f"ipttcdv9_lowcopy.mlp.fuse_base_proj.layer{self.layer_idx}"):
                W_update_t = torch.matmul(W_base_t, ttt_proj_weight_unsq)
        else:
            W_update_t = W_base_t

        if not self.ttt_weight_renorm:
            with record_function(f"ipttcdv9_lowcopy.mlp.update_scan.layer{self.layer_idx}"):
                if T > 1:
                    z_target_upd = z_target_padded[:, :-1]
                    z_hat_upd = z_hat_padded[:, :-1]
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
                        proj_diff.transpose(-1, -2),
                        z_hat_norm,
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

                    d_w_states = self._cast_if_needed(d_w_chunks, scan_state_dtype)
                    w_states = torch.cat(
                        [W_base_state.unsqueeze(1), ttt_lr_state * d_w_states],
                        dim=1,
                    )
                    w_run = torch.cumsum(w_states, dim=1)
                else:
                    w_run = W_base_state.unsqueeze(1)

            with record_function(f"ipttcdv9_lowcopy.mlp.output_proj.layer{self.layer_idx}"):
                output = torch.matmul(
                    z_output_padded,
                    self._prepare_output_weight(w_run, compute_dtype).transpose(-1, -2),
                )
        else:
            with record_function(f"ipttcdv9_lowcopy.mlp.grouped_scan.layer{self.layer_idx}"):
                W_cur = W_base_state
                output = z_output_padded.new_empty(b, T, C, self.hidden_size)
                for gs in range(0, T, update_every):
                    ge = min(gs + update_every, T)
                    Kg = ge - gs
                    upd_end = min(ge, T - 1)
                    K_upd = upd_end - gs

                    if K_upd > 0:
                        z_target_g = z_target_padded[:, gs:upd_end]
                        z_hat_g = z_hat_padded[:, gs:upd_end]
                        diff_g = z_target_g - z_hat_g
                        proj_diff_g = torch.matmul(diff_g, W_update_t)
                        if self.ttt_norm_proj_diff:
                            proj_diff_g = F.normalize(proj_diff_g, p=2, dim=-1, eps=1e-6)

                        if compute_norm_metrics:
                            with torch.no_grad():
                                z_hat_token_norm = z_hat_g.detach().float().norm(dim=-1)
                                z_hat_avg_sum += float(z_hat_token_norm.mean().item()) * K_upd
                                z_hat_max_norm = max(
                                    z_hat_max_norm,
                                    float(z_hat_token_norm.max().item()),
                                )

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
                            proj_diff_g.transpose(-1, -2),
                            z_hat_g_norm,
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

                        d_w_states = self._cast_if_needed(d_w_chunks, scan_state_dtype)
                        cs = ttt_lr_state * torch.cumsum(d_w_states, dim=1)
                    else:
                        cs = None

                    w_run = W_cur.unsqueeze(1).expand(-1, Kg, -1, -1).clone()
                    if K_upd > 0:
                        fill = min(Kg, K_upd + 1)
                        if fill > 1:
                            w_run[:, 1:fill] = w_run[:, 1:fill] + cs[:, :fill - 1]
                        if fill < Kg:
                            w_run[:, fill:] = w_run[:, fill:] + cs[:, -1:].expand(-1, Kg - fill, -1, -1)
                        w_run = self._renorm_weight_states(
                            w_run.float(),
                            W_base_frob_norm.unsqueeze(1),
                        ).to(dtype=scan_state_dtype)
                        W_cur = self._renorm_weight_states(
                            W_cur.float() + cs[:, -1].float(),
                            W_base_frob_norm,
                        ).to(dtype=scan_state_dtype)

                    output[:, gs:ge] = torch.matmul(
                        z_output_padded[:, gs:ge],
                        self._prepare_output_weight(w_run, compute_dtype).transpose(-1, -2),
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


class IPTTCDBlock(_BaseIPTTCDBlock):

    def __init__(self, config: IPTTCDv9LowCopyConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.mlp = IPTTCDMLP(config=config, layer_idx=layer_idx)


class IPTTCDPreTrainedModel(_BaseIPTTCDPreTrainedModel):
    config_class = IPTTCDv9LowCopyConfig
    _no_split_modules = ["IPTTCDBlock"]


class IPTTCDModel(_BaseIPTTCDModel):

    config_class = IPTTCDv9LowCopyConfig
    _no_split_modules = ["IPTTCDBlock"]

    def __init__(
        self,
        config: IPTTCDv9LowCopyConfig,
    ) -> IPTTCDModel:
        TransformerPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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

        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [IPTTCDBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = (RMSNorm if config.fuse_norm else torch.nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.gradient_checkpointing = False

        self.post_init()


class IPTTCDForCausalLM(_BaseIPTTCDForCausalLM):

    _tied_weights_keys = ["lm_head.weight"]
    config_class = IPTTCDv9LowCopyConfig

    def __init__(self, config):
        TransformerPreTrainedModel.__init__(self, config)
        self.model = IPTTCDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()


__all__ = ["IPTTCDForCausalLM", "IPTTCDModel", "IPTTCDPreTrainedModel"]
