# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import record_function

from custom_models.ipttcd.configuration_ipttcdv9 import IPTTCDv9Config
from custom_models.ipttcd.modeling_ipttcdv9_lowcopy import (
    _FA3_AVAILABLE,
    _fa3_flash_attn_func,
    _fa3_flash_attn_varlen_func,
    IPTTCDBlock as _LowCopyBlock,
    IPTTCDForCausalLM as _LowCopyForCausalLM,
    IPTTCDMLP as _LowCopyMLP,
    IPTTCDModel as _LowCopyModel,
    IPTTCDPreTrainedModel as _LowCopyPreTrainedModel,
    RMSNorm,
    TransformerPreTrainedModel,
    logger,
    swiglu,
    swiglu_linear,
    zeropower_via_newtonschulz5,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class IPTTCDMLP(_LowCopyMLP):

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        use_scanfuse = t is not None and self.is_ttt_layer and not self.ttt_weight_renorm
        if not use_scanfuse:
            return super().forward(x, t, **kwargs)

        with record_function(f"ipttcdv9_scanfuse.mlp.teacher_gateup.layer{self.layer_idx}"):
            gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu and (t is None or not self.is_ttt_layer):
            return swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)

        with record_function(f"ipttcdv9_scanfuse.mlp.teacher_swiglu.layer{self.layer_idx}"):
            z = swiglu(gate, up)

        seq_len = x.shape[1]
        with record_function(f"ipttcdv9_scanfuse.mlp.teacher_ttt_prepare.layer{self.layer_idx}"):
            z_padded = self._padding(z)
            b, T, C, _ = z_padded.shape
            compute_dtype = z_padded.dtype
            z_target_padded = self._apply_ttt_conv(z_padded, self.ttt_teacher_conv)
            teacher_output_padded = (
                z_target_padded if self.ttt_output_use_teacher_conv else z_padded
            )

        with record_function(f"ipttcdv9_scanfuse.mlp.student_gateup.layer{self.layer_idx}"):
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
            with record_function(f"ipttcdv9_scanfuse.mlp.student_beta.layer{self.layer_idx}"):
                beta_padded = (t_padded @ self.b_proj).unsqueeze(-1).sigmoid()

        scan_state_dtype = self._get_ttt_scan_state_dtype(compute_dtype)
        W_base_state = self._get_base_weight_state(b, scan_state_dtype)
        W_base_t = self._get_base_weight_transposed(compute_dtype)
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
            with record_function(f"ipttcdv9_scanfuse.mlp.fuse_base_proj.layer{self.layer_idx}"):
                W_update_t = torch.matmul(W_base_t, ttt_proj_weight_unsq)
        else:
            W_update_t = W_base_t

        with record_function(f"ipttcdv9_scanfuse.mlp.scanfused_output.layer{self.layer_idx}"):
            W_cur = W_base_state
            output_groups = []
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

                group_outputs = []
                for local_idx in range(Kg):
                    if local_idx == 0 or K_upd == 0:
                        W_step = W_cur
                    else:
                        prefix_idx = min(local_idx - 1, K_upd - 1)
                        W_step = W_cur + cs[:, prefix_idx]
                    group_outputs.append(torch.matmul(
                        z_output_padded[:, gs + local_idx],
                        self._prepare_output_weight(W_step, compute_dtype).transpose(-1, -2),
                    ))

                if K_upd > 0:
                    W_cur = W_cur + cs[:, -1]
                output_groups.append(torch.stack(group_outputs, dim=1))

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

        output = torch.cat(output_groups, dim=1)
        return rearrange(output, "b t c d -> b (t c) d")[:, :seq_len, :]


class IPTTCDBlock(_LowCopyBlock):

    def __init__(self, config: IPTTCDv9Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.mlp = IPTTCDMLP(config=config, layer_idx=layer_idx)


class IPTTCDPreTrainedModel(_LowCopyPreTrainedModel):
    config_class = IPTTCDv9Config
    _no_split_modules = ["IPTTCDBlock"]


class IPTTCDModel(_LowCopyModel):

    config_class = IPTTCDv9Config
    _no_split_modules = ["IPTTCDBlock"]

    def __init__(self, config: IPTTCDv9Config) -> IPTTCDModel:
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
            logger.warning_once("Flash Attention 3 (Hopper) enabled — patched fla.layers.attn")

        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [IPTTCDBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = (RMSNorm if config.fuse_norm else torch.nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.gradient_checkpointing = False

        self.post_init()


class IPTTCDForCausalLM(_LowCopyForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = IPTTCDv9Config

    def __init__(self, config):
        TransformerPreTrainedModel.__init__(self, config)
        self.model = IPTTCDModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()


__all__ = ["IPTTCDForCausalLM", "IPTTCDModel", "IPTTCDPreTrainedModel"]
