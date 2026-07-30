# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from custom_models.ipttcd.configuration_deltanet_baseline import DeltaNetBaselineConfig
import fla.layers.attn as _fla_attn
from fla.layers.attn import Attention
from fla.models.transformer.modeling_transformer import TransformerPreTrainedModel
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.mlp import GatedMLP
from fla.modules.l2warp import l2_warp
from fla.ops.delta_rule import chunk_delta_rule

try:
    from flash_attn_interface import (
        flash_attn_func as _fa3_flash_attn_func,
        flash_attn_varlen_func as _fa3_flash_attn_varlen_func,
    )

    _FA3_AVAILABLE = True
except Exception:
    _FA3_AVAILABLE = False

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class DeltaNetBaselineBlock(GradientCheckpointingLayer):

    def __init__(self, config: DeltaNetBaselineConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.is_deltanet_layer = bool(config.ttt_mode and layer_idx in set(config.ttt_layers))

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
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
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        if self.is_deltanet_layer:
            self.dn_num_heads = config.num_heads
            self.dn_head_dim = config.hidden_size // config.num_heads
            self.dn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
                config.hidden_size, eps=config.norm_eps
            )
            self.dn_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.dn_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.dn_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.dn_o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.dn_b_proj = nn.Linear(config.hidden_size, config.num_heads, bias=False)

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

        attentions = None

        attn_output, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        if self.is_deltanet_layer and self.training:
            # dn_norm fuses attention residual, then DeltaNet on the normalized output
            if self.config.fuse_norm:
                dn_input, residual = self.dn_norm(attn_output, residual, True)
            else:
                residual = residual + attn_output
                dn_input = self.dn_norm(residual)

            d = self.dn_head_dim
            q = rearrange(self.dn_q_proj(dn_input), 'b t (h d) -> b t h d', d=d)
            k = rearrange(self.dn_k_proj(dn_input), 'b t (h d) -> b t h d', d=d)
            v = rearrange(self.dn_v_proj(dn_input), 'b t (h d) -> b t h d', d=d)
            beta = self.dn_b_proj(dn_input).sigmoid()

            o, _ = chunk_delta_rule(
                q=q, k=k, v=v, beta=beta,
                use_qk_l2norm_in_kernel=True,
                output_final_state=False,
                cu_seqlens=kwargs.get('cu_seqlens'),
            )
            dn_output = self.dn_o_proj(rearrange(o, 'b t h d -> b t (h d)'))

            # mlp_norm fuses DeltaNet residual
            if self.config.fuse_norm:
                hidden_states, residual = self.mlp_norm(dn_output, residual, True)
            else:
                residual = residual + dn_output
                hidden_states = self.mlp_norm(residual)
        else:
            # No DeltaNet: mlp_norm fuses attention residual directly
            if self.config.fuse_norm:
                hidden_states, residual = self.mlp_norm(attn_output, residual, True)
            else:
                residual = residual + attn_output
                hidden_states = self.mlp_norm(residual)

        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


class DeltaNetBaselinePreTrainedModel(TransformerPreTrainedModel):
    config_class = DeltaNetBaselineConfig
    _no_split_modules = ["DeltaNetBaselineBlock"]

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
        if isinstance(module, DeltaNetBaselineBlock) and module.is_deltanet_layer:
            with torch.no_grad():
                nn.init.zeros_(module.dn_o_proj.weight)


class DeltaNetBaselineModel(DeltaNetBaselinePreTrainedModel):

    def __init__(
        self,
        config: DeltaNetBaselineConfig,
    ) -> DeltaNetBaselineModel:
        super().__init__(config)
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

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeltaNetBaselineBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
                "`DeltaNetBaselineModel` does not support output attention weights now, "
                "so `output_attentions` is set to `False`.",
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


class DeltaNetBaselineForCausalLM(DeltaNetBaselinePreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeltaNetBaselineModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

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


__all__ = [
    "DeltaNetBaselineConfig",
    "DeltaNetBaselineForCausalLM",
    "DeltaNetBaselineModel",
    "DeltaNetBaselinePreTrainedModel",
]
