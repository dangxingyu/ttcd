# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from custom_models.ipttt.configuration_ipttt import IPTTTConfig
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

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)

try:
    from opt_einsum import contract
except ImportError:
    contract = torch.einsum


class IPTTTMLP(nn.Module):

    def __init__(self, config: IPTTTConfig, layer_idx: int):
        super().__init__()
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
            self.ttt_lr = config.ttt_lr
            self.ttt_conv_causal = bool(getattr(config, "ttt_conv_causal", False))
            self.ttt_conv_kernel_size = int(getattr(config, "ttt_conv_kernel_size", 5))
            self.ttt_proj = (
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                if config.ttt_proj
                else None
            )
            if self.ttt_proj is not None:
                self.ttt_proj._is_ttt_proj = True
            self.ttt_conv = None
            self.ttt_activation_conv = None
            if bool(getattr(config, "ttt_conv", True)):
                conv_padding = 0 if self.ttt_conv_causal else self.ttt_conv_kernel_size // 2
                self.ttt_conv = nn.Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_size=self.ttt_conv_kernel_size,
                    padding=conv_padding,
                    groups=self.hidden_size,
                    bias=False,
                )
                self.ttt_conv._is_ttt_conv = True
            if bool(getattr(config, "ttt_activation_conv", False)):
                conv_padding = 0 if self.ttt_conv_causal else self.ttt_conv_kernel_size // 2
                self.ttt_activation_conv = nn.Conv1d(
                    self.intermediate_size,
                    self.intermediate_size,
                    kernel_size=self.ttt_conv_kernel_size,
                    padding=conv_padding,
                    groups=self.intermediate_size,
                    bias=False,
                )
                self.ttt_activation_conv._is_ttt_conv = True

    def _padding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_ttt_layer:
            return x
        if x.shape[1] % self.ttt_chunk != 0:
            pad_tokens = self.ttt_chunk - x.shape[1] % self.ttt_chunk
            padding = torch.zeros(
                (x.shape[0], pad_tokens, x.shape[2]),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def _apply_ttt_conv(self, x: torch.Tensor, conv: nn.Conv1d | None) -> torch.Tensor:
        if conv is None:
            return x
        batch_size, chunk_num, chunk_size, _ = x.shape
        x_flat = x.transpose(-1, -2).reshape(batch_size * chunk_num, -1, chunk_size)
        if self.ttt_conv_causal:
            x_flat = F.pad(x_flat, (conv.kernel_size[0] - 1, 0))
        return conv(x_flat).transpose(-1, -2).reshape(batch_size, chunk_num, chunk_size, -1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu and (t is None or not self.is_ttt_layer):
            return swiglu_linear(gate, up, self.down_proj.weight, self.down_proj.bias)

        h = swiglu(gate, up)
        if t is None or not self.is_ttt_layer:
            return self.down_proj(h)

        t = self._padding(t)
        h_padded = self._padding(h)
        batch_size = t.shape[0]
        t = self._apply_ttt_conv(t, self.ttt_conv)
        h_padded = self._apply_ttt_conv(h_padded, self.ttt_activation_conv)
        if self.ttt_proj is not None:
            d_down_proj = contract(
                "b t c h, b t c d, d e -> b t e h",
                h_padded[:, :-1],
                t[:, :-1],
                self.ttt_proj.weight,
            )
        else:
            d_down_proj = contract(
                "b t c h, b t c d -> b t d h",
                h_padded[:, :-1],
                t[:, :-1],
            )
        if self.training:
            self._last_delta_w_norm = d_down_proj.detach().norm().item()
        d_down_proj = torch.cat(
            [
                repeat(self.down_proj.weight, "d h -> b 1 d h", b=batch_size),
                d_down_proj * self.ttt_lr,
            ],
            dim=1,
        )
        d_down_proj_sum = d_down_proj.cumsum(dim=1)
        down_proj = contract("b t d h, b t c h -> b t c d", d_down_proj_sum, h_padded)
        return rearrange(down_proj, "b t c d -> b (t c) d")[:, : x.shape[1], :]


class IPTTTBlock(GradientCheckpointingLayer):

    def __init__(self, config: IPTTTConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.is_ttt_layer = bool(config.ttt_mode and layer_idx in set(config.ttt_layers))

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=getattr(config, "head_dim", None),
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
        self.mlp = IPTTTMLP(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        ttt_target_states: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        if self.is_ttt_layer:
            if getattr(self.config, "ttt_target", "hidden") == "embedding":
                target_states = ttt_target_states
            else:
                target_states = hidden_states
        else:
            target_states = None
        hidden_states = self.mlp(hidden_states, t=target_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


class IPTTTPreTrainedModel(TransformerPreTrainedModel):
    config_class = IPTTTConfig
    _no_split_modules = ["IPTTTBlock"]

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
            ttt_proj_init = getattr(self.config, "ttt_proj_init", "diagonal_gaussian")
            if ttt_proj_init in {"diagonal", "diagonal_gaussian"}:
                self._init_ttt_proj_as_diagonal(module.weight, self.config.initializer_range)
            elif ttt_proj_init == "zero":
                with torch.no_grad():
                    module.weight.zero_()
        if isinstance(module, nn.Conv1d) and getattr(module, "_is_ttt_conv", False):
            with torch.no_grad():
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _init_ttt_proj_as_diagonal(weight: torch.Tensor, scale: float = 1.0) -> None:
        with torch.no_grad():
            if weight.device.type == "meta":
                return
            diag_size = min(weight.shape[0], weight.shape[1])
            if DTensor is not None and distribute_tensor is not None and isinstance(weight, DTensor):
                local_weight = weight.to_local()
                diagonal_weight = torch.zeros(
                    weight.shape[0],
                    weight.shape[1],
                    device=local_weight.device,
                    dtype=local_weight.dtype,
                )
                generator = torch.Generator(device=local_weight.device)
                generator.manual_seed(42)
                diag_values = torch.randn(
                    diag_size,
                    generator=generator,
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


class IPTTTModel(IPTTTPreTrainedModel):

    def __init__(
        self,
        config: IPTTTConfig,
    ) -> IPTTTModel:
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
            [IPTTTBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
                "`IPTTTModel` does not support output attention weights now, so `output_attentions` is set to `False`.",
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
        shared_ttt_target_states = (
            inputs_embeds if getattr(self.config, "ttt_target", "hidden") == "embedding" else None
        )
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
                ttt_target_states=shared_ttt_target_states,
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


class IPTTTForCausalLM(IPTTTPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = IPTTTModel(config)
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


__all__ = ["IPTTTConfig", "IPTTTForCausalLM", "IPTTTModel", "IPTTTPreTrainedModel"]
