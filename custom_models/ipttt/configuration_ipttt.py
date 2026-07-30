# -*- coding: utf-8 -*-

import warnings
from typing import Sequence

from transformers.configuration_utils import PretrainedConfig


class IPTTTConfig(PretrainedConfig):

    model_type = "ipttt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        use_output_gate: bool = False,
        gate_fn: str = "sigmoid",
        gate_logit_normalizer: int = 16,
        use_flash_attn_3: bool = False,
        use_l2_softmax: bool = False,
        ttt_mode: bool = False,
        ttt_layers: Sequence[int] | None = None,
        ttt_chunk: int = 8192,
        ttt_lr: float = 0.3,
        ttt_proj: bool = True,
        ttt_proj_init: str = "diagonal_gaussian",
        ttt_target: str = "hidden",
        ttt_conv: bool = True,
        ttt_conv_causal: bool = False,
        ttt_conv_kernel_size: int = 5,
        ttt_activation_conv: bool = False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        self.use_output_gate = use_output_gate
        self.gate_fn = gate_fn
        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_flash_attn_3 = use_flash_attn_3
        self.use_l2_softmax = use_l2_softmax

        self.ttt_mode = ttt_mode
        self.ttt_layers = list(ttt_layers) if ttt_layers is not None else []
        self.ttt_chunk = ttt_chunk
        self.ttt_lr = ttt_lr
        self.ttt_proj = ttt_proj
        self.ttt_proj_init = str(ttt_proj_init)
        ttt_target = str(ttt_target)
        ttt_target_aliases = {
            "input_embed": "embedding",
            "input_embeds": "embedding",
            "embed": "embedding",
            "hidden_states": "hidden",
            "hidden_state": "hidden",
        }
        self.ttt_target = ttt_target_aliases.get(ttt_target, ttt_target)
        self.ttt_conv = ttt_conv
        self.ttt_conv_causal = ttt_conv_causal
        self.ttt_conv_kernel_size = int(ttt_conv_kernel_size)
        self.ttt_activation_conv = bool(ttt_activation_conv)

        if any(layer < 0 or layer >= num_hidden_layers for layer in self.ttt_layers):
            raise ValueError(
                "`ttt_layers` must be within `[0, num_hidden_layers)`."
            )
        if self.ttt_chunk <= 0:
            raise ValueError("`ttt_chunk` must be > 0.")
        if self.ttt_lr < 0:
            raise ValueError("`ttt_lr` must be >= 0.")
        if self.ttt_proj_init not in {"diagonal", "diagonal_gaussian", "zero"}:
            raise ValueError("`ttt_proj_init` must be one of {'diagonal', 'diagonal_gaussian', 'zero'}.")
        if self.ttt_target not in {"hidden", "embedding"}:
            raise ValueError("`ttt_target` must be one of {'hidden', 'embedding'}.")
        if self.ttt_conv_kernel_size <= 0 or self.ttt_conv_kernel_size % 2 == 0:
            raise ValueError("`ttt_conv_kernel_size` must be a positive odd integer.")

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
