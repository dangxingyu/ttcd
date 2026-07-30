# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa: F401 - register ipttcd in AutoModel

try:
    from torchtitan.tools.logging import init_logger, logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    init_logger = lambda: None  # noqa: E731


_LAYER_PATTERNS = [
    (
        re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"),
        lambda m: f"model.layers.{m.group(1)}.attn.{m.group(2)}.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.bias$"),
        lambda m: f"model.layers.{m.group(1)}.attn.{m.group(2)}.bias",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$"),
        lambda m: f"model.layers.{m.group(1)}.attn.{m.group(2)}.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),
        lambda m: f"model.layers.{m.group(1)}.attn_norm.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"),
        lambda m: f"model.layers.{m.group(1)}.mlp_norm.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"),
        lambda m: f"model.layers.{m.group(1)}.mlp.{m.group(2)}.weight",
    ),
]


def map_qwen_key_to_ipttcd(src_key: str) -> str | None:
    if src_key == "model.embed_tokens.weight":
        return "model.embeddings.weight"
    if src_key == "model.norm.weight":
        return "model.norm.weight"
    if src_key == "lm_head.weight":
        return "lm_head.weight"
    for pattern, mapper in _LAYER_PATTERNS:
        match = pattern.match(src_key)
        if match is not None:
            return mapper(match)
    return None


def _copy_with_vocab_resize(
    src_key: str,
    src_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
) -> bool:
    if src_key not in {"model.embed_tokens.weight", "lm_head.weight"}:
        return False
    if src_tensor.ndim != 2 or tgt_tensor.ndim != 2:
        return False
    if src_tensor.shape[1] != tgt_tensor.shape[1]:
        return False
    rows = min(src_tensor.shape[0], tgt_tensor.shape[0])
    tgt_tensor[:rows].copy_(src_tensor[:rows].to(dtype=tgt_tensor.dtype))
    return True


@torch.inference_mode()
def convert_qwen_to_ipttcd_dcp(
    model: str,
    config: str,
    checkpoint: Path,
    tokenizer: str | None = None,
) -> None:
    logger.info(f"Loading source HF model from {model}")
    source = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
    )
    source_state = source.state_dict()

    logger.info(f"Loading target config from {config}")
    target_config = AutoConfig.from_pretrained(config, trust_remote_code=True)
    if tokenizer:
        logger.info(f"Loading tokenizer from {tokenizer} for vocab alignment")
        tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        target_config.vocab_size = max(target_config.vocab_size, tokenizer_obj.vocab_size)
        logger.info(f"Aligned target vocab_size to {target_config.vocab_size}")
    target = AutoModelForCausalLM.from_config(target_config, trust_remote_code=True)
    target_state = target.state_dict()

    src_layers = getattr(source.config, "num_hidden_layers", None)
    tgt_layers = getattr(target.config, "num_hidden_layers", None)
    src_hidden = getattr(source.config, "hidden_size", None)
    tgt_hidden = getattr(target.config, "hidden_size", None)
    logger.info(
        f"Source layers/hidden={src_layers}/{src_hidden}, "
        f"Target layers/hidden={tgt_layers}/{tgt_hidden}"
    )

    mapped = 0
    mapped_target_keys = set()
    skipped_unmapped = []
    skipped_shape = []
    partial_vocab_copied = []

    for src_key, src_tensor in source_state.items():
        tgt_key = map_qwen_key_to_ipttcd(src_key)
        if tgt_key is None:
            skipped_unmapped.append(src_key)
            continue
        if tgt_key not in target_state:
            skipped_shape.append((src_key, tgt_key, tuple(src_tensor.shape), None))
            continue
        tgt_tensor = target_state[tgt_key]
        if tuple(src_tensor.shape) != tuple(tgt_tensor.shape):
            if _copy_with_vocab_resize(src_key, src_tensor, tgt_tensor):
                mapped += 1
                mapped_target_keys.add(tgt_key)
                partial_vocab_copied.append((src_key, tgt_key, tuple(src_tensor.shape), tuple(tgt_tensor.shape)))
                continue
            skipped_shape.append(
                (src_key, tgt_key, tuple(src_tensor.shape), tuple(tgt_tensor.shape))
            )
            continue
        target_state[tgt_key].copy_(src_tensor.to(dtype=tgt_tensor.dtype))
        mapped += 1
        mapped_target_keys.add(tgt_key)
    not_mapped_target = sorted(set(target_state.keys()) - mapped_target_keys)

    logger.info(f"Mapped tensors: {mapped}")
    logger.info(f"Unmapped source tensors: {len(skipped_unmapped)}")
    logger.info(f"Shape/key mismatched tensors: {len(skipped_shape)}")
    logger.info(f"Partially copied (vocab resize) tensors: {len(partial_vocab_copied)}")
    if skipped_unmapped:
        logger.info("Sample unmapped source keys:")
        for k in skipped_unmapped[:20]:
            logger.info(f"  {k}")
    if partial_vocab_copied:
        logger.info("Vocab-resized copy entries:")
        for src_key, tgt_key, src_shape, tgt_shape in partial_vocab_copied:
            logger.info(f"  {src_key} -> {tgt_key}, src={src_shape}, tgt={tgt_shape}")
    if skipped_shape:
        logger.info("Sample mismatched entries:")
        for src_key, tgt_key, src_shape, tgt_shape in skipped_shape[:20]:
            logger.info(f"  {src_key} -> {tgt_key}, src={src_shape}, tgt={tgt_shape}")
    logger.info(f"Target tensors not mapped (kept init): {len(not_mapped_target)}")
    if not_mapped_target:
        logger.info("Sample target keys kept as init:")
        for k in not_mapped_target[:20]:
            logger.info(f"  {k}")

    # CT-friendly initialization: set teacher conv on output path to identity
    # (delta kernel) so the model matches the original at step 0.
    ttt_output_use_teacher_conv = bool(
        getattr(target_config, "ttt_output_use_teacher_conv", True)
    )
    ct_conv_count = 0
    if ttt_output_use_teacher_conv:
        for k in list(target_state.keys()):
            if ".mlp.ttt_teacher_conv.weight" in k:
                w = target_state[k]
                with torch.no_grad():
                    w.zero_()
                    w[:, 0, -1] = 1.0  # last tap = current position for causal conv
                ct_conv_count += 1
        if ct_conv_count > 0:
            logger.info(
                f"CT init: set {ct_conv_count} teacher conv(s) to delta kernel "
                f"(identity at current position) for continual training."
            )

    # Materialize tied weights as independent tensors for DCP export.
    if (
        "lm_head.weight" in target_state
        and "model.embeddings.weight" in target_state
        and target_state["lm_head.weight"].data_ptr() == target_state["model.embeddings.weight"].data_ptr()
    ):
        logger.info("Materializing tied lm_head.weight as standalone tensor for DCP export")
        target_state["lm_head.weight"] = target_state["lm_head.weight"].clone()

    logger.info(f"Writing converted DCP checkpoint to {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    # The continual-pretrain init path loads model weights directly from the
    # checkpoint root, so persist a raw model state_dict here as well.
    DCP.save(target_state, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert Qwen-style HF weights to IPTTCD-style DCP checkpoint."
    )
    parser.add_argument("--model", type=str, required=True, help="HF source model path or repo id")
    parser.add_argument("--config", type=str, required=True, help="Target IPTTCD config path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path for target vocab alignment (recommended).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output DCP checkpoint dir")
    args = parser.parse_args()

    convert_qwen_to_ipttcd_dcp(args.model, args.config, args.checkpoint, tokenizer=args.tokenizer)
