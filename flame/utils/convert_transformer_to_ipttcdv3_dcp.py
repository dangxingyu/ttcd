# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401
import custom_models.ipttcd  # noqa: F401 - register ipttcd models in AutoModel
from torchtitan.tools.logging import init_logger, logger


def _copy_with_vocab_resize(
    key: str,
    src_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
) -> bool:
    if key not in {"model.embeddings.weight", "lm_head.weight"}:
        return False
    if src_tensor.ndim != 2 or tgt_tensor.ndim != 2:
        return False
    if src_tensor.shape[1] != tgt_tensor.shape[1]:
        return False
    rows = min(src_tensor.shape[0], tgt_tensor.shape[0])
    tgt_tensor[:rows].copy_(src_tensor[:rows].to(dtype=tgt_tensor.dtype))
    return True


@torch.inference_mode()
def convert_transformer_to_ipttcdv3_dcp(
    model: str,
    config: str,
    checkpoint: Path,
    tokenizer: str | None = None,
) -> None:
    logger.info(f"Loading source transformer HF model from {model}")
    source = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
    )
    source_state = source.state_dict()

    logger.info(f"Loading target IPTTCDv3 config from {config}")
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
        if src_key not in target_state:
            skipped_unmapped.append(src_key)
            continue
        tgt_tensor = target_state[src_key]
        if tuple(src_tensor.shape) != tuple(tgt_tensor.shape):
            if _copy_with_vocab_resize(src_key, src_tensor, tgt_tensor):
                mapped += 1
                mapped_target_keys.add(src_key)
                partial_vocab_copied.append((src_key, tuple(src_tensor.shape), tuple(tgt_tensor.shape)))
                continue
            skipped_shape.append((src_key, tuple(src_tensor.shape), tuple(tgt_tensor.shape)))
            continue
        target_state[src_key].copy_(src_tensor.to(dtype=tgt_tensor.dtype))
        mapped += 1
        mapped_target_keys.add(src_key)

    not_mapped_target = sorted(set(target_state.keys()) - mapped_target_keys)

    if (
        "lm_head.weight" in target_state
        and "model.embeddings.weight" in target_state
        and target_state["lm_head.weight"].data_ptr() == target_state["model.embeddings.weight"].data_ptr()
    ):
        logger.info("Materializing tied lm_head.weight as a standalone tensor for DCP export")
        target_state["lm_head.weight"] = target_state["lm_head.weight"].clone()

    logger.info(f"Mapped tensors: {mapped}")
    logger.info(f"Unmapped source tensors: {len(skipped_unmapped)}")
    logger.info(f"Shape/key mismatched tensors: {len(skipped_shape)}")
    logger.info(f"Partially copied (vocab resize) tensors: {len(partial_vocab_copied)}")
    logger.info(f"Target tensors not mapped (kept init): {len(not_mapped_target)}")
    if skipped_unmapped:
        logger.info("Sample unmapped source keys:")
        for key in skipped_unmapped[:20]:
            logger.info(f"  {key}")
    if skipped_shape:
        logger.info("Sample mismatched entries:")
        for key, src_shape, tgt_shape in skipped_shape[:20]:
            logger.info(f"  {key}, src={src_shape}, tgt={tgt_shape}")
    if not_mapped_target:
        logger.info("Sample target keys kept as init:")
        for key in not_mapped_target[:20]:
            logger.info(f"  {key}")

    logger.info(f"Writing converted DCP checkpoint to {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save(target_state, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert a matching FLA transformer HF checkpoint to an IPTTCDv3 DCP checkpoint."
    )
    parser.add_argument("--model", type=str, required=True, help="HF source model path or repo id")
    parser.add_argument("--config", type=str, required=True, help="Target IPTTCDv3 config path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path for target vocab alignment (recommended).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output DCP checkpoint dir")
    args = parser.parse_args()

    convert_transformer_to_ipttcdv3_dcp(
        args.model,
        args.config,
        args.checkpoint,
        tokenizer=args.tokenizer,
    )
