# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = model.state_dict()

    # DCP load does not preserve tied aliases reliably for our init path, so
    # materialize lm_head as an independent tensor when it shares storage with
    # the input embeddings.
    if (
        "lm_head.weight" in state_dict
        and "model.embeddings.weight" in state_dict
        and state_dict["lm_head.weight"].data_ptr() == state_dict["model.embeddings.weight"].data_ptr()
    ):
        logger.info("Materializing tied lm_head.weight as a standalone tensor for DCP export")
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    # `initial_load_model_weights_only` expects a raw model state_dict rather than
    # a nested {"model": ...} payload.
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
