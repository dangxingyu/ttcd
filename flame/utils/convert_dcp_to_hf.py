# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import sys
import tempfile
from datetime import timedelta

# Add flame_ut to sys.path to import fla
script_dir = os.path.dirname(os.path.abspath(__file__))
flame_ut_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
if flame_ut_dir not in sys.path:
    sys.path.insert(0, flame_ut_dir)

import fla  # noqa
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models
from flame.utils.legacy_state_dict import strip_legacy_runtime_ttt_keys


@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str,
    output_path: str = None,
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path if output_path is None else output_path}")
    config.save_pretrained(path if output_path is None else output_path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path if output_path is None else output_path}")
    tokenizer.save_pretrained(path if output_path is None else output_path)

    with tempfile.TemporaryDirectory(dir=None if output_path is None else output_path) as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
        state_dict, removed = strip_legacy_runtime_ttt_keys(model, state_dict)
        if removed:
            logger.info(f"Removed legacy runtime keys: {removed}")
        model.load_state_dict(state_dict)

        logger.info(f"Saving the model to {path if output_path is None else output_path}")
        model.save_pretrained(path if output_path is None else output_path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer, args.output_path)
