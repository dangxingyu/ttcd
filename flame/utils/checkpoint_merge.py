# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

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
    output_path: str,
    steps: list[int],
    weights: list[float],
    onlymlp: bool,
    config: str,
    tokenizer: str,
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving the config to {output_path}")
    config.save_pretrained(output_path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {output_path}")
    tokenizer.save_pretrained(output_path)
    merged_parameters = None

    logger.info(f"Merging checkpoints from {path} at\nsteps:\n{steps}\nweights:\n{weights}")

    def ajust_weights(name, step, weight):
        # return weight
        if not onlymlp:
            return weight
        if ".mlp." in name:
            return weight
        if step == max(steps):
            return 1
        else:
            return 0

    with tempfile.TemporaryDirectory(dir=output_path) as tmpdir:
        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        for step, weight in zip(steps, weights):
            checkpoint = os.path.join(path, f'checkpoint/step-{step}')
            checkpoint_path = os.path.join(tmpdir, f'checkpoint-{step}.pt')
            dcp_to_torch_save(checkpoint, checkpoint_path)
            # torch.load now with default weights_only=True will work
            model_parameter = torch.load(checkpoint_path, map_location='cpu')['model']
            if merged_parameters is None:
                merged_parameters = dict()
                for name, param in model_parameter.items():
                    merged_parameters[name] = param * ajust_weights(name, step, weight)
            else:
                for name, param in model_parameter.items():
                    merged_parameters[name] += param * ajust_weights(name, step, weight)
            logger.info(f"Checkpoint at step {step} has been merged with weight {weight}.")

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")
        merged_parameters, _ = strip_legacy_runtime_ttt_keys(model, merged_parameters)
        model.load_state_dict(merged_parameters)
        logger.info(f"Saving the model to {output_path}")
        model.save_pretrained(output_path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--merge_type", type=str, choices=["sma", "ema", "wma"], default="linear")
    parser.add_argument("--alpha", type=float, default=0.9) # min lr in wma
    parser.add_argument("--onlymlp", action='store_true')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    args.steps = sorted(args.steps)
    if args.merge_type == "sma":
        weights = [1 / len(args.steps)] * len(args.steps)
    elif args.merge_type == "ema":
        weights = [(args.alpha if i>0 else 1) * ((1 - args.alpha) ** (len(args.steps) - i - 1)) for i in range(len(args.steps))]
    elif args.merge_type == "wma":
        def get_lr_ratio(i: int) -> float:
            total_steps = max(args.steps)
            start_decay_step = min(args.steps)
            decay_steps = total_steps - start_decay_step + 1
            return 1.0 if i < start_decay_step else args.alpha if i >= total_steps else 1.0 + (args.alpha - 1.0) * (i - start_decay_step) / (decay_steps - 1)
        weights = []
        for i in range(1, len(args.steps)):
            if i == len(args.steps) - 1:
                c = get_lr_ratio(args.steps[i])
            else:
                c = get_lr_ratio(args.steps[i]) - get_lr_ratio(args.steps[i + 1])
            weights.append(c)
        weights = [1-(sum(weights))] + weights
    else:
        raise ValueError(f"Unknown merge type: {args.merge_type}")
    save_pretrained(args.path, args.output_path, args.steps, weights, args.onlymlp, args.config, args.tokenizer)
