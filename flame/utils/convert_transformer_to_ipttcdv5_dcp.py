# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from flame.utils.convert_transformer_to_ipttcdv3_dcp import (
    convert_transformer_to_ipttcdv3_dcp,
)
from torchtitan.tools.logging import init_logger


def convert_transformer_to_ipttcdv5_dcp(
    model: str,
    config: str,
    checkpoint: Path,
    tokenizer: str | None = None,
) -> None:
    # Reuse the generic key-matching converter implementation.
    # The actual target architecture is determined by the provided config (model_type=ipttcdv5).
    convert_transformer_to_ipttcdv3_dcp(
        model=model,
        config=config,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert a matching FLA transformer HF checkpoint to an IPTTCDv5 DCP checkpoint."
    )
    parser.add_argument("--model", type=str, required=True, help="HF source model path or repo id")
    parser.add_argument("--config", type=str, required=True, help="Target IPTTCDv5 config path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path for target vocab alignment (recommended).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output DCP checkpoint dir")
    args = parser.parse_args()

    convert_transformer_to_ipttcdv5_dcp(
        model=args.model,
        config=args.config,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
    )
