"""End-to-end test: convert Qwen3-0.6B HF weights into a flame transformer
DCP checkpoint and verify the resulting flame model produces logits that
agree with the original HF Qwen3 on a real input.

Run:
    pytest test/test_qwen3_to_flame_transformer.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch.distributed.checkpoint as DCP  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import fla  # noqa: F401, E402 - registers the transformer model_type

import custom_models  # noqa: F401, E402

from flame.utils.convert_qwen_to_ipttcd_dcp import convert_qwen_to_ipttcd_dcp  # noqa: E402


HF_MODEL_DIR = "/scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B"
TARGET_CONFIG = "configs/transformer/transformer_qwen3_0.6B.json"
TOKENIZER_DIR = "tokenizers/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for parity comparison")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def converted_state(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("qwen3_dcp") / "step-0"
    convert_qwen_to_ipttcd_dcp(
        model=HF_MODEL_DIR,
        config=TARGET_CONFIG,
        checkpoint=Path(out_dir),
        tokenizer=TOKENIZER_DIR,
    )
    return out_dir


def _load_flame_from_dcp(dcp_dir, device):
    config = AutoConfig.from_pretrained(TARGET_CONFIG, trust_remote_code=True)
    config.fuse_cross_entropy = False
    config.fuse_linear_cross_entropy = False
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    state_dict = model.state_dict()
    storage_reader = DCP.filesystem.FileSystemReader(str(dcp_dir))
    DCP.load(state_dict, storage_reader=storage_reader)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _load_hf(device):
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model = model.to(device=device)
    model.eval()
    return model


@torch.no_grad()
def test_logits_parity(converted_state, device):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Continual pretraining lets a base language model adapt to longer "
        "documents while keeping its existing knowledge intact."
    )
    ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=device)

    hf = _load_hf(device)
    flame_model = _load_flame_from_dcp(converted_state, device)

    hf_logits = hf(ids).logits.float()
    flame_logits = flame_model(ids).logits.float()

    assert hf_logits.shape == flame_logits.shape

    # Compare top-1 token predictions across the sequence — this is the
    # invariant that matters for downstream eval, and is robust to small
    # numerical differences between SDPA and flash-attn-2 backends.
    hf_top1 = hf_logits.argmax(-1)
    flame_top1 = flame_logits.argmax(-1)
    agreement = (hf_top1 == flame_top1).float().mean().item()
    assert agreement > 0.95, f"Top-1 agreement only {agreement:.3f}"

    # Compare logit values themselves: should be close in bf16 (~3% relative).
    diff = (hf_logits - flame_logits).abs()
    rel_diff = diff / (hf_logits.abs() + 1e-3)
    print(
        f"top1_agree={agreement:.4f} "
        f"abs_max={diff.max().item():.4f} "
        f"abs_mean={diff.mean().item():.4f} "
        f"rel_max={rel_diff.max().item():.4f} "
        f"rel_mean={rel_diff.mean().item():.4f}"
    )
    assert diff.mean().item() < 0.5
