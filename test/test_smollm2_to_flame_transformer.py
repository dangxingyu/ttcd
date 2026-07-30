"""Convert SmolLM2-360M HF weights to a flame transformer DCP checkpoint and
verify logits agree with the upstream HF model on a real input.

Run:
    pytest test/test_smollm2_to_flame_transformer.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch.distributed.checkpoint as DCP  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import fla  # noqa: F401, E402

from flame.utils.convert_qwen_to_ipttcd_dcp import convert_qwen_to_ipttcd_dcp  # noqa: E402


HF_MODEL_DIR = "/scratch/gpfs/ARORA/xd7812/models/SmolLM2-360M"
TARGET_CONFIG = "configs/transformer/transformer_smollm2_360M.json"
TOKENIZER_DIR = "tokenizers/SmolLM2-360M"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for parity comparison")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def converted_state(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("smollm2_dcp") / "step-0"
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

    hf_top1 = hf_logits.argmax(-1)
    flame_top1 = flame_logits.argmax(-1)
    agreement = (hf_top1 == flame_top1).float().mean().item()
    assert agreement > 0.95, f"Top-1 agreement only {agreement:.3f}"

    diff = (hf_logits - flame_logits).abs()
    print(
        f"top1_agree={agreement:.4f} "
        f"abs_max={diff.max().item():.4f} "
        f"abs_mean={diff.mean().item():.4f}"
    )
    assert diff.mean().item() < 0.5
