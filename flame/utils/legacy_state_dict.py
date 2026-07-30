from __future__ import annotations

from typing import Dict, Tuple

import torch


def strip_legacy_runtime_ttt_keys(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list[str]]:
    """Drop legacy runtime-only IPTTCD keys that should not be restored.

    Older IPTTCD checkpoints persisted `*.ttt_lr` as model buffers. These are
    runtime hyperparameters and should now come from config instead of checkpoint
    state. Remove them when the current model does not expect them.
    """

    model_keys = set(model.state_dict().keys())
    removed = []
    cleaned = {}
    for key, value in state_dict.items():
        if key not in model_keys and key.endswith(".ttt_lr"):
            removed.append(key)
            continue
        cleaned[key] = value
    return cleaned, removed
