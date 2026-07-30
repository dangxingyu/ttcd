#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(path), trust_remote_code=True)


def main() -> None:
    import fla  # noqa: F401
    import custom_models  # noqa: F401

    ipttcd_mod = importlib.import_module("custom_models.ipttcd")
    importlib.import_module("custom_models.ipttt")
    importlib.import_module("custom_models.ipttt.inference")

    checks = {
        "ipttcdv9_340m": ROOT / "configs/ipttcd/ipttcdv9_340M_swa_teacher_student_causalconv.json",
        "ipttcdv9_760m": ROOT / "configs/ipttcd/ipttcdv9_760M_longcollections_train424.json",
        "ipttt": ROOT / "configs/ipttt/ipttt_340M_swa_causalconv.json",
        "deltanet_baseline": ROOT / "configs/ipttcd/deltanet_baseline_340M_swa.json",
        "gated_deltanet": ROOT / "configs/gated_deltanet/gated_deltanet_340M_swa_aligned_longcollections.json",
        "transformer": ROOT / "configs/transformer/transformer_340M_dpsk_swa.json",
    }

    loaded = {}
    for name, path in checks.items():
        cfg = load_config(path)
        loaded[name] = {
            "path": str(path),
            "model_type": getattr(cfg, "model_type", None),
            "ttt_force_grouped_scan": getattr(cfg, "ttt_force_grouped_scan", None),
        }

    print(json.dumps({
        "root": str(ROOT),
        "public_ipttcd_module": getattr(ipttcd_mod.IPTTCDv9ForCausalLM, "__module__", None),
        "loaded": loaded,
    }, indent=2))


if __name__ == "__main__":
    main()
