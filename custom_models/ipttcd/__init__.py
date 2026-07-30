# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from custom_models.ipttcd.configuration_deltanet_baseline import DeltaNetBaselineConfig
from custom_models.ipttcd.configuration_ipttcdv9 import IPTTCDv9Config
from custom_models.ipttcd.modeling_deltanet_baseline import (
    DeltaNetBaselineForCausalLM,
    DeltaNetBaselineModel,
)
from custom_models.ipttcd.modeling_ipttcdv9_lowcopy_scanfuse import (
    IPTTCDForCausalLM as IPTTCDv9ForCausalLM,
    IPTTCDModel as IPTTCDv9Model,
)


def _register(config_cls, model_cls, causal_cls):
    AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)
    AutoModel.register(config_cls, model_cls, exist_ok=True)
    AutoModelForCausalLM.register(config_cls, causal_cls, exist_ok=True)


_register(IPTTCDv9Config, IPTTCDv9Model, IPTTCDv9ForCausalLM)
_register(DeltaNetBaselineConfig, DeltaNetBaselineModel, DeltaNetBaselineForCausalLM)


__all__ = [
    "IPTTCDv9Config",
    "IPTTCDv9Model",
    "IPTTCDv9ForCausalLM",
    "DeltaNetBaselineConfig",
    "DeltaNetBaselineModel",
    "DeltaNetBaselineForCausalLM",
]
