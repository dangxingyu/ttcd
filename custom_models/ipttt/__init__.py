# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from custom_models.ipttt.configuration_ipttt import IPTTTConfig
from custom_models.ipttt.modeling_ipttt import IPTTTForCausalLM, IPTTTModel

AutoConfig.register(IPTTTConfig.model_type, IPTTTConfig, exist_ok=True)
AutoModel.register(IPTTTConfig, IPTTTModel, exist_ok=True)
AutoModelForCausalLM.register(IPTTTConfig, IPTTTForCausalLM, exist_ok=True)

__all__ = ["IPTTTConfig", "IPTTTForCausalLM", "IPTTTModel"]
