# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager
from torchtitan.config_manager import JobConfig

from torchtitan.components.optimizer import OptimizersContainer, OptimizersInBackwardContainer, FTOptimizersContainer

# from .muon_v1 import Muon
from .muon_v2 import Muon, get_default_muon_param_groups, default_is_muon

T = TypeVar("T", bound=Optimizer)

class MuonOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            # Check if router params should use AdamW instead of Muon
            # (useful for MoE: aux_loss optimized by AdamW, main model by Muon)
            router_use_adamw = getattr(
                getattr(model, 'config', None), 'router_use_adamw', False
            )
            if router_use_adamw:
                is_muon_func = lambda name, x: (
                    default_is_muon(name, x) and ".router." not in name
                )
            else:
                is_muon_func = default_is_muon

            param_groups = get_default_muon_param_groups(model, is_muon_func)
            self.optimizers.append(
                optimizer_cls(params=param_groups, **optimizer_kwargs)
            )
            for group in param_groups:
                all_params.extend(group["params"])
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

def build_optimizers(
    model_parts: list[nn.Module],
    job_config: JobConfig,
    ft_manager: FTManager,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    beta1 = job_config.optimizer.beta1
    beta2 = job_config.optimizer.beta2
    eps = job_config.optimizer.eps
    weight_decay = job_config.optimizer.weight_decay

    optim_implementation = job_config.optimizer.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    enable_muon = job_config.optimizer.enable_muon
    # TODO (alex): now it will ignore the optim_implementation
    if enable_muon:
        assert ft_manager.enabled == False and optim_in_bwd == False, "Current implementation of Muon is not compatible with FT or optim_in_bwd."
        use_distributed_muon = getattr(job_config.optimizer, 'use_distributed_muon', False)
        optimizer_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": job_config.optimizer.muon_momentum,
            "adamw_betas": (beta1, beta2),
            "adamw_eps": eps,
            "use_distributed_muon": use_distributed_muon,
        }
        return MuonOptimizersContainer(model_parts, Muon, optimizer_kwargs)

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
