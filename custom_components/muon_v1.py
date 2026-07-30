import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

import math

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_weight_decay: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        adamw_params=None,
        muon_params=None,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # TODO (alex): try p2p communication?

            # generate weight updates
            for base in range(0, len(params), world_size):
                current_block = params[base : base + world_size]
                full_grads = [None] * len(current_block)

                for local_idx, p in enumerate(current_block):
                    # sanity check
                    g = p.grad
                    if g is None:
                        continue
                    if isinstance(g, DTensor):
                        g_local = g._local_tensor
                    else:
                        g_local = g
                    owner = (base + local_idx) % world_size
                    if world_size == 1:
                        full_grads[local_idx] = g_local
                        continue
                    # if rank == owner:
                    #     shard_list = [
                    #         torch.empty_like(g_local) for _ in range(world_size)
                    #     ]
                    #     dist.gather(g_local, gather_list=shard_list, dst=owner)
                    #     # TODO (alex): now if the world size not divided by dim, there will be some padding issues!!!! check https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/tensor/placement_types.py#L50 
                    #     full_grads[local_idx] = torch.cat(shard_list, dits[0].dim) # TODO (alex): check dim, shape and local rank in device mesh
                    # else:
                    #     dist.gather(g_local, gather_list=None, dst=owner)
                    g_replicate = g.redistribute(placements=[Replicate()])
                    if rank == owner:
                        full_grads[local_idx] = g_replicate._local_tensor
                
                full_updates = [None] * len(current_block)
                # SMPD-style parallel
                for local_idx, p in enumerate(current_block):
                    owner = (base + local_idx) % world_size
                    if world_size > 1 and rank != owner:
                        continue
                    g_full = full_grads[local_idx]
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_full)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g_full)
                    if nesterov:
                        g_full = g_full.add(buf, alpha=momentum)
                    else:
                        g_full = buf
                    u_full = zeropower_via_newtonschulz5(g_full, steps=ns_steps).to(g_full.dtype)
                    full_updates[local_idx] = u_full

                for local_idx, p in enumerate(current_block):
                    g = p.grad
                    if g is None:
                        continue
                    if isinstance(g, DTensor):
                        g_local = g._local_tensor
                    else:
                        g_local = g
                    owner = (base + local_idx) % world_size
                    if world_size == 1:
                        u_local = full_updates[local_idx]
                    else:
                        # if rank == owner:
                        #     # Scatter full update to all workers
                        #     u_full = full_updates[local_idx]
                        #     if u_full is None:
                        #         shard_list = [torch.zeros_like(g_local) for _ in range(world_size)]
                        #     else:
                        #         if isinstance(g, u_full):
                        #             shard_list = list(u_full.chunk(world_size, dim=g.placements[0].dim))
                        #         else:
                        #             shard_list = [u_full] * world_size
                        # else:
                        #     shard_list = None
                        # u_local = torch.empty_like(g_local)
                        # dist.scatter(
                        #     u_local,
                        #     scatter_list=shard_list,
                        #     src=owner,
                        # )
                        if rank == owner:
                            current_ufull = full_updates[local_idx]
                        else:
                            current_ufull = torch.empty(g.shape, dtype=g.dtype, device=g.device)

                        dist.broadcast(current_ufull, src=owner)

                        if isinstance(p, DTensor):
                            mesh = p.device_mesh
                            u_rep = DTensor.from_local(current_ufull, mesh, placements=[Replicate()])
                            u_shard = u_rep.redistribute(placements=p.placements)
                            u_local = u_shard._local_tensor
                        else:
                            u_local = current_ufull
                    
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    if isinstance(p, DTensor):
                        p_local = p._local_tensor # inplace update _local_tensor
                        p_local.mul_(1 - lr * weight_decay)
                        p_local.add_(u_local, alpha=-adjusted_lr)
                    else:
                        p.data.mul_(1 - lr * weight_decay)
                        p.data.add_(u_local, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss