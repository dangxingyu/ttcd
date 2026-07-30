import logging
import math
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.placement_types import Placement

from .distributed.utils import construct_shard_mesh, get_slices_of_dtensor
from .matmul_transpose_triton import matmul_transpose_assign

logger = logging.getLogger(__name__)

COMM_DTYPE = torch.bfloat16
DEFAULT_CHUNK_SIZE_RATIO = 4


# This code snippet is a modified version adapted from the following GitHub repositories:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# Muon's Newton–Schulz iteration causes high variance in singular values
# Idea: give each iteration its own 3 coefficients and optimize them via gradient descent.
@torch.no_grad()
# matmul_transpose_assign from : https://github.com/nil0x9/flash-muon
def _zeropower_via_newtonschulz5(G, steps):
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
    assert G.dtype == COMM_DTYPE
    X = G  # no manual typecast

    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        buf1.mul_(b).add_(buf2, alpha=c)
        X = torch.addmm(X, buf1, X, alpha=1.0, beta=a)

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.no_grad()
def _zeropower_via_newtonschulz5_batched(G, steps):
    """
    Batched Newton-Schulz iteration for 3D tensors [E, M, N].
    Each [M, N] slice (e.g. one MoE expert) is orthogonalized independently
    in parallel using torch.bmm, avoiding a sequential for-loop over experts.
    """
    assert len(G.shape) == 3
    assert G.dtype == COMM_DTYPE
    E, M, N = G.shape
    X = G

    transposed = M > N
    if transposed:
        X = X.transpose(-1, -2)  # [E, N, M]

    # Per-expert spectral norm normalization
    norms = X.flatten(1).norm(dim=1).unsqueeze(-1).unsqueeze(-1)  # [E, 1, 1]
    X = X / (norms + 1e-7)

    K = X.size(1)  # min(M, N)
    buf1 = torch.empty(E, K, K, dtype=X.dtype, device=X.device)
    buf2 = torch.empty(E, K, K, dtype=X.dtype, device=X.device)

    # Perform the NS iterations (same coefficients as 2D version)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        # buf1 = X @ X^T  (per-expert, batched)
        torch.bmm(X, X.transpose(-1, -2), out=buf1)
        # buf2 = buf1 @ buf1^T  (per-expert, batched)
        torch.bmm(buf1, buf1.transpose(-1, -2), out=buf2)
        buf1.mul_(b).add_(buf2, alpha=c)
        # X = a * X + buf1 @ X
        X = torch.baddbmm(X, buf1, X, alpha=1.0, beta=a)

    if transposed:
        X = X.transpose(-1, -2)
    return X


@dataclass
class _muon_state:
    # TODO: use Optional
    worker_rank: int
    process_group: ProcessGroup
    shard_mesh: DeviceMesh
    shard_placements: tuple[Placement, ...]
    name: str
    qk_clip_state: torch.Tensor | None = None
    gathered_grad: torch.Tensor | None = None
    scattered_u: DTensor | None = None
    computed_u: torch.Tensor | None = None
    gather_event: torch.cuda.Event | None = None
    compute_event: torch.cuda.Event | None = None
    scatter_event: torch.cuda.Event | None = None


def numel_for_rank(
    param: DTensor,
    local_rank: int,
    state: _muon_state,
) -> int:
    slices = get_slices_of_dtensor(
        param,
        local_rank,
        state.shard_mesh,
        state.shard_placements,
    )

    numel = 1
    for s, dim in zip(slices, param.shape):
        start, stop, step = s.indices(dim)
        length = max(0, (stop - start + (step - 1)) // step)
        numel *= length

    return numel


@torch.no_grad()
def _alloc_gathered_grad(params, param_to_state, rank, compute_stream):
    """
    Pre-allocate gathered_grad buffer on compute_stream
    before launching all2all gather
    """
    with torch.cuda.stream(compute_stream):
        for p in params:
            state = param_to_state[id(p)]
            if rank == state.worker_rank:
                state.gathered_grad = torch.empty(p.shape,
                                                  dtype=COMM_DTYPE,
                                                  device="cuda")
            else:
                state.gathered_grad = None

        alloc_event = torch.cuda.Event()
        alloc_event.record(compute_stream)
        return alloc_event


@torch.no_grad()
def _all2all_gather(params, param_to_state, rank, comm_stream, none_grad,
                    alloc_event):
    """
    All2all gathers shards so each owner rank reconstructs its full gradient
    """
    with torch.cuda.stream(comm_stream):
        process_group = param_to_state[id(params[0])].process_group
        num_ranks = dist.get_world_size(group=process_group)

        # Construct sending buffers
        per_dst = [[] for _ in range(num_ranks)]
        send_counts = [0] * num_ranks

        for p in params:
            state = param_to_state[id(p)]
            dst = state.worker_rank
            assert dst < num_ranks
            shard_elems = numel_for_rank(p, rank, state)
            g = p.grad
            g = g.to_local().to(COMM_DTYPE).contiguous()
            assert g.numel() == shard_elems
            per_dst[dst].append(g.view(-1))
            send_counts[dst] += shard_elems

        assert any(
            len(v) > 0 for v in per_dst
        ), "At least one destination rank must receive a sharded tensor"
        # list[list[Tensor]] -> list[Tensor]
        per_dst = [t for dst in per_dst for t in dst]

        send_buf = torch.cat(per_dst, dim=0)

        owned_params = [
            p for p in params if param_to_state[id(p)].worker_rank == rank
        ]

        # Compute receive sizes and allocate receiving buffers
        recv_counts = [0] * num_ranks

        for src in range(num_ranks):
            total = 0
            for p in owned_params:
                state = param_to_state[id(p)]
                assert state.worker_rank == rank
                total += numel_for_rank(p, src, state)
            recv_counts[src] = total

        recv_total = sum(recv_counts)
        recv_buf = torch.empty(recv_total, dtype=COMM_DTYPE, device="cuda")

        #All2All
        logger.debug(f"send_buf size: {send_buf.numel()}, "
                     f"recv_buf size: {recv_buf.numel()}, "
                     f"recv_counts: {recv_counts}, "
                     f"send_counts: {send_counts}, "
                     f"process_group: {str(process_group)}")
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=process_group,
        )

        # Reconstructs gathered grad from the received buffer
        #
        #                  recv_buf (num ranks = 3)
        #
        #      From rank 0        From rank 1        From rank 2
        # | p1_0, p2_0, p3_0 | p1_1, p2_1, p3_1 | p1_2, p2_2, p3_2 |
        #
        # Outer loop:
        # rank 0 -> rank 1 -> rank2
        #
        # Inner loop:
        # p1_n -> p2_n -> p3_n

        comm_stream.wait_event(alloc_event)

        off = 0
        for src in range(num_ranks):
            if recv_counts[src] == 0:
                continue

            block = recv_counts[src]
            inner_off = 0
            for p in owned_params:
                state = param_to_state[id(p)]
                assert state.worker_rank == rank

                # get the slice of the full dtensor corresponding to rank src.
                slices = get_slices_of_dtensor(state.gathered_grad, src,
                                               state.shard_mesh,
                                               state.shard_placements)

                dst = state.gathered_grad[slices]
                assert dst._base is state.gathered_grad

                n = dst.numel()
                assert n > 0

                sg = recv_buf.narrow(0, off + inner_off, n)
                sg = sg.reshape_as(dst)
                dst.copy_(sg)

                inner_off += n
            off += block

        for p in params:
            state = param_to_state[id(p)]
            if state.worker_rank == rank:
                state.gather_event = torch.cuda.Event()
                state.gather_event.record(comm_stream)
            else:
                state.gathered_grad = None
                state.gather_event = None
            if none_grad:
                p.grad = None


@torch.no_grad()
def _compute_u(p, state, steps, rank, compute_stream):
    """
    On worker_rank, compute the orthogonalized update using Newton-Schulz iteration.
    """
    with torch.cuda.stream(compute_stream):
        if rank == state.worker_rank:
            if state.gather_event is None:
                raise RuntimeError("Gather event must be set before compute.")
            compute_stream.wait_event(state.gather_event)
            g = state.gathered_grad
            assert g is not None
            # Per-expert batched NS for 3D (MoE), standard NS for 2D
            if g.ndim > 2:
                u = _zeropower_via_newtonschulz5_batched(g, steps)
            else:
                u = _zeropower_via_newtonschulz5(g, steps)
            state.gathered_grad = None
            state.computed_u = u
            state.compute_event = torch.cuda.Event()
            state.compute_event.record()
        else:
            state.computed_u = None
            state.compute_event = None


@torch.no_grad()
def _alloc_scattered_u(params, param_to_state, rank, compute_stream):
    """
    Pre-allocate scattered_u buffer on compute_stream
    before launching all2all gather
    """
    with torch.cuda.stream(compute_stream):
        for p in params:
            state = param_to_state[id(p)]
            state.scattered_u = torch.empty_like(p.to_local(),
                                                 dtype=COMM_DTYPE)

        alloc_event = torch.cuda.Event()
        alloc_event.record(compute_stream)
        return alloc_event


def _all2all_scatter(params, param_to_state, rank, comm_stream, alloc_event):
    """
    All2all scatters full gradients to all ranks
    """
    with torch.cuda.stream(comm_stream):
        process_group = param_to_state[id(params[0])].process_group
        num_ranks = dist.get_world_size(group=process_group)
        owned_params = [
            p for p in params if param_to_state[id(p)].worker_rank == rank
        ]

        # Construct sending buffer
        per_dst = [[] for _ in range(num_ranks)]
        send_counts = [0] * num_ranks

        if owned_params:
            for p in owned_params:
                state = param_to_state[id(p)]
                if state.compute_event is None:
                    raise RuntimeError(
                        "Compute event must be set before scatter.")
                comm_stream.wait_event(state.compute_event)
                state.gathered_grad = None

                assert state.computed_u is not None

                u_full = state.computed_u.to(COMM_DTYPE).contiguous()

                offset = 0
                for dst in range(num_ranks):
                    # get the slice of the full tensor corresponding to rank dst.
                    slices = get_slices_of_dtensor(u_full, dst,
                                                   state.shard_mesh,
                                                   state.shard_placements)
                    su = u_full[slices].flatten()

                    n = su.numel()
                    assert n > 0

                    per_dst[dst].append(su)
                    send_counts[dst] += n
                    offset += n

                assert offset == u_full.numel()

        lengths = [len(v) for v in per_dst]
        if all(l > 0 for l in lengths):
            assert all(
                l == lengths[0] for l in lengths
            ), "All destination ranks must have the same number of sharded tensor"
            # list[list[Tensor]] -> list[Tensor]
            per_dst = [t for dst in per_dst for t in dst]
            send_buf = torch.cat(per_dst, dim=0)
        else:
            # all_to_all requires participation from all ranks
            # Even non-owner ranks must join the collective call
            send_buf = torch.empty(0, dtype=COMM_DTYPE, device="cuda")

        # Compute receive sizes and allocate receiving buffers
        recv_counts = [0] * num_ranks

        for src in range(num_ranks):
            total = 0
            for p in params:
                state = param_to_state[id(p)]
                if state.worker_rank != src:
                    continue
                total += numel_for_rank(p, rank, state)
            recv_counts[src] = total

        recv_total = sum(recv_counts)
        assert recv_total > 0
        recv_buf = torch.empty(recv_total, dtype=COMM_DTYPE, device="cuda")

        #All2All
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=process_group,
        )

        # Copy to pre-allocated scattered_u buffer from the received buffer
        #
        #                  recv_buf (num ranks = 3, local_rank = 0)
        #
        #      From rank 0        From rank 1       From rank 2
        # | p1_0, p2_0, p3_0 |      p4_0       |    p5_0, p6_0    |
        #
        # Outer loop:
        # rank 0 -> rank 1 -> rank2
        #
        # Inner loop:
        # src(0) :  p1_0 -> p2_0 -> p3_0
        # src(1) :  p4_0
        # src(2) :  p5_0 -> p6_0

        comm_stream.wait_event(alloc_event)

        off = 0
        for src in range(num_ranks):
            block = recv_counts[src]
            if block == 0:
                continue

            inner_off = 0
            for p in params:
                state = param_to_state[id(p)]
                if state.worker_rank != src:
                    continue
                n = numel_for_rank(p, rank, state)
                assert n > 0

                flat_local = recv_buf.narrow(0, off + inner_off,
                                             n).view_as(p.to_local())
                state.scattered_u.copy_(flat_local)

                state.scatter_event = torch.cuda.Event()
                state.scatter_event.record(comm_stream)
                inner_off += n

            assert inner_off == block
            off += block


def _update_param(p, state, lr, adjusted_lr, weight_decay, rank,
                  compute_stream):
    """
    Update sharded parameter p with the scattered_u.
    Only worker_rank frees computed_u.
    """
    with torch.cuda.stream(compute_stream):
        if state.scatter_event is None:
            raise RuntimeError("Scatter event must be set before update")
        compute_stream.wait_event(state.scatter_event)
        u_dtensor = DTensor.from_local(
            state.scattered_u,
            placements=p.placements,
            device_mesh=p.device_mesh,
        )

        state.scattered_u = u_dtensor

        if rank == state.worker_rank:
            # Free computed_u
            state.computed_u = None

        Muon._update_p(p, state.scattered_u, lr, adjusted_lr, weight_decay)
        state.scattered_u = None
        u_dtensor = None

        scales_full = Muon._compute_scales(
            p,
            state.qk_clip_state) if state.qk_clip_state is not None else None
        if scales_full is not None:
            # Have to slice scales_full among dim 0
            weight_slices = get_slices_of_dtensor(p, rank, state.shard_mesh,
                                                  state.shard_placements)
            ratio = p.shape[0] // scales_full.shape[0]
            scales_slice = slice(
                None if weight_slices[0].start is None else
                weight_slices[0].start // ratio,
                None if weight_slices[0].stop is None else
                weight_slices[0].stop // ratio,
                None,
            )

            scales_local = scales_full[scales_slice]
            scales_local = DTensor.from_local(
                scales_local,
                placements=p.placements,
                device_mesh=p.device_mesh,
            )
            Muon._qk_clip(p, scales_local, state.qk_clip_state.head_dim)


def default_is_muon(name, x):
    skip_keys = ["embeddings", "lm_head", "model.norm"]
    return x.ndim >= 2 and not any(key in name for key in skip_keys)


def get_default_muon_param_groups(model, is_muon_func=default_is_muon):
    muon_params, muon_names = [], []
    non_muon_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_muon_func(n, p):
            muon_params.append(p)
            muon_names.append(n)
        else:
            non_muon_params.append(p)

    return [
        {
            "params": muon_params,
            "names": muon_names,
            "use_muon": True,
        },
        {
            "params": non_muon_params,
            "use_muon": False,
        },
    ]


def parse_qk_layer(name: str) -> tuple[str | None, int]:
    """
    Parse a parameter name to check if it is a query/key projection layer
    ('wq', 'wk', 'q_proj', 'k_proj') and return (kind, layer_index).

    Returns:
        (kind, layer_idx) or (None, -1) if not matched.

    Example:
        'model.3.attn.wq.weight'      -> ('wq', 3)
        'model.5.attn.wk.weight'      -> ('wk', 5)
        'model.2.attn.q_proj.weight'  -> ('q_proj', 2)
        'model.7.attn.k_proj.weight'  -> ('k_proj', 7)
        'model.4.attn.v_proj.weight'  -> (None, -1)
    """
    parts = name.split('.')
    if len(parts) < 3:
        return None, -1

    kind = parts[-2]

    layer_idx = -1
    for part in reversed(parts):
        if part.isdigit():
            layer_idx = int(part)
            break

    if kind in ('wq', 'wk', 'q_proj', 'k_proj'):
        return kind, layer_idx

    return None, -1


@dataclass
class QKClipInfo:
    """Per-parameter dynamic info computed from config + runtime logits."""
    kind: str | None  # 'wq'/'q_proj' or 'wk'/'k_proj' or None
    indices: list[int]  # which heads to consider for clipping
    head_dim: int  # from config
    threshold: float  # from config
    logit: torch.Tensor | None


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
        model: The model to be optimized by Muon.
        is_muon_func: A function that takes a parameter and its name, and returns whether the parameter should be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        weight_decay: The weight decay for Muon and AdamW.
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        none_grad: Whether to set p.grad to None after gathering the gradients. This can save memory.
        debug: Whether to print debug information.
        clip_info : Configuration for QK clipping. Expected keys:
            - "q_indices" (list[int]): Indices of query heads to consider.
            - "k_indices" (list[int]): Indices of key heads to consider.
            - "head_dim" (int): Dimensionality of each attention head.
            - "threshold" (float): Threshold value; heads whose QK logits exceed
            this value will be scaled down.
            Default is:
                {
                    "q_indices": [],
                    "k_indices": [],
                    "head_dim": 128,
                    "threshold": 100
                }
        warmup_step : How many all2all gather, compute operations are launched in advance
                      before the corresponding all2all scatter steps begin.
                      A higher warmup_step increases memory usage but can improve
                      performance by overlapping communication.
                      Parallel muon only.
        chunk_size : Batch size of parameters to process in each
                     all2all gather/compute/scatter step.
                     Use shard ranks * DEFAULT_CHUNK_SIZE_RATIO when -1 is specified.
        use_distributed_muon: Use distributed muon by Liu et al. (2024).
                              For testing purpose only.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 adamw_betas=(0.9, 0.95),
                 adamw_eps=1e-8,
                 none_grad=True,
                 debug=False,
                 clip_config={
                     "q_indices": [],
                     "k_indices": [],
                     "head_dim": 128,
                     "threshold": 100
                 },
                 warmup_step=5,
                 chunk_size=-1,
                 use_distributed_muon=False):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            none_grad=none_grad,
            use_muon=True,
        )
        error_message = "The key 'use_muon' is not set in parameter group {idx}. Assuming all parameters in the group will use muon optimization, which may lead to unexpected behavior."
        instruction_code = "\n\n please follow this code snippet \n```optimizer = get_kernel('motif-technologies/optimizer')\n\n\nparams = optimizer.muon.get_default_muon_param_groups(model)\n\noptim = optimizer.Muon(params, ...)```"

        if isinstance(params, types.GeneratorType):
            raise ValueError(error_message.format(idx=0) + instruction_code)
        for _idx, param_group in enumerate(params):
            if param_group.get("use_muon", None) is None:
                raise ValueError(
                    error_message.format(idx=_idx) + instruction_code)

        super().__init__(params, defaults)

        self.rank = None

        self.comm_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.debug = debug
        self.clip_config = clip_config
        self.warmup_step = warmup_step
        self.chunk_size = chunk_size
        self.use_distributed_muon = use_distributed_muon

    def _calc_flops(self, G, steps):
        if G.ndim > 2:
            # Per-expert batched NS: total flops = num_experts * flops_per_expert
            num_experts = G.shape[0]
            M, N = G.shape[-2], G.shape[-1]
        else:
            assert len(G.shape) == 2
            num_experts = 1
            M, N = G.shape
        if M > N:
            M, N = N, M

        per_expert = steps * ((M**3) * 2 + (M**2 * N) * 4 + M * N * 2 + M**2 * 3)
        return num_experts * per_expert

    def adjust_lr_for_muon(self, lr, param_shape):
        # For >2D params (e.g. MoE [E, out, in]), each expert is an independent
        # [out, in] weight matrix. Use the last two dims for LR scaling.
        # For 2D, param_shape[-2:] == param_shape[:2], so this is universal.
        A, B = param_shape[-2], param_shape[-1]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def set_rank_once(self, rank):
        if self.rank is None:
            self.rank = rank
        else:
            assert self.rank == rank

    def get_shard_mesh(self, p):
        """
        Get the shard mesh for a parameter p on the given rank.
        """
        assert isinstance(
            p, DTensor), "Parallel Muon only supports DTensor parameters."

        shard_mesh, shard_pg, shard_placements = construct_shard_mesh(
            p.placements, p.device_mesh)

        # set rank with the local rank in the shard process group
        self.set_rank_once(dist.get_rank(group=shard_pg))

        return shard_mesh, shard_pg, shard_placements

    def init_state_and_assign_params(self, names, params, group, qk_logits):
        param_to_state = {}
        param_to_flops = {}

        total_flops = 0
        for p in params:
            g = p.grad
            if g is None:
                continue
            assert g.ndim >= 2, "Muon requires at least 2D parameters."

            flops = self._calc_flops(g, group["ns_steps"])
            param_to_flops[id(p)] = flops
            total_flops += flops

        if self.debug:
            print(f"Total TFLOPs for Muon: {total_flops / 1e12:.2f} TFLOPs",
                  flush=True)

        paired = list(zip(names, params))

        paired_sorted = sorted(paired,
                               key=lambda x: param_to_flops[id(x[1])],
                               reverse=True)

        names_sorted, params_sorted = zip(*paired_sorted)
        ordered_names = list(names_sorted)
        ordered_params = list(params_sorted)

        round_robin = 0
        mesh = ordered_params[0].device_mesh
        placements = ordered_params[0].placements

        shard_mesh, shard_pg, shard_placements = self.get_shard_mesh(
            ordered_params[0])
        shard_mesh_flattened = shard_mesh.mesh.flatten()
        num_ranks = dist.get_world_size(group=shard_pg)

        for n, p in zip(ordered_names, ordered_params):
            if mesh != p.device_mesh:
                raise ValueError("All parameters must be on the same mesh.")
            if placements != p.placements:
                raise ValueError("All parameters must have same placements.")

            worker_rank = shard_mesh_flattened[round_robin].item() % num_ranks
            round_robin = (round_robin + 1) % len(shard_mesh_flattened)
            qk_clip_state = self.get_qk_clip_info(n, qk_logits)

            param_to_state[id(p)] = _muon_state(
                worker_rank=worker_rank,
                process_group=shard_pg,
                shard_mesh=shard_mesh,
                shard_placements=shard_placements,
                name=n,
                qk_clip_state=qk_clip_state,
            )

        return param_to_state, ordered_params

    def base(self, names, params, group, lr, weight_decay, momentum,
             qk_logits):
        # generate weight updates in distributed fashion
        for n, p in zip(names, params):
            g = p.grad
            if g is None:
                continue
            assert g is not None

            # Momentum update is element-wise, works with any shape (2D or 3D)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if group["nesterov"]:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

            # Newton-Schulz orthogonalization — batched per-expert for 3D (MoE)
            g_ns = g.to(COMM_DTYPE)
            if g_ns.ndim > 2:
                u = _zeropower_via_newtonschulz5_batched(g_ns,
                                                         steps=group["ns_steps"])
            else:
                u = _zeropower_via_newtonschulz5(g_ns,
                                                 steps=group["ns_steps"])

            adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
            Muon._update_p(p, u, lr, adjusted_lr, weight_decay)

            qk_clip_state = self.get_qk_clip_info(n, qk_logits)

            scales_full = self._compute_scales(
                p, qk_clip_state) if qk_clip_state is not None else None
            if scales_full is not None:
                Muon._qk_clip(p, scales_full, qk_clip_state.head_dim)

    def distributed_muon(
        self,
        names: list[str],
        params: list[torch.nn.Parameter],
        group: dict[str, Any],
        lr: float,
        weight_decay: float,
        momentum: float,
        qk_logits: list[torch.Tensor | DTensor] | None,
    ):
        """ Implementation of Distributed Muon by Liu et al. """
        if qk_logits is not None:
            raise NotImplementedError("QK clipping is not supported yet")

        if isinstance(params[0], DTensor):
            shard_mesh, _, shard_placements = construct_shard_mesh(
                placements=params[0].placements,
                mesh=params[0].device_mesh,
            )

        for n, p in zip(names, params):
            g = p.grad
            if g is None:
                continue
            assert g is not None

            # Momentum update is element-wise, works with any shape (2D or 3D)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if group["nesterov"]:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf

            # Gather G
            if isinstance(p.data, DTensor):
                g = g.full_tensor()

            # Newton-Schulz orthogonalization — batched per-expert for 3D (MoE)
            g_ns = g.to(COMM_DTYPE)
            if g_ns.ndim > 2:
                u = _zeropower_via_newtonschulz5_batched(g_ns,
                                                         steps=group["ns_steps"])
            else:
                u = _zeropower_via_newtonschulz5(g_ns,
                                                 steps=group["ns_steps"])

            if isinstance(p.data, DTensor):
                slices = get_slices_of_dtensor(
                    target=p,
                    local_rank=dist.get_rank(),
                    shard_mesh=shard_mesh,
                    shard_placements=shard_placements,
                )
                u_shard = u[slices]
                u = DTensor.from_local(
                    u_shard,
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                )

            adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
            Muon._update_p(p, u, lr, adjusted_lr, weight_decay)

    def _update_g(self, p, g, group, momentum):
        # calc update
        state = self.state[p]
        buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
        torch.add(g, buf, alpha=momentum, out=buf)
        if group["nesterov"]:
            g.add_(buf, alpha=momentum)
            return g
        return buf

    @staticmethod
    def _update_p(p, u, lr, adjusted_lr, weight_decay):
        # apply weight decay
        p.data.mul_(1 - lr * weight_decay)
        # apply update
        p.data.add_(u, alpha=-adjusted_lr)

    def get_qk_clip_info(self, n, qk_logits):
        if self.clip_config is None:
            return None

        head_dim = self.clip_config.get('head_dim')
        threshold = self.clip_config.get('threshold')
        kind, layer_idx = parse_qk_layer(n)

        logit, indices = None, []
        if qk_logits is not None and kind is not None:
            logit = qk_logits[layer_idx]
            indices_key = 'q_indices' if 'q' in kind else 'k_indices'
            indices = self.clip_config.get(indices_key, []) or []

            if isinstance(logit, DTensor):
                # In TP settings, qk_logits may be DTensor
                # We convert it to full tensor here for simplicity
                logit = logit.full_tensor()

        return QKClipInfo(
            kind=kind,
            indices=indices,
            head_dim=head_dim,
            threshold=threshold,
            logit=logit,
        )

    @staticmethod
    def _compute_scales(p, qk_clip_state):
        kind = qk_clip_state.kind
        indices = qk_clip_state.indices
        head_dim = qk_clip_state.head_dim
        threshold = qk_clip_state.threshold
        logit = qk_clip_state.logit

        H_global = p.shape[0] // head_dim
        scales_full = torch.ones(H_global, device=p.data.device)
        scaling = 0

        for logit_idx, head_idx in enumerate(indices):
            v_ele = float(logit[logit_idx])
            if v_ele > threshold:
                new_scale = math.sqrt(threshold / v_ele)
                if new_scale < scales_full[head_idx]:
                    scales_full[head_idx] = new_scale
                    logger.info(
                        f"[{kind}] Head {head_idx} exceeded threshold "
                        f"(value={v_ele:.4f}, threshold={threshold:.4f}) -> applying scale={new_scale:.4f}"
                    )
                    scaling += 1

        return scales_full if scaling > 0 else None

    @staticmethod
    def _qk_clip(p, scales, head_dim):
        W = p.data.view(-1, head_dim, p.data.shape[1])
        W.mul_(scales.view(-1, 1, 1))

    def parallel(self, names, params, group, lr, weight_decay, momentum,
                 qk_logits):
        """
        Perform a parallel optimization step using Muon.
        """

        for p in params:
            g = p.grad
            if g is None:
                continue

            # Momentum update is element-wise, works with any shape (2D or 3D).
            # No flatten needed here — avoids DTensor .view() compatibility issues.
            # The 3D→2D flatten for Newton-Schulz is handled later in _compute_u().
            g = self._update_g(
                p,
                g,
                group,
                momentum=momentum,
            )
            p.grad = g

        param_to_state, ordered_params = self.init_state_and_assign_params(
            names, params, group, qk_logits)

        assert self.rank is not None

        def enqueue_all2all_gather(start_idx, chunk_size):
            target_params = ordered_params[start_idx:start_idx + chunk_size]
            if target_params:
                alloc_event = _alloc_gathered_grad(target_params,
                                                   param_to_state, self.rank,
                                                   self.compute_stream)
                _all2all_gather(target_params, param_to_state, self.rank,
                                self.comm_stream, group["none_grad"],
                                alloc_event)

        def enqueue_computes(start_idx, chunk_size):
            for p in ordered_params[start_idx:start_idx + chunk_size]:
                state = param_to_state[id(p)]
                _compute_u(p, state, group["ns_steps"], self.rank,
                           self.compute_stream)

        def enqueue_all2all_scatter(start_idx, chunk_size):
            target_params = ordered_params[start_idx:start_idx + chunk_size]
            if target_params:
                alloc_event = _alloc_scattered_u(target_params, param_to_state,
                                                 self.rank,
                                                 self.compute_stream)
                _all2all_scatter(target_params, param_to_state, self.rank,
                                 self.comm_stream, alloc_event)

        def enqueue_update_param(start_idx, chunk_size):
            for p in ordered_params[start_idx:start_idx + chunk_size]:
                state = param_to_state[id(p)]
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                _update_param(p, state, lr, adjusted_lr, weight_decay,
                              self.rank, self.compute_stream)

        if self.chunk_size == -1:
            shard_ranks = dist.get_world_size(param_to_state[id(
                params[0])].process_group)
            chunk_size = shard_ranks * DEFAULT_CHUNK_SIZE_RATIO
        elif self.chunk_size > 0:
            chunk_size = self.chunk_size
        else:
            raise ValueError("chunk_size must be -1 or a positive integer.")

        # Wait grad update
        self.comm_stream.wait_stream(torch.cuda.current_stream())

        warmup_step = self.warmup_step
        for i in range(0, warmup_step):
            enqueue_all2all_gather(i * chunk_size, chunk_size)
            enqueue_computes(i * chunk_size, chunk_size)

        for i in range(0, len(params) + chunk_size - 1, chunk_size):
            enqueue_all2all_scatter(i, chunk_size)
            enqueue_all2all_gather(i + warmup_step * chunk_size, chunk_size)
            enqueue_update_param(i, chunk_size)
            enqueue_computes(i + warmup_step * chunk_size, chunk_size)

        # Wait the last update_param to finish
        torch.cuda.current_stream().wait_stream(self.compute_stream)

    @staticmethod
    def _fused_adamw(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        max_exp_avg_sqs: list[torch.Tensor],
        state_steps: list[torch.Tensor],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float | torch.Tensor,
        weight_decay: float,
        eps: float,
        maximize: bool,
    ) -> None:
        if not params:
            return

        # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
        # treating it as a scalar.
        lr_dict = ({
            lr.device: lr
        } if isinstance(lr, torch.Tensor) and str(lr.device) != "cpu" else
                                      None)
        grouped_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
            [
                params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                state_steps
            ]  # type: ignore[list-item]
        )
        for (device, _), (
            (
                device_params_,
                device_grads_,
                device_exp_avgs_,
                device_exp_avg_sqs_,
                device_max_exp_avg_sqs,
                device_state_steps_,
            ),
                _,
        ) in grouped_tensors.items():
            device_params = cast(list[torch.Tensor], device_params_)
            device_grads = cast(list[torch.Tensor], device_grads_)
            device_exp_avgs = cast(list[torch.Tensor], device_exp_avgs_)
            device_exp_avg_sqs = cast(list[torch.Tensor], device_exp_avg_sqs_)
            device_state_steps = cast(list[torch.Tensor], device_state_steps_)

            if lr_dict is not None and device not in lr_dict:
                lr_dict[device] = lr.to(
                    device=device,
                    non_blocking=True)  # type: ignore[union-attr]
                lr = lr_dict[device]
            torch._foreach_add_(device_state_steps, 1)
            func = torch._fused_adamw_
            func(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
                device_max_exp_avg_sqs,  # type: ignore[arg-type]
                device_state_steps,
                amsgrad=amsgrad,
                lr=lr,  # type: ignore[arg-type]
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                eps=eps,
                maximize=maximize,
            )

    def _step_muon(self, group, qk_logits=None):
        params = group["params"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        names = group["names"]

        param_dtensors = []
        param_tensors = []
        name_dtensors = []
        name_tensors = []

        if self.use_distributed_muon:
            self.distributed_muon(names=names,
                                  params=params,
                                  group=group,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum,
                                  qk_logits=qk_logits)
            return

        for n, p in zip(names, params):
            if p is None or p.grad is None:
                continue
            if isinstance(p.data, DTensor):
                if all(
                        isinstance(placement, Replicate)
                        for placement in p.placements):
                    param_tensors.append(p)
                    name_tensors.append(n)
                else:
                    param_dtensors.append(p)
                    name_dtensors.append(n)
            elif isinstance(p.data, torch.Tensor):
                param_tensors.append(p)
                name_tensors.append(n)
            else:
                raise TypeError(f"Unsupported parameter type: {type(p.data)}")

        logger.debug(
            f"[Muon] {len(param_dtensors)} DTensors, {len(param_tensors)} Tensors"
        )

        if len(param_dtensors) > 0:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Parallel Muon requires torch.distributed to be initialized."
                )

            # To support different placements, we group parameters by placements
            # and run parallel Muon on each group.

            placement_to_params = defaultdict(lambda: ([], []))
            # type: dict[tuple[Placement, DeviceMesh], tuple[list[str], list[DTensor]]]

            assert len(name_dtensors) == len(param_dtensors)
            for n, p in zip(name_dtensors, param_dtensors):
                placement_to_params[tuple([p.placements,
                                           p.device_mesh])][0].append(n)
                placement_to_params[tuple([p.placements,
                                           p.device_mesh])][1].append(p)

            for _, (names, params) in placement_to_params.items():
                self.parallel(
                    names,
                    params,
                    group,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    qk_logits=qk_logits,
                )

        if len(param_tensors) > 0:
            self.base(
                name_tensors,
                param_tensors,
                group,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                qk_logits=qk_logits,
            )

    def _step_adamw_params(self, params, group):
        params_with_grads = []
        grads = []
        moment1 = []
        moment2 = []
        max_exp_avg_sqs = []
        state_steps = []
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["weight_decay"]

        for p in params:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            params_with_grads.append(p)
            grads.append(g)
            if "step" not in state:
                state["step"] = (torch.zeros((),
                                             dtype=torch.float32,
                                             device=p.device))
                state["moment1"] = torch.zeros_like(g)
                state["moment2"] = torch.zeros_like(g)
            moment1.append(state["moment1"])
            moment2.append(state["moment2"])
            if not isinstance(state["step"], torch.Tensor):
                step_tensor = torch.tensor(state["step"],
                                           dtype=torch.float32,
                                           device=p.device)
            else:
                step_tensor = state["step"]
            state_steps.append(step_tensor)

        self._fused_adamw(
            params_with_grads,
            grads,
            moment1,
            moment2,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
        )

    def _step_adamw(self, group):
        params = group["params"]

        # group params with it's type and placement
        placement_to_params: dict[tuple[Placement | type,
                                        DeviceMesh | None]] = defaultdict(list)
        for p in params:
            match p:
                case DTensor():
                    placement_to_params[tuple([p.placements,
                                               p.device_mesh])].append(p)
                case torch.Tensor():
                    placement_to_params[tuple([torch.Tensor, None])].append(p)

        for params in placement_to_params.values():
            self._step_adamw_params(params, group)

    def step(self, closure=None, qk_logits=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
            qk_logits (dict[int, Tensor], optional): A dictionary mapping layer indices
                to 1D tensors of shape (num_heads,), representing the maximum
                QK logits across all tokens, computed as
                (1 / sqrt(head_dim)) * (Q @ K^T).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group, qk_logits=qk_logits)
            else:
                self._step_adamw(group)

        return loss
