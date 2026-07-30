import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    Placement,
    Shard,
    _StridedShard,
)


def get_slices_of_dtensor(
    target: DTensor | torch.Tensor,
    local_rank: int,
    shard_mesh: DeviceMesh,
    shard_placements: tuple[Placement],
) -> tuple[slice]:
    """
    Get the slice of local tensor for a given rank from a tensor.
    Args:
        target (DTensor | torch.Tensor): The target tensor.
        rank (int): The local rank of the shard group.
        shard_mesh (DeviceMesh): The shard mesh. It consists of global ranks.
        shard_placements (tuple[Placement]): The shard placements.
    """

    slices: list[slice] = [slice(0, dim_size) for dim_size in target.size()]

    # find the global rank of the local rank in the shard mesh
    rank = sorted(shard_mesh.mesh.flatten().tolist())[local_rank]

    rank_coords = (shard_mesh.mesh == rank).nonzero()

    assert len(rank_coords) == 1
    rank_coords = tuple(rank_coords[0].tolist())

    assert len(rank_coords) == len(shard_placements)

    # Caution: Assuming replicate-to-shard of the shard mesh goes with
    # left-to-right sharding. This is ensured by the sorting logic of
    # construct_shard_mesh function.
    for i, (rank_coord,
            placement) in enumerate(zip(rank_coords, shard_placements)):
        assert isinstance(placement, Shard)

        num_ranks = shard_mesh.mesh.shape[i]

        dim = placement.dim
        dim_size = (slices[dim].stop - slices[dim].start)

        if dim_size % num_ranks != 0:
            raise NotImplementedError(
                f"Dimension size {dim_size} is not divisible "
                f"by number of ranks {num_ranks} for shard "
                f"placement on dim {dim}.")

        shard_size = dim_size // num_ranks

        start = slices[dim].start + rank_coord * shard_size
        end = start + shard_size

        assert start < end <= slices[dim].stop

        slices[dim] = slice(start, end)

    return tuple(slices)


_ranks_to_dist_cache: dict[tuple[int, ...], tuple[DeviceMesh, ProcessGroup]] = dict()


def construct_shard_mesh(
    placements: tuple[Placement],
    mesh: DeviceMesh,
) -> (DeviceMesh, ProcessGroup, tuple[Placement]):
    """
    Construct Shard Mesh and Placements for unsharding.
    It removes Replicate placements and constructs a new Mesh and ProcessGroup.
    """
    my_rank = dist.get_rank()

    assert mesh.mesh.device.type == 'cpu'

    # Copy mesh to avoid modifying the original mesh
    mesh = mesh.mesh.clone()

    # 1. Sort placements. Replicate first, then Shard by dim ascending.

    # For Shard, strided shard comes after regular shard on the same dim
    # to preserve left-to-right order of replicate-to-shard.
    # This is because that strided shard is using stride to represent
    # more fine-grained sharding on the same dim.
    # Please check the URL below for _StridedShard.
    # https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/tensor/placement_types.py#L366

    def placement_sort_key(
        placement_with_index: tuple[float, Placement]
    ) -> tuple[int, float, int]:  # (dim, split factor, original index)
        index, placement = placement_with_index
        is_replicate = placement.is_replicate()
        is_shard = placement.is_shard()
        is_partial = placement.is_partial()

        assert is_replicate or is_shard, f"Unsupported placement type: {type(placement)}"
        assert not is_partial, "Partial placement is not supported."

        if is_replicate:
            return (-1.0, 0, index)
        elif is_shard:
            if isinstance(placement, _StridedShard):
                return (placement.dim, 1 / placement.split_factor, index)
            return (placement.dim, 0, index)
        else:
            raise TypeError(f"Unknown placement type: {type(placement)}")

    placements_with_index: list[tuple[int,
                                      Placement]] = list(enumerate(placements))
    placements_with_index = sorted(placements_with_index,
                                   key=placement_sort_key)

    sorted_indices, sorted_placements = zip(*placements_with_index)

    # 2. Permute mesh according to sorted placements.
    sorted_mesh = mesh.permute(sorted_indices)

    # 3. Collect list of shard meshes by removing replicate dims
    # For example, (2, 3, 4, 4) with placements [R, R, S(0), S(1)]
    # shard_meshes should be list with 2 * 3 = 6 shard meshes of shape (4, 4)
    num_replicates = sum(1 for p in sorted_placements if p.is_replicate())

    # merge replicate dims
    # shard_meshes became a list of shard meshes with a length of replicate degree
    if num_replicates > 0:
        sorted_mesh = sorted_mesh.flatten(
            0, num_replicates - 1) if num_replicates > 1 else sorted_mesh
        shard_meshes = list(torch.unbind(sorted_mesh, dim=0))
    else:
        shard_meshes = [sorted_mesh]
    shard_placements = sorted_placements[num_replicates:]

    # assume all shard placements are different
    assert len(shard_placements) == len(set(shard_placements))

    # 4. Construct ProcessGroups
    # Caution: all groups should be created in the same order in all processes,
    # even though each process only needs its own group.

    # To use tensor as dict key, convert it to tuple
    def tensor_to_tuple(t):
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        if isinstance(t, list):
            return tuple(tensor_to_tuple(x) for x in t)
        return t

    my_shard_mesh_as_tuple = None
    for shard_mesh in shard_meshes:
        assert isinstance(shard_mesh, torch.Tensor)
        shard_mesh_as_tuple = tensor_to_tuple(shard_mesh)

        if (my_rank == shard_mesh).any().item():
            assert my_shard_mesh_as_tuple is None
            my_shard_mesh_as_tuple = shard_mesh_as_tuple

        # update global cache
        if shard_mesh_as_tuple not in _ranks_to_dist_cache:
            shard_process_group = dist.new_group(shard_mesh.flatten().tolist())
            _ranks_to_dist_cache[shard_mesh_as_tuple] = (
                DeviceMesh(device_type="cuda", mesh=shard_mesh),
                shard_process_group,
            )

    my_shard_mesh, my_shard_process_group = _ranks_to_dist_cache[
        my_shard_mesh_as_tuple]

    return my_shard_mesh, my_shard_process_group, shard_placements
