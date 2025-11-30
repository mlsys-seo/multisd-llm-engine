from typing import Optional, List
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from sdprofiler.utils.common import print_debug

def get_global_rank():
    assert dist.is_initialized()
    return dist.get_rank(group=get_global_group())


def get_global_world_size():
    assert dist.is_initialized()
    return dist.get_world_size(group=get_global_group())


def get_global_group():
    assert dist.is_initialized()
    if not hasattr(get_global_group, '_group'):
        get_global_group._group = dist.new_group()
    return get_global_group._group



"""
global_world_size: number of all the ray actors (across all Ray Actors(num_workers * num_model_runners))
global_rank: rank of all the ray actors (across all Ray Actors(num_workers * num_model_runners))

local_parallel_group: parallel group of the ray actors in current model_runners (tp/pp ranks)
local_world_size: number of the ray actors in current model_runners (tp/pp ranks)
local_rank: rank of the ray actors in current model_runners (tp/pp ranks)

pp_world_size: number of the model_runners in current model_runners (pp ranks)
pp_rank: rank of the model_runners in current model_runners (pp ranks)

tp_world_size: number of the model_runners in current model_runners (tp ranks)
tp_rank: rank of the model_runners in current model_runners (tp ranks)

device_id: multiple Ray Actors can share the same GPU, so we need to specify the device_id for each Ray Actor.
"""


def build_model_parallel_groups(local_rank, global_ranks, tp_size, pp_size):

    if tp_size > 1:
        tp_groups = [global_ranks[i:i + tp_size] for i in range(0, len(global_ranks), tp_size)]

        for tp_group in tp_groups:
            current_tp_group = tp_group
            if local_rank in current_tp_group:
                break
        print(f"local_rank: {local_rank}, global_ranks: {global_ranks}, current_tp_group: {current_tp_group}")
        tp_group_obj = dist.new_group(current_tp_group)
    else:
        tp_group_obj = None
    
    if pp_size > 1:
        pp_groups = [global_ranks[i:i + pp_size] for i in range(0, len(global_ranks), pp_size)]

        for pp_group in pp_groups:
            current_pp_group = pp_group
            if local_rank in current_pp_group:
                break
        print(f"local_rank: {local_rank}, global_ranks: {global_ranks}, current_pp_group: {current_pp_group}")
        pp_group_obj = dist.new_group(current_pp_group)
    else:
        pp_group_obj = None

    
    return tp_group_obj, pp_group_obj



@dataclass
class ParallelStatus:
    global_world_size: int
    global_ranks: List[int] # index는 local_rank
    device_ids: List[int] # index는 local_rank
    local_rank: int
    tensor_parallel_size: int
    pipeline_parallel_size: int

    @property
    def local_world_size(self):
        return len(self.global_ranks)


class ExecutorParallelGroup:
    def __init__(self, 
                 global_world_size,
                 global_ranks, 
                 device_ids, 
                 local_rank,
                 tensor_parallel_size, 
                 pipeline_parallel_size):
        self.global_world_size = global_world_size
        self.global_ranks = global_ranks
        self.device_ids = device_ids
        self.local_rank = local_rank
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size

        self.tp_group = None
        self.pp_group = None
        self.local_group = None

    def build_model_parallel_groups(self):
        self.tp_group, self.pp_group = build_model_parallel_groups(self.local_rank, self.global_ranks, self.tensor_parallel_size, self.pipeline_parallel_size)
        if len(self.global_ranks) > 1:
            self.local_group = dist.new_group(self.global_ranks)

        # warmup
        self.barrier(self.local_group)
        self.all_reduce(torch.zeros(1).cuda(), self.local_group)
        self.barrier(self.tp_group)
        self.all_reduce(torch.zeros(1).cuda(), self.tp_group)
        self.barrier(self.pp_group)
        self.all_reduce(torch.zeros(1).cuda(), self.pp_group)

    @property
    def world_size(self):
        return len(self.global_ranks)
    
    @property
    def global_rank(self):
        return self.global_ranks[self.local_rank]
    
    @property
    def device_id(self):
        return self.device_ids[self.local_rank]
    

    # def all_gather_and_concat_numpy_arrays(
    #     self,
    #     local_arrays: List[np.ndarray],
    #     group: Optional[dist.ProcessGroup] = None
    # ) -> np.ndarray:
    #     world_size = dist.get_world_size(group)

    #     # 1) numpy 배열 → 파이썬 리스트(리스트 안의 리스트)
    #     local_lists: List[List] = [arr.tolist() for arr in local_arrays]

    #     # 2) 모든 rank의 리스트 수집
    #     gathered: List[List[List]] = [None] * world_size
    #     dist.all_gather_object(gathered, local_lists, group=group)

    #     # 3) 중첩된 리스트(flat)으로 풀어서 하나의 값 리스트로 변환
    #     flat_vals: List = []
    #     for rank_lists in gathered:
    #         for arr_list in rank_lists:
    #             flat_vals.extend(arr_list)

    #     # 4) numpy 배열로 복원
    #     #    (dtype은 첫 배열을 참조; 필요에 따라 별도 인자화 가능)
    #     return np.array(flat_vals, dtype=local_arrays[0].dtype)


    def broadcast_object_list(
        self,
        src: int = 0,
        object_list: Optional[list] = None,
        group: torch.distributed.ProcessGroup=None
    ) -> list:
        if object_list is None or src != self.local_rank:
            object_list = [None] * len(self.global_ranks)
        if group is None:
            group = self.tp_group

        dist.broadcast_object_list(object_list, src=src, group=group)
        torch.distributed.barrier(group=group)
        return object_list


    def all_reduce(self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup=None):
        assert dist.is_initialized()
        if group is None:
            group = self.tp_group
        if group is None:
            return tensor
        dist.all_reduce(tensor, group=group)
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: torch.distributed.ProcessGroup=None):
        assert dist.is_initialized()
        if group is None:
            group = self.tp_group
        if group is None:
            return tensor
        tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(tensor_list, tensor, group=group)
        return torch.cat(tensor_list, dim=-1)
    
    def all_gather_into_tensor(self, output_tensor: torch.Tensor, local_tensor: torch.Tensor, group: torch.distributed.ProcessGroup):
        assert dist.is_initialized()
        if group is None:
            group = self.tp_group
        if group is None:
            return local_tensor
        dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
        return output_tensor

    def send(self, tensor: torch.Tensor, target_rank: int, group: torch.distributed.ProcessGroup):
        assert dist.is_initialized()
        dist.send(tensor, dst=target_rank, group=group)

    def recv(self, tensor: torch.Tensor, source_rank: int, group: torch.distributed.ProcessGroup):
        assert dist.is_initialized()
        dist.recv(tensor, src=source_rank, group=group)
        return tensor
    
    def barrier(self, group: torch.distributed.ProcessGroup=None):
        assert dist.is_initialized()
        if group is None:
            group = self.local_group
        if group is None:
            return
        dist.barrier(group=group)

    def __str__(self):
        return f"ExecutorParallelGroup(" \
            f"global_world_size: {self.global_world_size}, " \
            f"global_ranks: {self.global_ranks}, " \
            f"device_ids: {self.device_ids}, " \
            f"local_rank: {self.local_rank}, " \
            f"tensor_parallel_size: {self.tensor_parallel_size}, " \
            f"pipeline_parallel_size: {self.pipeline_parallel_size} " \
            f")"

RunnerParallelGroup = ExecutorParallelGroup

class ParallelGroup:
    '''
    1개의 Model Runner에 대응
    '''
    def __init__(self, tp_ranks):

        self.ranks = tp_ranks   # TODO: For pipeline parallel
        self.world_size = len(self.ranks)
        self.tp_ranks = tp_ranks
        self.tp_group = None
    
    def get_local_rank(self, global_rank: Optional[int] = None):
        start_rank = self.ranks[0]
        if global_rank is None:
            global_rank = get_global_rank()
        return global_rank - start_rank

    def all_reduce(self, tensor: torch.Tensor):
        assert dist.is_initialized()

        if self.tp_group is None:
            self.tp_group = dist.new_group(self.tp_ranks)

        dist.all_reduce(tensor, group=self.tp_group)

        return tensor

    def all_gather(self, tensor: torch.Tensor):
        assert dist.is_initialized()

        if self.tp_group is None:
            self.tp_group = dist.new_group(self.tp_ranks)

        tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=self.tp_group))
        ]
        dist.all_gather(tensor_list, tensor, group=self.tp_group)

        return torch.cat(tensor_list, dim=-1)
    
    def __str__(self):
        return f"ParallelGroup(ranks={self.ranks})"


class ParallelGroupManager:
    def __init__(self):
        # key : ModelRunner instance, value : ParallelGroup instance
        # TODO: 이렇게 하면 Ray Init할 때 Serialize 되는 데이터가 너무 커질 것 같음..
        self.parallel_groups = {}
        self.joined_num_ranks = 0

    def register_new_parallel_group(self, registry, type=None):
        if type is None:
            model_type = "model"
        else:
            model_type = f"{type}_model"

        model_runner_world_size = registry.get(f'val.{model_type}.world_size')
        model_runner_tp_size = registry.get(f'val.{model_type}.tensor_parallel_size')
        model_runner_pp_size = registry.get(f'val.{model_type}.pipeline_parallel_size')
        
        assert model_runner_world_size == model_runner_tp_size * model_runner_pp_size

        if model_type not in self.parallel_groups.keys(): 
            start_rank = self.joined_num_ranks
            self.joined_num_ranks += model_runner_world_size

            tp_ranks = list(range(
                start_rank, 
                start_rank + model_runner_tp_size
            ))
            self.parallel_groups[model_type] = ParallelGroup(tp_ranks)

        return self.parallel_groups[model_type]
    
    def get_parallel_group(self, model_type):
        assert model_type in self.parallel_groups.keys()

        return self.parallel_groups[model_type]

    def get_global_world_size(self):
        global_world_size = 0
        for _, parallel_group in self.parallel_groups.items():
            global_world_size += parallel_group.world_size

        return global_world_size
    
    def __str__(self):
        return f"ParallelGroupManager(parallel_groups={self.parallel_groups})"


def get_tensor_ipc_info(tensor):
    tensor_ipc_info = tensor.untyped_storage()._share_cuda_()

    tensor_info = (
        type(tensor), tensor.size(), tensor.stride(),
        tensor.storage_offset(), tensor.dtype,
        type(tensor.untyped_storage()))

    tensor_ipc_info = tensor_info + tensor_ipc_info

    (tensor_cls, tensor_size, tensor_stride,
    tensor_offset, tensor_dtype,
    storage_cls, device, handle,
    storage_size_bytes, storage_offset_bytes,
    ref_counter_handle, ref_counter_offset,
    event_handle, event_sync_required) = tensor_ipc_info

    return tensor_ipc_info


def rebuild_tensor_from_ipc_info(ipc_info):
    (tensor_cls, tensor_size, tensor_stride,
    tensor_offset, tensor_dtype,
    storage_cls, device, handle,
    storage_size_bytes, storage_offset_bytes,
    ref_counter_handle, ref_counter_offset,
    event_handle, event_sync_required) = ipc_info
    
    storage = storage_cls._new_shared_cuda(
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required
    )

    rebuilded_tensor = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=storage.untyped(), dtype=tensor_dtype),
        tensor_offset, tensor_size, tensor_stride
    )

    return rebuilded_tensor
