import os

from pprint import pprint
from abc import abstractmethod
from typing import List, Dict, Tuple, Optional

import ray
import torch
import torch.distributed as dist
import numpy as np

from .registry import EngineRegistry
from .model_executor import ModelExecutor, TensorParallelModelExecutor
from .utils.parallel_group_utils import ParallelStatus, rebuild_tensor_from_ipc_info
from .utils.common import profile_nvtx, print_debug
from .request import FlashInferMetadata
from .sampler import SamplingParams

DEBUG = False

class ModelRunner:
    """
    TODO: apply registry
    """
    def __init__(
        self, 
        registry: EngineRegistry, 
        type = None,
        local_rank: int = -1,
    ):
        self.registry = registry
        self.type = type if type is not None else None
        self.local_rank = local_rank

        if self.type is None:
            model_type = "model"
        else:
            model_type = f"{type}_model"

        self.model_type = model_type
        self.model_name_or_path = self.registry.get(f'val.{model_type}.model_name_or_path')
        self.model_executors = None
        self.is_ray_runner = None

        self.use_ray_worker = self.registry.get('val.engine.use_ray_worker')
        self.use_ray_executor = self.registry.get('val.engine.use_ray_executor')
        self.use_data_parallel_draft = self.registry.get('val.engine.use_data_parallel_draft')
        self.use_cache_offloading = self.registry.get('val.engine.use_cache_offloading') if self.model_type != 'model' else False

        self.device_ids = registry.get(f'val.{model_type}.device_ids') 

        
        self.parallel_status = ParallelStatus(
            global_world_size = registry.get(f'val.engine.global_world_size'),
            global_ranks = registry.get(f'val.{model_type}.global_ranks'),
            device_ids = self.device_ids,
            local_rank = self.local_rank if self.use_ray_worker else None,
            tensor_parallel_size = registry.get(f'val.{model_type}.tensor_parallel_size'),
            pipeline_parallel_size = registry.get(f'val.{model_type}.pipeline_parallel_size'),
        )

        if self.use_ray_executor and self.model_type == "model":
            self.is_ray_runner = True
            self._init_distributed_model_executors()
        else:
            self.is_ray_runner = False
            self._init_single_model_executor()

    def ensure_download_model(self):
        from transformers import AutoModelForCausalLM
        if self.registry.get(f'val.{self.model_type}.world_size') > 1:
            model_name_or_path = self.registry.get(f'val.{self.model_type}.model_name_or_path')
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=self.model_cache_dir, use_safetensors=True)
            del model


    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def _init_single_model_executor(self):
        if self.parallel_status.tensor_parallel_size > 1:
            executor_cls = TensorParallelModelExecutor
        else:
            executor_cls = ModelExecutor

        self.model_executors = [executor_cls(
            registry=self.registry, 
            type=self.type, 
            parallel_status=self.parallel_status,
            local_rank=self.local_rank
        )]
        self.run = self.run_model

    def _init_distributed_model_executors(self):
        import ray
        print(f"\n-------- ModelRunner._init_distributed_model_executors --------")
        print(f"self.parallel_status: {self.parallel_status}")

        runtime_env = {}
        if self.registry.get('val.engine.enable_nsight'):
            print("Enable nsight")
            os.makedirs("./nsys", exist_ok=True)
            runtime_env.update({
                "nsight": {
                    "t": "cuda,osrt,nvtx,cudnn,cublas",
                    "o": "./nsys/worker_process_%p",
                    "cuda-graph-trace": "graph", # "node",
                    "cuda-memory-usage": "true",
                    "w": "true",
                    "f": "true",
                }})

        self.model_executors = []

        if self.use_ray_worker:
            device_id = self.parallel_status.device_ids[self.local_rank]
            runtime_env.update({
                    "env_vars": {
                        "TORCH_CUDA_ARCH_LIST": os.environ.get('TORCH_CUDA_ARCH_LIST', None),
                        "CUDA_VISIBLE_DEVICES": f"{device_id}",
                        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.registry.get('val.engine.verify_mps_percentage')),  # SM 사용 비율 100% 설정 
                        "CUDA_MPS_CLIENT_PRIORITY": "0"              # Normal 우선순위 힌트 
                    }
                })
            ray_actor_cls = ray.remote(
                num_cpus=1,
                num_gpus=0.1,
                resources={f"GPU_{device_id}": 0.1},
                runtime_env=runtime_env,
            )(TensorParallelModelExecutor)
            
            print(f"Spawning TensorParallelModelExecutor for local_rank {self.local_rank} with device_id {device_id}, global_rank {self.parallel_status.global_ranks[self.local_rank]}")
            ray_actor = ray_actor_cls.remote(
                registry=self.registry, 
                type=self.type,
                parallel_status=self.parallel_status,
                local_rank=self.local_rank,
            )
            self.model_executors.append(ray_actor)

        else:    
            for local_rank, device_id in enumerate(self.parallel_status.device_ids):
                runtime_env.update({
                    "env_vars": {
                        "TORCH_CUDA_ARCH_LIST": os.environ.get('TORCH_CUDA_ARCH_LIST', None),
                        "CUDA_VISIBLE_DEVICES": f"{device_id}",       # Normal 우선순위 힌트 
                    }
                })
                ray_actor_cls = ray.remote(
                    num_cpus=1,
                    num_gpus=0.1,
                    resources={f"GPU_{device_id}": 0.1},
                    runtime_env=runtime_env,
                )(TensorParallelModelExecutor)
                
                print(f"Spawning TensorParallelModelExecutor for local_rank {local_rank} with device_id {device_id}, global_rank {self.parallel_status.global_ranks[local_rank]}")
                ray_actor = ray_actor_cls.remote(
                    registry=self.registry, 
                    type=self.type,
                    parallel_status=self.parallel_status,
                    local_rank=local_rank,
                )
                self.model_executors.append(ray_actor)

        self.run = self.run_distributed_model
            
    def ensure_init_done(self):
        if self.registry.get('val.engine.use_ray_executor'):
            ray.get([executor.ensure_init_done.remote() for executor in self.model_executors])
        return True
    
    @profile_nvtx("ModelRunner.profile_latency")
    def profile_latency(self):
        if self.registry.get('val.engine.use_ray_executor'):
            futures = [
                executor.profile_latency.remote() 
                for executor in self.model_executors
            ]
            return ray.get(futures)[0]
        else:
            return self.model_executors[0].profile_latency()

    @profile_nvtx("ModelRunner.run_model")
    def run_model(
            self, 
            input_ids_tensor: torch.Tensor,
            position_ids_tensor: torch.Tensor,
            flashinfer_metadata: FlashInferMetadata,
            sampling_params: List[SamplingParams],
            num_logits_to_keep: int = None,
            **kwargs
        ):
        outputs = self.model_executors[0].run_model(
            input_ids_tensor=input_ids_tensor,
            position_ids_tensor=position_ids_tensor,
            flashinfer_metadata=flashinfer_metadata,
            sampling_params=sampling_params,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs
        )
        return outputs

    def onloading_cache(self):
        if self.use_cache_offloading:
            self.model_executors[0].onloading_cache()

    def offloading_cache(self):
        if self.use_cache_offloading:
            self.model_executors[0].offloading_cache()
    
    def eagle3_vocab_mapping(self, index):
        return self.model_executors[0].eagle3_vocab_mapping(index)

    @profile_nvtx("ModelRunner.run_distributed_model")
    def run_distributed_model(
            self,
            input_ids_tensor: torch.Tensor,
            position_ids_tensor: torch.Tensor,
            flashinfer_metadata: FlashInferMetadata,
            sampling_params: List[SamplingParams],
            num_logits_to_keep: int = None,
            run_async=False,
            **kwargs
        ):
        
        futures = [
            executor.run_model.remote(
                input_ids_tensor=input_ids_tensor, 
                position_ids_tensor=position_ids_tensor, 
                flashinfer_metadata=flashinfer_metadata,
                num_logits_to_keep=num_logits_to_keep,
                sampling_params=sampling_params,
                **kwargs
            ) 
            for executor in self.model_executors
        ]
        
        if run_async:
            return futures
        else:
            return futures
        # return self.get_ray_model_results(futures)
        

    def ensure_ray_futures_done(self, futures):
        results = ray.get(futures)
        return results


    @profile_nvtx("ModelRunner.get_distributed_model_results")
    def get_ray_model_results(self, futures):
        worker_results = ray.get(futures)
        del futures

        results = worker_results[0]

        outputs = results
        outputs[1] = rebuild_tensor_from_ipc_info(outputs[1])
        return outputs