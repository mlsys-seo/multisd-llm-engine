import os
import time
import json
import copy
import atexit
import itertools

from collections import deque
from datetime import timedelta
from typing import List, Union, Optional, Dict
from dataclasses import dataclass, field, fields

import ray
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, GenerationConfig

from .config import ModelConfig, ParallelConfig, EngineConfig
from .registry import EngineRegistry
from .request import Request
from .scheduler import Scheduler
from .sampler import SamplingParams
from .utils.common import print_debug, profile_nvtx, start_nsys_profile, stop_nsys_profile, int_key_object_hook

DEBUG = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class EngineStat:
    num_iterations: int = 0

    warmup_start_idx: int = None
    warmup_end_idx: int = None
    
    expected_throughputs: Optional[List[Dict[str, np.ndarray]]] = field(default_factory=lambda: [])

    is_prefill_step: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=bool))
    num_requests: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    time_elapsed: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=float))
    num_generated_tokens: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    num_processed_tokens: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    
    prefill_indices: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    generation_indices: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))

    scheduling_num_expected_tokens: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    scheduling_num_selected_drafting_tokens: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    scheduling_expected_latency: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))
    scheduling_expected_throughput: Optional[np.ndarray] = field(default_factory=lambda: np.array([], dtype=int))

    def register_scheduling_stats(self, scheduling_stats: List[Dict[str, np.ndarray]]):
        if len(scheduling_stats) == 0 and scheduling_stats is None:
            return
        try:
            num_expected_tokens = np.array([ item['num_expected_tokens_per_batch'].sum() for item in scheduling_stats ])
            expected_latency = np.array([ item['expected_latency'] for item in scheduling_stats ])
            expected_throughput = np.array([ item['expected_throughput'] for item in scheduling_stats ])

            self.scheduling_num_expected_tokens = np.concatenate((self.scheduling_num_expected_tokens, num_expected_tokens))
            self.scheduling_expected_latency = np.concatenate((self.scheduling_expected_latency, expected_latency))
            self.scheduling_expected_throughput = np.concatenate((self.scheduling_expected_throughput, expected_throughput))
        except Exception as e:
            print(f"Error registering scheduling stats: {e}")

    def register_start_warmup(self):
        if self.warmup_start_idx is None:
            self.warmup_start_idx = self.num_iterations

    def register_end_warmup(self):
        if self.warmup_end_idx is None:
            self.warmup_end_idx = self.num_iterations

    def register_step(self, stat: Dict[str, np.ndarray]):
        self.num_requests = np.append(self.num_requests, stat['num_requests'])
        self.time_elapsed = np.append(self.time_elapsed, stat['time_elapsed'])
        self.num_generated_tokens = np.append(self.num_generated_tokens, stat['num_generated_tokens'])
        self.num_processed_tokens = np.append(self.num_processed_tokens, stat['num_processed_tokens'])
        self.is_prefill_step = np.append(self.is_prefill_step, stat['is_prefill_step'])

        if stat['is_prefill_step']:
            self.prefill_indices = np.append(self.prefill_indices, self.num_iterations)
        else:
            self.generation_indices = np.append(self.generation_indices, self.num_iterations)
        self.num_iterations += 1

        
    def register_expected_throughput(self, expected_throughput: Dict[str, np.ndarray]):
        self.expected_throughputs.append(expected_throughput)

    def clear(self):
        self.num_iterations = 0

        self.warmup_start_idx = None
        self.warmup_end_idx = None
        
        self.expected_throughputs = []

        self.is_prefill_step = np.array([], dtype=bool)
        self.num_requests = np.array([], dtype=int)
        self.time_elapsed = np.array([], dtype=float)
        self.num_generated_tokens = np.array([], dtype=int)
        self.num_processed_tokens = np.array([], dtype=int)
        
        self.prefill_indices = np.array([], dtype=int)
        self.generation_indices = np.array([], dtype=int)

        self.scheduling_num_expected_tokens = np.array([], dtype=int)
        self.scheduling_num_selected_drafting_tokens = np.array([], dtype=int)
        self.scheduling_expected_latency = np.array([], dtype=int)
        self.scheduling_expected_throughput = np.array([], dtype=int)




    def get_prefill_indices(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices
        return prefill_indices

    def get_generation_indices(self, apply_warmup: bool = False):
        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices
        return generation_indices

    def get_engine_prefill_time(self, apply_warmup: bool = False):
        prefill_indices = self.get_prefill_indices(apply_warmup)
        prefill_time = self.time_elapsed[prefill_indices].sum()
        return prefill_time

    def get_engine_generation_time(self, apply_warmup: bool = False):
        generation_indices = self.get_generation_indices(apply_warmup)
        generation_time = self.time_elapsed[generation_indices].sum()
        return generation_time
    
    def get_engine_emitted_throughput(self, apply_warmup: bool = False):
        if apply_warmup:
            all_time_elapsed = self.time_elapsed[self.warmup_start_idx:self.warmup_end_idx].sum()
            all_num_tokens = self.num_generated_tokens[self.warmup_start_idx:self.warmup_end_idx].sum()
            return all_num_tokens / all_time_elapsed
        else:
            all_time_elapsed = self.time_elapsed.sum()
            all_num_tokens = self.num_generated_tokens.sum()
            return all_num_tokens / all_time_elapsed

    def get_engine_prefill_throughput(self, apply_warmup: bool = False):
        prefill_indices = self.get_prefill_indices(apply_warmup)
        return self.num_generated_tokens[prefill_indices].sum() / self.time_elapsed[prefill_indices].sum()

    def get_engine_prefill_processed_throughput(self, apply_warmup: bool = False):
        prefill_indices = self.get_prefill_indices(apply_warmup)
        return self.num_processed_tokens[prefill_indices].sum() / self.time_elapsed[prefill_indices].sum()

    def get_engine_generation_throughput(self, apply_warmup: bool = False):
        generation_indices = self.get_generation_indices(apply_warmup)
        return self.num_generated_tokens[generation_indices].sum() / self.time_elapsed[generation_indices].sum()


    ''' Statistics by step '''
    def get_num_prefill_steps(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices

        return len(prefill_indices)

    def get_num_generation_steps(self, apply_warmup: bool = False):
        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices

        return len(generation_indices)

    def get_prefill_num_requests_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices

        return self.num_requests[prefill_indices]

    def get_prefill_time_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices

        return self.time_elapsed[prefill_indices]

    def get_prefill_num_generated_tokens_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices

        return self.num_generated_tokens[prefill_indices]

    def get_prefill_num_processed_tokens_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            prefill_indices = self.prefill_indices[ (self.prefill_indices >= self.warmup_start_idx) & (self.prefill_indices <= self.warmup_end_idx) ]
        else:
            prefill_indices = self.prefill_indices

        return self.num_processed_tokens[prefill_indices]
        
    def get_generation_num_requests_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices

        return self.num_requests[generation_indices]

    def get_generation_time_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices

        return self.time_elapsed[generation_indices]

    def get_generation_num_generated_tokens_per_step(self, apply_warmup: bool = False):
        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices

        return self.num_generated_tokens[generation_indices]
    
    def get_prefill_throughputs_per_step(self, apply_warmup: bool = False):
        return self.get_prefill_num_generated_tokens_per_step(apply_warmup) / self.get_prefill_time_per_step(apply_warmup)
    
    def get_prefill_processed_throughputs_per_step(self, apply_warmup: bool = False):
        return self.get_prefill_num_processed_tokens_per_step(apply_warmup) / self.get_prefill_time_per_step(apply_warmup)
    
    def get_generation_throughputs_per_step(self, apply_warmup: bool = False):
        return self.get_generation_num_generated_tokens_per_step(apply_warmup) / self.get_generation_time_per_step(apply_warmup)

    def get_target_generation_throughputs_per_step(self, apply_warmup: bool = False):

        if apply_warmup:
            generation_indices = self.generation_indices[ (self.generation_indices >= self.warmup_start_idx) & (self.generation_indices <= self.warmup_end_idx) ]
        else:
            generation_indices = self.generation_indices
       
        max_generation_num_requests = self.num_requests[generation_indices].max()
        if apply_warmup:
            generation_indices = np.where(self.num_requests[generation_indices] == max_generation_num_requests)[0]
        else:
            generation_indices = np.where(self.num_requests[generation_indices] == max_generation_num_requests)[0]

        num_generated_tokens = self.num_generated_tokens[generation_indices]
        target_generation_times = self.time_elapsed[generation_indices]
        return num_generated_tokens / target_generation_times



class FlashLLMEngine:
    def __init__(
            self,
            engine_config: Optional[EngineConfig] = None,
            model_config: Optional[ModelConfig] = None,
            parallel_config: Optional[ParallelConfig] = None,
            **kwargs,
        ):
        # Engine Configs
        engine_config, parallel_config, model_config, draft_model_config, verify_model_config = self._build_configs(engine_config, model_config, parallel_config, **kwargs)
        self.registry = self._build_registry(engine_config, model_config, parallel_config, draft_model_config, verify_model_config)
        
        # Model
        model_name_or_path = self.registry.get('val.model.model_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=self.registry.get('val.engine.hf_model_token'))
        self.model_config = AutoConfig.from_pretrained(model_name_or_path, token=self.registry.get('val.engine.hf_model_token'))
        self.generation_config = GenerationConfig.from_model_config(self.model_config)

        # Request
        self.request_class = self.registry.get('cls.engine.request_class')
        self.request_kwargs = self._build_request_kwargs()

        # Engine Stat
        self.registry.define_schema({
            'val.engine.stat': {
                'type': EngineStat, 'required': False},
        })
        self.engine_stat = EngineStat()
        self.registry.register('val.engine.stat', self.engine_stat)
        
        self.global_waiting_queue = deque()
        self.global_request_counter = 0

        # Options
        self.use_data_parallel_draft = self.registry.get('val.engine.use_data_parallel_draft')
        self.use_ray_worker = self.registry.get('val.engine.use_ray_worker')


        # Worker
        self.is_ray_initialized = False
        self.workers = None
        self.worker_class = self.registry.get('cls.engine.worker_class')
        self.set_workers()
        

                    

        torch.cuda.set_device('cuda:0')
        atexit.register(self.destroy)
        start_nsys_profile()

    
    def _build_configs(
            self,
            model_config: Optional[ModelConfig] = None,
            engine_config: Optional[EngineConfig] = None,
            parallel_config: Optional[ParallelConfig] = None,
            **kwargs,
        ):

        # TODO: think about how to handle kwargs

        print(f"kwargs: {kwargs}")
        # Filter kwargs to only include valid fields for each config

        prefix_tuple = ('draft_', 'verify_')
        
        model_kwargs = {k: v for k, v in kwargs.items() if k in ModelConfig.__annotations__}
        engine_kwargs = {k: v for k, v in kwargs.items() if k in EngineConfig.__annotations__}
        parallel_kwargs = {k: v for k, v in kwargs.items() if k in ParallelConfig.__annotations__}
        
        if 'eagle3_enabled' in engine_kwargs:
            model_kwargs['eagle3_enabled'] = engine_kwargs['eagle3_enabled']

        draft_model_kwargs = model_kwargs.copy()
        draft_model_kwargs['model_name_or_path'] = kwargs['draft_model_name_or_path']
        if 'use_data_parallel_draft' in kwargs:
            draft_model_kwargs['tensor_parallel_size'] = 1

        verify_model_kwargs = model_kwargs.copy()
        verify_model_kwargs['model_name_or_path'] = kwargs['verify_model_name_or_path']

        if 'use_data_parallel_draft' in kwargs:
            verify_model_kwargs['tensor_parallel_size'] = 1
        if engine_config is None:
            engine_config = EngineConfig(**engine_kwargs)
        else:
            engine_config.update(**engine_kwargs)
        
        if DEBUG:
            print(f"engine_config: {engine_config}")

        if parallel_config is None:
            parallel_config = ParallelConfig(**parallel_kwargs)
        else:
            parallel_config.update(**parallel_kwargs)

        if model_config is None:
            model_config = ModelConfig(**model_kwargs)
        else:
            model_config.update(**model_kwargs)

        if len(draft_model_kwargs) > 0 and draft_model_kwargs['model_name_or_path'] is not None:
            draft_model_config = ModelConfig(type='draft', **draft_model_kwargs)
        else:
            draft_model_config = None

        if len(verify_model_kwargs) > 0 and verify_model_kwargs['model_name_or_path'] is not None:
            verify_model_config = ModelConfig(type='verify', **verify_model_kwargs)
        else:
            verify_model_config = None
            
        return engine_config, parallel_config, model_config, draft_model_config, verify_model_config


    def clear_engine_stat(self):
        self.engine_stat.clear()
        if self.use_ray_worker:
            ray.get([worker.clear_scheduling_stats.remote() for worker in self.workers])  
        else:
            for worker in self.workers:
                worker.clear_scheduling_stats()


    def clear_scheduler(self):
        self.global_request_counter = 0
        self.global_waiting_queue = deque()
        if self.use_ray_worker:
            ray.get([worker.clear_scheduler.remote() for worker in self.workers])  
        else:
            for worker in self.workers:
                worker.clear_scheduler()


    def _build_registry(self, *configs):
        if configs is not None:
            configs = [config for config in configs if config is not None]
        engine_registry = EngineRegistry(*configs)
        # engine_registry.print_schema()
        engine_registry.print_nodes()
        return engine_registry
    

    def _build_request_kwargs(self):
        # TODO: save in registry and get from there
        request_kwargs = {}
        if self.registry.get('val.draft_model.model_name_or_path') is not None:
            request_kwargs['num_draft_steps'] = self.registry.get('val.engine.num_draft_steps')

            if self.registry.get('val.verify_model.model_name_or_path') is not None:
                request_kwargs['num_verify_steps'] = self.registry.get('val.engine.num_verify_steps')
                
            request_kwargs['vocab_size'] = self.registry.get('val.engine.vocab_size')

        # Add SmartSpec related configurations
        if self.registry.get('val.engine.smart_spec_enabled'):
            request_kwargs['moving_average_window'] = self.registry.get('val.engine.moving_average_window')

        # Add SVIP related configurations
        if self.registry.get('val.engine.svip_enabled'):
            request_kwargs['threshold_for_svip'] = self.registry.get('val.engine.threshold_for_svip')

        if DEBUG:
            print_debug(
                function_name="FlashLLMEngine._build_request_kwargs",
                request_kwargs=request_kwargs
            )
        return request_kwargs
    


    def set_workers(self):
        if (self.registry.get('val.engine.use_ray_executor') or self.use_ray_worker) and not self.is_ray_initialized:
            import ray
            ray.init(
                resources={f"GPU_{idx}": 1 for idx in self.registry.get('val.model.device_ids')},
                runtime_env={
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": "0",
                        "CUDA_HOME": "/usr/local/cuda"
                    },
                }
            )
            self.is_ray_initialized = True

        if not self.use_ray_worker:
            print(f"\n-------- FlashLLMEngine.set_workers [{self.worker_class}] [single local worker] --------")
            self.num_workers = 1 # parallel_config.world_size
            self.workers = [
                self.worker_class(self.registry, worker_idx) for worker_idx in range(self.num_workers)
            ]
        else:
            import ray
            print(f"\n-------- FlashLLMEngine.set_workers [{self.worker_class}] [{self.registry.get('val.model.world_size')} ray workers] --------")

            runtime_env = {}
            if self.registry.get('val.engine.enable_nsight'):
                print("  Spawning Ray Workers :: Enable nsight")
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
            
            self.workers = []
            print(f"self.registry.get('val.model.device_ids'): {self.registry.get('val.model.device_ids')}")
            for local_rank, device_id in enumerate(self.registry.get('val.model.device_ids')):
                runtime_env.update({
                    "env_vars": {
                        "CUDA_VISIBLE_DEVICES": f"{device_id}",
                        "TORCH_CUDA_ARCH_LIST": os.environ.get('TORCH_CUDA_ARCH_LIST', None),
                        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.registry.get('val.engine.draft_mps_percentage')),
                        "CUDA_MPS_CLIENT_PRIORITY": "0"
                    }
                })
                ray_actor_cls = ray.remote(
                    num_cpus=1,
                    num_gpus=0.1,
                    resources={f"GPU_{device_id}": 0},
                    runtime_env=runtime_env,
                )(self.worker_class)
                
                print(f"Spawning {self.worker_class.__name__} for local_rank {local_rank} with device_id {device_id}")
                ray_actor = ray_actor_cls.remote(
                    self.registry,
                    local_rank=local_rank,
                )
                self.workers.append(ray_actor)

            ray.get([worker.ensure_init_done.remote() for worker in self.workers])

    def reset_workers(self):
        if self.use_ray_worker:
            for worker in self.workers:
                ray.kill(worker)
        self.workers = []


    def execute_multiple_workers(self, func_name:str, *args, run_async: bool = False, **kwargs):
        futures = [getattr(worker, func_name).remote(*args, **kwargs) for worker in self.workers]
        return futures
    
    def execute_single_worker(self, worker, func_name:str, *args, **kwargs):
        if self.use_ray_worker:
            return getattr(worker, func_name).remote(*args, **kwargs)
        else:
            return getattr(worker, func_name)(*args, **kwargs)

    def set_config(self): # TODO: make it out
        self.model_config.set_kv_cache_block_size(self.engine_config.kv_cache_block_size)

    def _tokenize_text(self, request: Request):
        request.tokenize(self.tokenizer)
        return request
    
    def _pre_process_requests(self, requests: Union[Request, List[Request]]):
        if isinstance(requests, Request):
            requests = [requests]
        for request in requests:
            request = self._tokenize_text(request)
            request.after_tokenize_time = time.time()
        return requests

    def add_request(self, requests: Union[Request, List[Request]]):
        if isinstance(requests, Request):
            requests = [requests]
        for request in requests:
            if not request.is_tokenized:
                request = self._pre_process_requests(request)
        # self.scheduler.add_request(requests)
        self.global_waiting_queue.extend(requests)
        self.global_request_counter += len(requests)
    
    def _post_process_request(self, request: Request):
        request.generation_end_time = time.time()
        # self.scheduler.remove_request(request)
        self.global_request_counter -= 1
        request.generated_text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        return request


    def schedule(self):
        data_parallel_size = self.registry.get('val.model.world_size') if self.use_data_parallel_draft else 1
        ideal_local_queue_size = int(self.registry.get('val.engine.num_max_batch_requests') // data_parallel_size)
        num_global_waiting_queue = len(self.global_waiting_queue)

        if self.use_ray_worker:
            if data_parallel_size > 1:
                futures = self.execute_multiple_workers('get_local_scheduler_num_requests')
                local_scheduler_num_requests = ray.get(futures)
            else:
                local_scheduler_num_requests = [ray.get(self.workers[0].get_local_scheduler_num_requests.remote())]
        else:
            local_scheduler_num_requests = [self.workers[0].get_local_scheduler_num_requests()]

        local_scheduler_num_requests = np.array(local_scheduler_num_requests)
        need_to_add_requests = ideal_local_queue_size - local_scheduler_num_requests
        need_to_add_requests = np.maximum(0, need_to_add_requests)

        if DEBUG:
            print_debug(
                function_name="FlashLLMEngine.schedule",
                num_global_waiting_queue=num_global_waiting_queue,
                need_to_add_requests=need_to_add_requests,
                ideal_local_queue_size=ideal_local_queue_size,
                local_scheduler_num_requests=local_scheduler_num_requests,
                data_parallel_size=data_parallel_size,
            )

        if need_to_add_requests.sum() > num_global_waiting_queue:
            base_requests_per_worker = num_global_waiting_queue // len(self.workers)
            remainder = num_global_waiting_queue % len(self.workers)
            need_to_add_requests = np.array([base_requests_per_worker + (1 if i < remainder else 0) for i in range(len(self.workers))])
        
        for worker_idx, num_requests in enumerate(need_to_add_requests):
            if num_requests == 0:
                continue

            requests = []
            for _ in range(num_requests):
                requests.append(self.global_waiting_queue.popleft())

            if len(requests) > 0:
                if self.use_ray_worker:
                    self.execute_single_worker(self.workers[worker_idx], 'add_request', requests)
                else:
                    self.workers[worker_idx].add_request(requests)
                    
        # is_global_prefill_step = True if (np.max(need_to_add_requests) > 0) else False
        # global_prefill_rank = np.argmax(need_to_add_requests > 0)
                    
        # return is_global_prefill_step, global_prefill_rank

    # 워커가 혼자 decoding 돌 때 = None, 워커 안에서 pipe가 혼자 돌 때 0
    def step(self) -> List[Request]:
        self.schedule()
        
        if self.use_ray_worker:
            future = None
            # ray는 함수포인터 관련 오류가 있는 것 같아 (인자가 잘 전달되지 않아서) if 분기로 수정
            if self.use_data_parallel_draft:
                futures = self.execute_multiple_workers('run_data_parallel_draft')
                worker_results = ray.get(futures)
                finished_requests = [request for result in worker_results for request in result[0]]
                stats = [result[1] for result in worker_results]
                # stat 측정 방법 확인 필요
                # import pdb; pdb.set_trace()
                if stats[0] is not None:
                    stat = {
                    'is_prefill_step': stats[0]['is_prefill_step'],
                    'time_elapsed': np.max([result['time_elapsed'] for result in stats]),
                    'num_requests': np.sum([result['num_requests'] for result in stats]),
                    'num_generated_tokens': np.sum([result['num_generated_tokens'] for result in stats]),
                    'num_processed_tokens': np.sum([result['num_processed_tokens'] for result in stats])
                    }
                else:
                    stat = None
            else:
                futures = self.execute_multiple_workers('run')
                finished_requests, stat = ray.get(futures[0])
                del futures
        else:
            finished_requests, stat = self.workers[0].run()
        if stat is not None:
            self.engine_stat.register_step(stat)
        return finished_requests
    

    def _generate_loop(self):
        num_iterations = 0
        finished_requests = []
        len_requests = self.global_request_counter
        with tqdm(total=len_requests, desc=f"Processing Requests", unit=" request") as pbar:
            while self.global_request_counter > 0:
                requests = self.step()
                if DEBUG:
                    self.check_requests_in(requests, "FlashLLMEngine._generate_loop")
                num_iterations += 1
                pbar.set_postfix(
                    iter = num_iterations,
                    prompt_tput = self.engine_stat.get_engine_prefill_throughput(),
                    gen_tput = self.engine_stat.get_engine_generation_throughput()
                )
                if len(requests) != 0:
                    self.engine_stat.register_start_warmup()
                    for idx in range(len(requests)):
                        if requests[idx].is_finished and not requests[idx].is_prefill_step:
                            requests[idx] = self._post_process_request(requests[idx])
                            finished_requests.append(requests[idx])
                        pbar.update(1)

                        if self.global_request_counter < self.registry.get('val.engine.num_max_batch_requests'):
                            self.engine_stat.register_end_warmup()
                            if self.registry.get('val.engine.static_batch_profile'):
                                self.clear_scheduler()
                                return finished_requests
        
        return finished_requests

    def get_worker_stats(self):
        if self.use_ray_worker:
            return ray.get(self.workers[0].get_scheduling_stats.remote())
        else:
            return self.workers[0].get_scheduling_stats()

    def check_requests_in(self, requests, function_name):
        num_requests = len(requests)

        len_global_waiting_queue=len(self.global_waiting_queue)
        len_scheduler_waiting_requests=len(self.scheduler.waiting_requests)
        len_scheduler_scheduled_requests_bucket_0=len(self.scheduler.scheduled_requests.buckets[0])
        len_scheduler_scheduled_requests_bucket_1=len(self.scheduler.scheduled_requests.buckets[1])

        
        in_global_waiting_queue = []
        in_local_waiting_queue = []
        in_bucket_0 = []
        in_bucket_1 = []
        for request in requests:
            in_global_waiting_queue.append(True if request.request_id in [req.request_id for req in self.global_waiting_queue] else False)
            in_local_waiting_queue.append(True if request.request_id in [req.request_id for req in self.scheduler.waiting_requests] else False)
            in_bucket_0.append(True if request.request_id in [req.request_id for req in self.scheduler.scheduled_requests.buckets[0]] else False)
            in_bucket_1.append(True if request.request_id in [req.request_id for req in self.scheduler.scheduled_requests.buckets[1]] else False)

        if DEBUG:
            print_debug(
                function_name=function_name,
                num_requests=num_requests,
                len_global_waiting_queue=len(self.global_waiting_queue),
                len_scheduler_waiting_requests=len(self.scheduler.waiting_requests),
                len_scheduler_scheduled_requests_bucket_0=len(self.scheduler.scheduled_requests.buckets[0]),
                len_scheduler_scheduled_requests_bucket_1=len(self.scheduler.scheduled_requests.buckets[1]),
                len_pipelining_future_queue=len(self.pipelining_future_queue),
            )
        

        if sum(in_global_waiting_queue) + sum(in_local_waiting_queue) + sum(in_bucket_0) + sum(in_bucket_1) != num_requests:
            print("Where the requests came from?")
            assert False, "Where the requests came from?"
            # import pdb; pdb.set_trace()


    def generate(
            self,
            prompt_texts: Optional[Union[str, List[str]]] = None,
            prompt_ids: Optional[Union[List[List[int]], List[int]]] = None,
            sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        ):

        num_requests = len(prompt_texts) if prompt_texts is not None else len(prompt_ids)
        if num_requests == 0:
            raise ValueError("No requests to generate")
        
        if prompt_texts is not None:
            if isinstance(prompt_texts, str):
                prompt_texts = [prompt_texts]
        else:
            prompt_texts = [None] * num_requests
        
        if prompt_ids is not None:
            if isinstance(prompt_ids, list) and isinstance(prompt_ids[0], list):
                prompt_ids = [prompt_ids]
        else:
            prompt_ids = [None] * num_requests

        if sampling_params is not None:
            if isinstance(sampling_params, SamplingParams):
                sampling_params = [sampling_params]
            else:
                raise ValueError("sampling_params must be a SamplingParams or a list of SamplingParams")

        for sampling_param in sampling_params:
            sampling_param.eos_token_ids = np.array(self.tokenizer.eos_token_id)

        if num_requests != 1 and len(sampling_params) == 1:
            sampling_params = [copy.deepcopy(sampling_params[0]) for _ in range(len(prompt_texts))]

        
        requests = [self.request_class(prompt_text=prompt_text, prompt_ids=prompt_ids, sampling_params=sampling_params, **self.request_kwargs) for prompt_text, prompt_ids, sampling_params in zip(prompt_texts, prompt_ids, sampling_params)]
        self.add_request(requests)

        requests = self._generate_loop()
        self.engine_stat.register_scheduling_stats(self.get_worker_stats())
        
        request_outputs = [request.convert_to_request_output() for request in requests]
        return request_outputs
    

    def destroy(self):
        if self.is_ray_initialized:
            ray.shutdown()


    def categorize_requests(self, requests: List[Request]):
        prefill_requests = [request for request in requests if request.is_prefill_step]
        generation_requests = [request for request in requests if not request.is_prefill_step]
        return prefill_requests, generation_requests
    

    @profile_nvtx("FlashLLMEngine.update_requests_with_processed_results")
    def update_requests_with_processed_results(
        self, 
        running_requests: List[Request], 
        processed_requests: List[List[Request]],
    ):
        requests = []
        for request_list in processed_requests:
            if request_list:
                requests.extend(request_list)
        processed_requests = requests

        for processed_request in processed_requests:
            for running_request in running_requests:
                if running_request.request_id == processed_request.request_id:
                    for field in fields(processed_request):
                        setattr(running_request, field.name, getattr(processed_request, field.name))


    # def __del__(self):
    #     # import sys
    #     # if sys.meta_path is None:
    #     #     return
    #     self.destroy()
