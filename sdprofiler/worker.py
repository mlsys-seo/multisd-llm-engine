import time
from typing import List, Optional, Union
from datetime import timedelta

import torch
import numpy as np
import ray
import ray.util.collective as ray_collective


from .scheduler import Scheduler
from .registry import EngineRegistry
from .model_io_processor import ModelIOProcessor
from .request import Request, FlashInferMetadata, SmartSpecRequest, SVIPRequest, Eagle3Request, Eagle3SVRequest
from .model_runner import ModelRunner
from .utils.common import profile_nvtx, print_debug
from .utils.parallel_group_utils import rebuild_tensor_from_ipc_info

# import pygloo.rendezvous as _pygloo

# # 원래 Context.setTimeout을 저장
# _orig_setTimeout = _pygloo.Context.setTimeout

# def _patched_setTimeout(self, timeout):
#     # int로 넘어오면 datetime.timedelta로 감싸기
#     if isinstance(timeout, int):
#         timeout = datetime.timedelta(milliseconds=timeout)
#     # 이제 원본 호출
#     return _orig_setTimeout(self, timeout)

# # monkey-patch 적용
# _pygloo.Context.setTimeout = _patched_setTimeout


DEBUG = False

class Worker:
    def __init__(
            self, 
            registry: EngineRegistry,
            local_rank: int
        ):
        self.registry = registry
        
        device_ids = self.registry.get('val.model.device_ids')
        self.num_workers = len(device_ids)
        self.local_rank = local_rank
        
        self.use_ray_worker = self.registry.get('val.engine.use_ray_worker')
        self.use_ray_executor = self.registry.get('val.engine.use_ray_executor')
        self.use_sparse_probs = self.registry.get('val.engine.use_sparse_probs')
        self.use_data_parallel_draft = self.registry.get('val.engine.use_data_parallel_draft')
        self.is_driver_worker = bool(self.local_rank == 0)

        self.device = f'cuda:0'
        torch.cuda.set_device(self.device)
        self.engine_stat = self.registry.get('val.engine.stat')

        self._setup_worker_components()

        if DEBUG:
            print_debug(
                function_name="Worker.__init__",
                torch_distributed_is_initialized=torch.distributed.is_initialized(),
                self_num_workers=self.num_workers,
                self_local_rank=self.local_rank,
                self_use_data_parallel_draft=self.use_data_parallel_draft,
                self_is_driver_worker=self.is_driver_worker
            )
        
        # if self.num_workers > 1 and not self.is_driver_worker and not torch.distributed.is_initialized():


        # if DEBUG:
        print_debug(
            function_name="Worker.__init__",
            use_data_parallel_draft=self.use_data_parallel_draft,
            use_ray_worker=self.use_ray_worker,
            is_driver_worker=self.is_driver_worker
        )
        
        if (not self.use_data_parallel_draft) and (not self.is_driver_worker):
            self.scheduler = None
        else:
            self.scheduler = Scheduler(self.registry)



        self.scheduling_stats = []

        if DEBUG:
            print_debug(
                function_name="Worker.__init__",
                device=self.device,
                local_rank=self.local_rank,
                use_ray_worker=self.use_ray_worker,
                use_ray_executor=self.use_ray_executor,
                use_data_parallel_draft=self.use_data_parallel_draft,
                is_driver_worker=self.is_driver_worker,
                device_ids=device_ids
            )
    
    def _setup_worker_components(self):
        self.model_runner = ModelRunner(
            self.registry, 
            local_rank=self.local_rank
        )
        self.model_io_processor = ModelIOProcessor(self.registry)
        
        if self.use_ray_executor:
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.model_runner.model_executors]
            ray.get(ray_executors)

    def ensure_init_done(self):
        time.sleep(1)
        return True

    def update_registry(self, registry: EngineRegistry):
        self.registry = registry
        self.model_io_processor.update_registry(registry)
    
    '''Methods for the local scheduler'''
    def add_request(self, requests: Union[Request, List[Request]]):
        self.scheduler.add_request(requests)

    def remove_request(self, requests: Union[Request, List[Request]]):
        if isinstance(requests, Request):
            requests = [requests]
        for request in requests:
            self.scheduler.remove_request(request)

    def get_scheduling_stats(self):
        return self.scheduling_stats

    def clear_scheduling_stats(self):
        if hasattr(self, 'scheduling_stats'):
            self.scheduling_stats = []

    def clear_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.clear()


    def get_local_scheduler_num_requests(self):
        return self.scheduler.num_requests
    
    def local_schedule(self):
        return self.scheduler.schedule()
    
    '''Methods for the IN/OUT'''
    def _check_requests_finished(
        self,
        requests: List[Request]
    ):
        finished_requests = []
        for request in requests:
            if self.model_io_processor.check_finish_criteria(request):
                request.record_finished()
                self.scheduler.remove_request(request)
                finished_requests.append(request)
        return finished_requests


    @profile_nvtx("Worker._prepare_input_tensors")
    def _prepare_input_tensors(
        self,
        requests: Optional[List[Request]] = None
    ):        
        if self.use_ray_worker and not self.is_driver_worker:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = None, None, None
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = self.model_io_processor.build_input_tensors(requests)

            if DEBUG:
                print_debug(
                    function_name="Worker._prepare_input_tensors",
                    input_ids=input_ids_tensor.shape,
                    position_ids=position_ids_tensor.shape,
                    flashinfer_metadata=flashinfer_metadata
                )

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

    def run_empty(self, target_model_runner: ModelRunner):
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
            self.registry.get("val.engine.kv_cache_block_size"), 
        )
        
        runner_outputs = target_model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        del runner_outputs
        return None


    @profile_nvtx("Worker.run")
    def _run(
            self,
            requests: Optional[List[Request]] = None,
        ):

        if requests is None:
            self.run_empty(self.model_runner)
            return None

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_input_tensors(requests)
        
        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests] if requests else None
        )

        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, log_probs_per_request, probs_input_ids_indptr = runner_outputs
        
        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, log_probs_per_request, probs_input_ids_indptr)
        return requests
    
    def get_global_state_data_parallel_draft(self, requests: Optional[List[Request]] = None):
        if requests is None:
            requests = [] 
        is_local_prefill_step = requests[0].is_prefill_step if len(requests) > 0 else False
        len_requests = len(requests) if requests is not None else 0
        global_prefill_steps = [None] * self.num_workers


        # ray_collective.collective.allgather(objs, is_local_prefill_step)
        # is_global_prefill_step = any(objs)

        torch.distributed.all_gather_object(global_prefill_steps, (is_local_prefill_step, len_requests))
        is_global_prefill_step = any([item[0] for item in global_prefill_steps])
        num_all_requests = sum([item[1] for item in global_prefill_steps])

        requests = [] if is_global_prefill_step is True and is_local_prefill_step is False else requests

        return requests, is_local_prefill_step, is_global_prefill_step, num_all_requests


    def run_data_parallel_draft(self):
        requests = self.local_schedule()
        start_time = time.time()
        
        requests, is_local_prefill_step, is_global_prefill_step, num_all_requests = self.get_global_state_data_parallel_draft(requests)
        num_requests = len(requests) if requests is not None else 0
        if DEBUG:
            print_debug(
                function_name=f"Worker.run_data_parallel_draft_{self.local_rank}",
                num_requests=num_requests,
                is_local_prefill_step=requests[0].is_prefill_step if len(requests) > 0 else None,
                is_global_prefill_step=is_global_prefill_step,
            )

        requests = self._run_data_parallel_draft(requests, is_global_prefill_step=is_global_prefill_step)

        if requests is not None:
            finished_requests = self._check_requests_finished(requests)
        else:
            finished_requests = []
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        num_generated_tokens = np.sum([request.context_state.last_num_generated_tokens for request in requests]) if requests is not None else 0
        if is_global_prefill_step:
            num_processed_tokens = np.sum([request.context_state.scheduled_query_len for request in requests]) if requests is not None else 0
        else:
            num_processed_tokens = 0
        
        stat = {
            'is_prefill_step': is_global_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': num_generated_tokens,
            'num_processed_tokens': num_processed_tokens
        }

        if DEBUG:
            print_debug(
                function_name="Worker.run",
                is_prefill_step=is_global_prefill_step,
                num_requests=num_requests,
                time_elapsed=time_elapsed,
                num_generated_tokens=num_generated_tokens,
                num_processed_tokens=num_processed_tokens
            )
        return finished_requests, stat


    def run(self):
        if self.local_rank == 0:
            requests = self.local_schedule()
            assert len(requests) > 0, "No requests to run"
            is_prefill_step = requests[0].is_prefill_step
            num_requests = len(requests)
        else:
            requests = None
            is_prefill_step = None

        start_time = time.time()
        requests = self._run(requests)
    
        if self.local_rank > 0:
            return None, None

        finished_requests = self._check_requests_finished(requests)
        end_time = time.time()
        time_elapsed = end_time - start_time
        stat = {
            'is_prefill_step': is_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': np.sum([request.context_state.last_num_generated_tokens for request in requests]),
            'num_processed_tokens': np.sum([request.context_state.scheduled_query_len for request in requests]) if is_prefill_step else 0
        }

        # # if DEBUG:
        # print_debug(
        #     function_name="Worker.run",
        #     is_prefill_step=stat['is_prefill_step'],
        #     num_requests=stat['num_requests'],
        #     time_elapsed=stat['time_elapsed'],
        #     num_generated_tokens=stat['num_generated_tokens'],
        #     num_processed_tokens=stat['num_processed_tokens']
        # )
        # input()
        
        return finished_requests, stat


    @profile_nvtx("ModelRunner.get_distributed_model_results")
    def get_ray_model_results(self, futures):
        worker_results = ray.get(futures[0])
        del futures

        results = worker_results

        sampled_tokens_per_request, probs, probs_input_ids_indptr = results
        probs = rebuild_tensor_from_ipc_info(probs)
        return sampled_tokens_per_request, probs, probs_input_ids_indptr




from .request import SpeculativeRequest
from .model_io_processor import SpeculativeModelIOProcessor

class SpeculativeWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_draft_steps = self.registry.get('val.engine.num_draft_steps')
        
        if self.use_data_parallel_draft:
            self._run = self._run_data_parallel_draft

    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, local_rank=self.local_rank)
        self.model_runner = self.verify_model_runner

        self.model_io_processor = SpeculativeModelIOProcessor(self.registry)

        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.verify_model_runner.model_executors]
            ray.get(ray_executors)


    @profile_nvtx("Worker._prepare_draft_input_tensors")
    def _prepare_draft_input_tensors(
        self,
        requests: Optional[List[SpeculativeRequest]] = None
    ):
        if requests is not None:
            # driver worker
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = self.model_io_processor.build_draft_input_tensors(requests)
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = None, None, None

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata


    @profile_nvtx("SpeculativeWorker.run_draft")
    def run_draft(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_draft_input_tensors(requests)


        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=1
        )

        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        

        # nonzero_indices = probs_per_request.nonzero()
        # only driver worker
        requests = self.model_io_processor.append_draft_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests


    @profile_nvtx("Worker._prepare_verify_input_tensors")
    def _prepare_verify_input_tensors(
        self,
        requests: Optional[List[Request]] = None
    ):
        if requests is not None:
            # driver worker
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = self.model_io_processor.build_verify_input_tensors(requests)
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = None, None, None

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata


    @profile_nvtx("Worker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps + 1
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        
        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests
    
    @profile_nvtx("Worker.run_speculation")
    def run_speculation(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        self.model_io_processor.setup_speculative_step(requests)
        for i in range(self.num_draft_steps):
            requests = self.run_draft(self.draft_model_runner, requests)
        requests = self.run_verify(self.verify_model_runner, requests)

        rejection_sampler_outputs = self.model_io_processor.run_rejection_sampler(requests)
        self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs
        
    
    @profile_nvtx("Worker.run_prefill")
    def run_prefill(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = self.draft_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests]
        )
        
        del runner_outputs

        runner_outputs = self.verify_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests]
        )

        if self.verify_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs

        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        
        for request in requests:
            request.draft_context_state.context_len = request.verify_context_state.context_len
            
        return requests
    
    
    def run_prefill_empty(
            self,
            run_async: bool = False
        ):
        
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(self.registry.get("val.engine.kv_cache_block_size"))
        
        runner_outputs = self.verify_model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None
        )
        
        if self.verify_model_runner.is_ray_runner and not run_async:
            runner_outputs = self.get_ray_model_results(runner_outputs)
            
        del runner_outputs
    

    @profile_nvtx("SpeculativeWorker._run_data_parallel_draft")
    def _run_data_parallel_draft(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
            is_global_prefill_step: bool = False,
        ):
        if is_global_prefill_step:
            if len(requests) != 0 and requests[0].is_prefill_step:
                requests = self.run_prefill(requests)
                return requests
            self.run_prefill_empty()
            return None
        else:
            # 다른 worker가 decoding 끝낼 때까지 더미 데이터로 sync
            if len(requests) == 0:
                self.run_prefill_empty()
                return None
            else:
                requests, rejection_sampler_outputs = self.run_speculation(requests)
                requests = self.model_io_processor.speculative_post_process_requests(requests, rejection_sampler_outputs)
                return requests


    @profile_nvtx("SpeculativeWorker._run")
    def _run(
            self,
            requests: Optional[List[SpeculativeRequest]] = None
        ):
        if self.local_rank != 0:
            futures = self.verify_model_runner.run(
                None, 
                None, 
                None,
                None
            )
            del futures
            return None
        
        is_prefill_step = requests[0].is_prefill_step
        if is_prefill_step:
            requests = self.run_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_speculation(requests)
            requests = self.model_io_processor.speculative_post_process_requests(requests, rejection_sampler_outputs)
            
        return requests
    


from .request import HierarchicalSpeculativeRequest
from .model_io_processor import HierarchicalSpeculativeModelIOProcessor
class HierarchicalSpeculativeWorker(SpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_verify_steps = self.registry.get('val.engine.num_verify_steps')
        self.use_only_beta_cutoff = self.registry.get('val.engine.use_only_beta_cutoff')    
        self.profile_acceptance_prob = self.registry.get('val.engine.run_profile_acceptance_prob')
        if self.use_data_parallel_draft:
            self._profile_latency = self._profile_latency_data_parallel_draft

    
    
    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, type='verify', local_rank=self.local_rank)
        self.oracle_model_runner = ModelRunner(registry=self.registry, type=None, local_rank=self.local_rank)
        self.model_runner = self.oracle_model_runner

        self.model_io_processor = HierarchicalSpeculativeModelIOProcessor(self.registry, self.local_rank)
        
        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.oracle_model_runner.model_executors]
            ray.get(ray_executors)

    def setup_scheduling_buffers(self):
        self.model_io_processor.setup_scheduling_buffers()

    @profile_nvtx("Worker._prepare_oracle_input_tensors")
    def _prepare_oracle_input_tensors(
        self,
        requests: Optional[List[SpeculativeRequest]] = None
    ):
        if requests is not None:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = self.model_io_processor.build_oracle_input_tensors(requests)
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata = None, None, None
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    

    @profile_nvtx("Worker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps
        )

        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs

        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests

    @profile_nvtx("Worker.run_oracle")
    def run_oracle(
        self,
        requests: Optional[List[HierarchicalSpeculativeRequest]] = None
    ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)

        runner_outputs = self.oracle_model_runner.run(
            input_ids_tensor,
            position_ids_tensor,
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps + 1
        )

        if self.oracle_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        
        requests = self.model_io_processor.append_oracle_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        return requests


    @profile_nvtx("Worker.run_hierarchical_speculation")
    def run_hierarchical_speculation(self, requests: List[HierarchicalSpeculativeRequest], num_tokens_to_execute: Optional[int] = None):
        self.model_io_processor.setup_oracle_step(requests)
        self.model_io_processor.setup_speculative_step(requests)
        for i in range(self.num_draft_steps):
            requests = self.run_draft(self.draft_model_runner, requests)
        
        requests = self.run_verify(self.verify_model_runner, requests)
        
        if self.use_only_beta_cutoff:
            speculative_verification_outputs = self.model_io_processor.verify_info_gain_with_rejection_sampling(requests)
        else:
            speculative_verification_outputs, scheduling_stat = self.model_io_processor.verify_info_gain(requests, num_tokens_to_execute)
            self.scheduling_stats.append(scheduling_stat)
        requests, _ = self.model_io_processor.mid_process(requests, speculative_verification_outputs)

        requests = self.run_oracle(requests)

        requests, rejection_sampler_outputs = self.model_io_processor.run_oracle_rejection_sampler(
            requests, return_beta=True, return_accept_prob=self.profile_acceptance_prob)

        if self.use_sparse_probs:
            self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs


    @profile_nvtx("HierarchicalSpeculativeWorker.run_prefill")
    def run_prefill(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)

        runner_outputs = self.draft_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=0
        )
        del runner_outputs

        runner_outputs = self.verify_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=0
        )

        del runner_outputs

        
        runner_outputs = self.oracle_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=None
        )


        if self.oracle_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        
        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        
        for request in requests:
            request.draft_context_state.context_len = request.oracle_context_state.context_len
            request.verify_context_state.context_len = request.oracle_context_state.context_len
        
        return requests
    
    def run_prefill_empty(self):
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
            self.registry.get("val.engine.kv_cache_block_size"), 
        )
        runner_outputs = self.oracle_model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None,
            num_logits_to_keep=None
        )
        del runner_outputs
        
    def run_decoding_empty(self):
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
            self.registry.get("val.engine.kv_cache_block_size"), 
        )
        
        # 다른 worker가 decoding 할 때, Token Scheduling의 all_gather 싱크를 맞추기 위해 None 교환
        if self.registry.get('val.engine.use_data_parallel_draft'):
            allgather_results = [None for _ in range(self.num_workers)]
            torch.distributed.all_gather_object(allgather_results, None)
            # torch.distributed.barrier()

        # oracle model sync
        runner_outputs = self.oracle_model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None,
            num_logits_to_keep=None
        )
        del runner_outputs
    

    @profile_nvtx("HierarchicalSpeculativeWorker._run_data_parallel_draft")
    def _run_data_parallel_draft(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
            is_global_prefill_step: Optional[bool] = None,
        ):
        if is_global_prefill_step:
            if len(requests) != 0 and requests[0].is_prefill_step:
                requests = self.run_prefill(requests)
                return requests
            self.run_prefill_empty()
            return None
        else:
            # 다른 worker가 decoding 끝낼 때까지 더미 데이터로 sync
            if len(requests) == 0:
                self.run_decoding_empty()
                return None
            else:
                requests, rejection_sampler_outputs = self.run_hierarchical_speculation(requests)
                requests = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)
                return requests

    @profile_nvtx("HierarchicalSpeculativeWorker._run")
    def _run(
        self, 
        requests: Optional[List[HierarchicalSpeculativeRequest]] = None,
    ):
        if self.local_rank != 0:
            futures = self.oracle_model_runner.run(
                None, 
                None, 
                None,
                None
            )
            del futures
            return None

        is_prefill_step = requests[0].is_prefill_step
        if is_prefill_step:
            requests = self.run_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_hierarchical_speculation(requests)
            requests = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)

        return requests
    
    def _profile_latency_data_parallel_draft(
        self,
        batch_size: int,
        prompt_len: int,
        num_compute_tokens: int
    ):
        batch_size = batch_size // self.registry.get(f'val.engine.global_world_size')
        request_cls = self.registry.get('cls.engine.request_class')
        block_size = self.registry.get('val.engine.kv_cache_block_size')
        vocab_size = self.registry.get('val.engine.vocab_size')
        
        # warmup
        for i in range(3):
            dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
            requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
            _ = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)

        latencies = []
        for i in range(15):
            dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
            start_time = time.time()
            requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
            _ = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)
            end_time = time.time()

            latencies.append((end_time - start_time))
        
        if self.local_rank == 0:
            latencies = np.sort(latencies)[2:-2]
            latency = np.mean(latencies)
            return latency
        else:
            return None
    

    def _profile_latency(
            self,
            batch_size: int,
            prompt_len: int,
            num_compute_tokens: int
        ):
        request_cls = self.registry.get('cls.engine.request_class')
        block_size = self.registry.get('val.engine.kv_cache_block_size')
        vocab_size = self.registry.get('val.engine.vocab_size')

        # warmup
        for i in range(3):
            if self.local_rank != 0:
                futures = self.oracle_model_runner.run(
                    None, 
                    None, 
                    None,
                    None
                )
                del futures
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)

        latencies = []
        for i in range(12):
            if self.local_rank != 0:
                futures = self.oracle_model_runner.run(
                    None, 
                    None, 
                    None,
                    None
                )
                del futures
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                start_time = time.time()
                requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)
                end_time = time.time()

                latencies.append((end_time - start_time))
        
        if self.local_rank == 0:
            latencies = np.sort(latencies)[2:-2]
            latency = np.mean(latencies)
            return latency
        else:
            return None
    
    
def create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size, is_prefill_step: bool = False):
    num_required_kv_blocks = int(np.ceil(prompt_len / block_size))
    paged_kv_block_indices_list = [np.arange(start, start + num_required_kv_blocks) for start in range(0, num_required_kv_blocks*batch_size, num_required_kv_blocks)]
    return [
        request_cls.create_dummy(prompt_len, vocab_size, paged_kv_block_indices, is_prefill_step)
        for paged_kv_block_indices in paged_kv_block_indices_list
    ]

from .request import LayerSkipRequest
from .model_io_processor import LayerSkipIOProcessor
class LayerSkipWorker(SpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.draft_last_layer_idx = self.registry.get('val.engine.draft_last_layer_idx')
        self.verify_last_layer_idx = self.registry.get('val.engine.verify_last_layer_idx')
        # TODO: add config
        
        if self.draft_last_layer_idx == self.verify_last_layer_idx:
            self.speculation_func = self.run_speculation
        else:
            raise ValueError(
                f"draft_last_layer_idx ({self.draft_last_layer_idx}) != verify_last_layer_idx ({self.verify_last_layer_idx}), "
                "hierarchical speculation is not supported in this configuration."
                )

    def _setup_worker_components(self):
        self.model_runner = ModelRunner(registry=self.registry, type=None, local_rank=self.local_rank)
        self.model_io_processor = LayerSkipIOProcessor(self.registry)
        
        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.model_runner.model_executors]
            ray.get(ray_executors)

    @profile_nvtx("LayerSkipWorker.run_speculation")
    def run_speculation(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        self.draft_step_count = 0
        self.model_io_processor.setup_speculative_step(requests)
        for i in range(self.num_draft_steps+1):
            requests = self.run_draft(self.model_runner, requests)
        # requests = self.run_draft(self.model_runner, requests)
        requests = self.run_verify(self.model_runner, requests)

        rejection_sampler_outputs = self.model_io_processor.run_rejection_sampler(requests)
        self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs
    
    @profile_nvtx("LayerSkipWorker.run_draft")
    def run_draft(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[LayerSkipRequest]] = None,
        ):
        
        ### 이거 타이밍 언제?
        self.draft_step_count += 1
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_draft_input_tensors(requests)
        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=1,
            start_layer_idx=0,
            end_layer_idx=self.draft_last_layer_idx
        )
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        
        if self.draft_step_count <= self.num_draft_steps:
            requests = self.model_io_processor.append_draft_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        return requests

    @profile_nvtx("LayerSkipWorker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[LayerSkipRequest]] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps+1,
            start_layer_idx=self.draft_last_layer_idx+1,
            end_layer_idx=None
        )
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests


    @profile_nvtx("LayerSkipWorker.run_prefill")
    def run_prefill(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)

        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=None,
            start_layer_idx=0,
            end_layer_idx=None
        )
        
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs
        
        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        
        for request in requests:
            request.draft_context_state.context_len = request.verify_context_state.context_len
        
        return requests
    
    def run(self):
        if self.local_rank == 0:
            requests = self.local_schedule()
            assert len(requests) > 0, "No requests to run"
            is_prefill_step = requests[0].is_prefill_step
            num_requests = len(requests)
            
        else:
            requests = None
            is_prefill_step = None
            
        if torch.distributed.is_initialized():
            obj_list = [is_prefill_step]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            is_prefill_step = obj_list[0]
        start_time = time.time()
        requests = self._run(requests, is_prefill_step)
    
        if self.local_rank > 0:
            return None, None

        finished_requests = self._check_requests_finished(requests)
        end_time = time.time()
        time_elapsed = end_time - start_time
        stat = {
            'is_prefill_step': is_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': np.sum([request.context_state.last_num_generated_tokens for request in requests]),
            'num_processed_tokens': np.sum([request.context_state.scheduled_query_len for request in requests]) if is_prefill_step else 0
        }
        if DEBUG:
            print_debug(
                function_name="Worker.run",
                is_prefill_step=stat['is_prefill_step'],
                num_requests=stat['num_requests'],
                time_elapsed=stat['time_elapsed'],
                num_generated_tokens=stat['num_generated_tokens'],
                num_processed_tokens=stat['num_processed_tokens']
            )
        return finished_requests, stat
    
    def dummy_work(self):
        for i in range(self.num_draft_steps+1):
                futures = self.model_runner.run(
                    None, 
                    None, 
                    None,
                    None,
                    start_layer_idx=0,
                    end_layer_idx=self.draft_last_layer_idx,
                ) 
                del futures
                
        futures = self.model_runner.run(
            None,
            None,
            None,
            None,
            start_layer_idx=self.verify_last_layer_idx + 1,
            end_layer_idx=None
        )
        del futures
        torch.cuda.empty_cache()
        return None
    
    @profile_nvtx("LayerSkipWorker._run")
    def _run(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
            is_prefill_step: Optional[bool] = None,
        ):
        if self.local_rank != 0:
            if is_prefill_step:
                self.dummy_prefill()
                return None
            else:
                self.dummy_work()
            return None
        
        is_prefill_step = requests[0].is_prefill_step
        if is_prefill_step:
            requests = self.run_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_speculation(requests)
            requests = self.model_io_processor.layerskip_post_process_requests(requests, rejection_sampler_outputs)
            
        return requests
    
    def dummy_prefill(self):
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
            self.registry.get("val.engine.kv_cache_block_size"), 
        )
        runner_outputs = self.model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None,
            num_logits_to_keep=None,
            start_layer_idx=0,
            end_layer_idx=None
        )
        del runner_outputs
        torch.cuda.empty_cache()
        return None
    
from .request import HierarchicalLayerSkipRequest
from .model_io_processor import HiarchicalLayerSkipIOProcessor
class HierarchicalLayerSkipWorker(HierarchicalSpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.draft_last_layer_idx = self.registry.get('val.engine.draft_last_layer_idx')
        self.verify_last_layer_idx = self.registry.get('val.engine.verify_last_layer_idx')

        if self.draft_last_layer_idx != self.verify_last_layer_idx:
            self.speculation_func = self.run_hierarchical_speculation
        else:
            raise ValueError(
                f"draft_last_layer_idx ({self.draft_last_layer_idx}) != verify_last_layer_idx ({self.verify_last_layer_idx}), "
                "hierarchical speculation is not supported in this configuration."
                )
            
    def _setup_worker_components(self):
        self.model_runner = ModelRunner(registry=self.registry, type=None, local_rank=self.local_rank)
        self.model_io_processor = HiarchicalLayerSkipIOProcessor(self.registry, self.local_rank)
        self.oracle_model_runner = self.model_runner
        
        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.model_runner.model_executors]
            ray.get(ray_executors)
    
    
    def run(self):
        
        if self.local_rank == 0:
            requests = self.local_schedule()
            assert len(requests) > 0, "No requests to run"
            is_prefill_step = requests[0].is_prefill_step
            num_requests = len(requests)
            
        else:
            requests = None
            is_prefill_step = None
            
        if torch.distributed.is_initialized():
            obj_list = [is_prefill_step]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            is_prefill_step = obj_list[0]
        start_time = time.time()
        requests = self._run(requests, is_prefill_step)
    
        if self.local_rank > 0:
            return None, None

        finished_requests = self._check_requests_finished(requests)
        end_time = time.time()
        time_elapsed = end_time - start_time
        stat = {
            'is_prefill_step': is_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': np.sum([request.context_state.last_num_generated_tokens for request in requests]),
            'num_processed_tokens': np.sum([request.context_state.scheduled_query_len for request in requests]) if is_prefill_step else 0
        }
        
        if DEBUG:
            print_debug(
                function_name="Worker.run",
                is_prefill_step=stat['is_prefill_step'],
                num_requests=stat['num_requests'],
                time_elapsed=stat['time_elapsed'],
                num_generated_tokens=stat['num_generated_tokens'],
                num_processed_tokens=stat['num_processed_tokens']
            )
        return finished_requests, stat
    
    def dummy_work(self):
        
                # if is_prefill:
                #     futures = self.model_runner.run(
                #         None,
                #         None,
                #         None,
                #         None,
                #         num_logits_to_keep=None,
                #         start_layer_idx=0,
                #         end_layer_idx=None
                #     )
                #     del futures
        # import pdb; pdb.set_trace() 
        for i in range(self.num_draft_steps+1):
            if i < self.num_draft_steps:
                futures = self.model_runner.run(
                    None, 
                    None, 
                    None,
                    None,
                    start_layer_idx=0,
                    end_layer_idx=self.draft_last_layer_idx,
                ) 
                del futures
            else:
                hierachical = True  
                futures = self.model_runner.run(
                        None, 
                        None, 
                        None,
                        None,
                        start_layer_idx=0,
                        end_layer_idx=self.draft_last_layer_idx,
                        hierachical=hierachical
                    ) 
                del futures
        futures = self.model_runner.run(
            None,
            None,
            None,
            None,
            start_layer_idx=self.draft_last_layer_idx+1,
            end_layer_idx=self.verify_last_layer_idx
        )
        del futures
        
        futures = self.model_runner.run(
            None,
            None,
            None,
            None,
            start_layer_idx=self.verify_last_layer_idx + 1,
            end_layer_idx=None
        )
        del futures
        torch.cuda.empty_cache()
        return None
    
    def dummy_prefill(self):
        flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
            self.registry.get("val.engine.kv_cache_block_size"), 
        )
        runner_outputs = self.model_runner.run(
            torch.tensor([], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            flashinfer_metadata_dummy,
            sampling_params=None,
            num_logits_to_keep=None,
            start_layer_idx=0,
            end_layer_idx=None
        )
        del runner_outputs
        torch.cuda.empty_cache()
        return None
        
    @profile_nvtx("HierarchicalLayerSkipWorker.run_hierarchical_speculation")
    def run_hierarchical_speculation(self, requests: List[HierarchicalSpeculativeRequest], num_tokens_to_execute: Optional[int] = None):
        
        self.draft_step_count = 0
        self.model_io_processor.setup_oracle_step(requests)
        self.model_io_processor.setup_speculative_step(requests)

        for i in range(self.num_draft_steps+1):
            requests = self.run_draft(self.model_runner, requests)
        requests = self.run_verify(self.model_runner, requests)
        
        speculative_verification_outputs = self.model_io_processor.verify_info_gain_with_rejection_sampling(requests)
        
        requests, _ = self.model_io_processor.mid_process(requests, speculative_verification_outputs)
        
        requests = self.run_oracle(requests)
        
        requests, rejection_sampler_outputs = self.model_io_processor.run_oracle_rejection_sampler(
            requests, return_beta=True, return_accept_prob=self.profile_acceptance_prob)
        
        if self.use_sparse_probs:
            self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs
       
    @profile_nvtx("LayerSkiHierarchicalLayerSkipWorkerpWorker.run_draft")
    def run_draft(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[HierarchicalLayerSkipRequest]] = None,
        ):
        self.draft_step_count += 1
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_draft_input_tensors(requests)
        
        if self.draft_step_count <= self.num_draft_steps:
            runner_outputs = self.model_runner.run(
                input_ids_tensor, 
                position_ids_tensor, 
                flashinfer_metadata,
                sampling_params=[request.sampling_params for request in requests],
                num_logits_to_keep=1,
                start_layer_idx=0,
                end_layer_idx=self.draft_last_layer_idx,
            )    
            if self.model_runner.is_ray_runner:
                runner_outputs = self.get_ray_model_results(runner_outputs)
            sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs    
            requests = self.model_io_processor.append_draft_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        else:
            hierachical = True
            runner_outputs = self.model_runner.run(
                input_ids_tensor, 
                position_ids_tensor, 
                flashinfer_metadata,
                sampling_params=[request.sampling_params for request in requests],
                num_logits_to_keep=0,
                start_layer_idx=0,
                end_layer_idx=self.draft_last_layer_idx,
                hierachical=hierachical
            ) 
            # import pdb; pdb.set_trace()
            for req in requests:
                req.draft_context_state.context_len += 1
            del runner_outputs
            
        return requests

    @profile_nvtx("HierarchicalLayerSkipWorker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[HierarchicalLayerSkipRequest]] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps,
            start_layer_idx=self.draft_last_layer_idx+1,
            end_layer_idx=self.verify_last_layer_idx
        )
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs

        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests

    @profile_nvtx("HierarchicalLayerSkipWorker.run_oracle")
    def run_oracle(
        self,
        requests: Optional[List[HierarchicalLayerSkipRequest]] = None
    ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)

        runner_outputs = self.model_runner.run(
            input_ids_tensor,
            position_ids_tensor,
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps + 1,
            start_layer_idx=self.verify_last_layer_idx + 1,
            end_layer_idx=None
        )
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs

        requests = self.model_io_processor.append_oracle_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        return requests


    @profile_nvtx("HierarchicalLayerSkipWorker.run_prefill")
    def run_prefill(
            self,
            requests: Optional[List[HierarchicalLayerSkipRequest]] = None,
        ):

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)

        runner_outputs = self.model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=None,
            start_layer_idx=0,
            end_layer_idx=None
        )
        if self.model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr = runner_outputs

        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        
        for request in requests:
            request.draft_context_state.context_len = request.oracle_context_state.context_len
            request.verify_context_state.context_len = request.oracle_context_state.context_len
        
        return requests
    
    # def run_prefill_empty(self):
    #     flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
    #         self.registry.get("val.engine.kv_cache_block_size"), 
    #     )
    #     runner_outputs = self.model_runner.run(
    #         torch.tensor([], dtype=torch.int32),
    #         torch.tensor([], dtype=torch.int32),
    #         flashinfer_metadata_dummy,
    #         sampling_params=None,
    #         num_logits_to_keep=None,
    #         start_layer_idx=0,
    #         end_layer_idx=None
    #     )
    #     del runner_outputs
    
    # def run_decoding_empty(self):
    #     flashinfer_metadata_dummy = FlashInferMetadata.create_empty(
    #         self.registry.get("val.engine.kv_cache_block_size"), 
    #     )
        
    #     # 다른 worker가 decoding 할 때, Token Scheduling의 all_gather 싱크를 맞추기 위해 None 교환
    #     if self.registry.get('val.engine.use_data_parallel_draft'):
    #         allgather_results = [None for _ in range(self.num_workers)]
    #         torch.distributed.all_gather_object(allgather_results, None)
    #         # torch.distributed.barrier()

    #     # oracle model sync
    #     runner_outputs = self.model_runner.run(
    #         torch.tensor([], dtype=torch.int32),
    #         torch.tensor([], dtype=torch.int32),
    #         flashinfer_metadata_dummy,
    #         sampling_params=None,
    #         num_logits_to_keep=self.num_draft_steps + 1,
    #         start_layer_idx=self.verify_last_layer_idx + 1,
    #         end_layer_idx=None
    #     )
    #     del runner_outputs
    
    # @profile_nvtx("HierarchicalSpeculativeWorker._run_data_parallel_draft")
    # def _run_data_parallel_draft(
    #         self,
    #         requests: Optional[List[SpeculativeRequest]] = None,
    #         is_global_prefill_step: Optional[bool] = None,
    #     ):
    #     if is_global_prefill_step:
    #         if requests is None or not requests[0].is_prefill_step:
    #             self.run_prefill_empty()
    #             return None
    #         else:
    #             requests = self.run_prefill(requests)
    #             return requests
    #     else:
    #         # 다른 worker가 decoding 끝낼 때까지 더미 데이터로 sync
    #         if len(requests) == 0:
    #             self.run_decoding_empty()
    #             return None
    #         else:
    #             requests, rejection_sampler_outputs = self.run_hierarchical_speculation(requests)
    #             requests = self.model_io_processor.layerskip_post_process_requests(requests, rejection_sampler_outputs)
    #             return requests
    # def get_global_state(self, requests: Optional[List[Request]] = None):
    #     is_local_prefill_step = requests[0].is_prefill_step if len(requests) > 0 else False
    #     len_requests = len(requests) if requests is not None else 0
    #     global_prefill_steps = [None] * self.num_workers

    #     torch.distributed.all_gather_object(global_prefill_steps, (is_local_prefill_step, len_requests))
    #     is_global_prefill_step = any([item[0] for item in global_prefill_steps])

    #     return is_global_prefill_step
    
    # def get_global_state(self, requests: Optional[List[Request]] = None):
    #     is_local_prefill_step = requests[0].is_prefill_step if requests is not None else False
    #     is_prefill_steps = [None] * self.num_workers
    #     torch.distributed.all_gather_object(is_prefill_steps, is_local_prefill_step)
    #     is_prefill_step = any(is_prefill_steps)
    #     return is_prefill_step
    
    @profile_nvtx("HierarchicalLayerSkipWorker._run")
    def _run(
        self, 
        requests: Optional[List[HierarchicalSpeculativeRequest]] = None,
        is_prefill_step: Optional[bool] = None,
    ):
        # import pdb; pdb.set_trace()
        # requests = self.local_schedule()
        # is_local_prefill_step = requests[0].is_prefill_step if len(requests) > 0 else False
        # global_prefill_steps = [None] * self.num_workers
        # torch.distributed.all_gather_object(global_prefill_steps, is_local_prefill_step)
        # is_global_prefill_step = any(global_prefill_steps)
        if self.local_rank != 0:
            if is_prefill_step:
                self.dummy_prefill()
                return None
            else:
                self.dummy_work()
                return None


        is_prefill_step = requests[0].is_prefill_step
        if is_prefill_step:
            requests = self.run_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_hierarchical_speculation(requests)
            requests = self.model_io_processor.layerskip_post_process_requests(requests, rejection_sampler_outputs)

        return requests
    
    def _profile_latency(
            self,
            batch_size: int,
            prompt_len: int,
            num_compute_tokens: int
        ):
        request_cls = self.registry.get('cls.engine.request_class')
        block_size = self.registry.get('val.engine.kv_cache_block_size')
        vocab_size = self.registry.get('val.engine.vocab_size')

        # warmup
        for i in range(3):
            if self.local_rank != 0:
                # futures = self.model_runner.run(
                #     None, 
                #     None, 
                #     None,
                #     None,
                #     start_layer_idx=self.verify_last_layer_idx + 1,
                #     end_layer_idx=None
                # )
                # del futures
                self.dummy_work()
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.layerskip_post_process_requests(requests, rejection_sampler_outputs)

        latencies = []
        for i in range(15):
            if self.local_rank != 0:
                # futures = self.oracle_model_runner.run(
                #     None, 
                #     None, 
                #     None,
                #     None
                # )
                # del futures
                self.dummy_work()
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                start_time = time.time()
                requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.layerskip_post_process_requests(requests, rejection_sampler_outputs)
                end_time = time.time()

                latencies.append((end_time - start_time))
        
        if self.local_rank == 0:
            latencies = np.sort(latencies)[2:-2]
            latency = np.mean(latencies)
            return latency
        else:
            return None


from .model_io_processor import SmartSpecModelIOProcessor

class SmartSpecWorker(SpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, local_rank=self.local_rank)
        self.model_runner = self.verify_model_runner

        self.model_io_processor = SmartSpecModelIOProcessor(self.registry)

        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.verify_model_runner.model_executors]
            ray.get(ray_executors)
    
    @profile_nvtx("SmartSpecWorker.run_speculation")
    def run_speculation(
            self,
            requests: Optional[List['SmartSpecRequest']] = None,
        ):
        self.model_io_processor.setup_speculative_step(requests)
        
        # dirty coding.. I know...
        dynamic_draft_lengths, max_draft_length = self.model_io_processor.get_dynamic_draft_length(requests)
        original_draft_steps = self.num_draft_steps
        self.num_draft_steps = max_draft_length
        for i, request in enumerate(requests):
            request.num_draft_steps = max_draft_length
            request.num_generation_tokens = dynamic_draft_lengths[i] + 1
        
        for i in range(self.num_draft_steps):
            requests = self.run_draft(self.draft_model_runner, requests)
        requests = self.model_io_processor.mid_process(requests)
        requests = self.run_verify(self.verify_model_runner, requests)
        rejection_sampler_outputs = self.model_io_processor.run_rejection_sampler(requests)
        self.model_io_processor.reset_rejection_probs()
            
        self.num_draft_steps = original_draft_steps
        for request in requests:
            request.num_draft_steps = original_draft_steps
            request.num_generation_tokens = original_draft_steps + 1
        
        return requests, rejection_sampler_outputs
    
    
from .model_io_processor import SVIPModelIOProcessor

class SVIPWorker(SpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, local_rank=self.local_rank)
        self.model_runner = self.verify_model_runner

        self.model_io_processor = SVIPModelIOProcessor(self.registry)

        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.verify_model_runner.model_executors]
            ray.get(ray_executors)
    
    @profile_nvtx("SmartSpecWorker.run_speculation")
    def run_speculation(
            self,
            requests: Optional[List['SVIPRequest']] = None,
        ):
        self.model_io_processor.setup_speculative_step(requests)
        
        for i in range(self.num_draft_steps):
            # num_valid_requests = len([request for request in requests if not request.should_stop_drafting])
            # print(f"run_speculation[{i}], num_valid_requests: {num_valid_requests}")
            requests = self.run_draft(self.draft_model_runner, requests)
            is_all_stop = self.model_io_processor.should_stop_drafting(requests, i)
            if is_all_stop:
                break
        
        requests = self.model_io_processor.mid_process(requests)
        requests = self.run_verify(self.verify_model_runner, requests)
        rejection_sampler_outputs = self.model_io_processor.run_rejection_sampler(requests)
        self.model_io_processor.reset_rejection_probs()
        
        return requests, rejection_sampler_outputs
    
from .model_io_processor import Eagle3ModelIOProcessor

class Eagle3Worker(SpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, local_rank=self.local_rank)
        self.model_runner = self.verify_model_runner
        
        self.model_io_processor = Eagle3ModelIOProcessor(self.registry)

        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.verify_model_runner.model_executors]
            ray.get(ray_executors)

    @profile_nvtx("Eagle3Worker._run")
    def _run(
            self,
            requests: Optional[List[Eagle3Request]] = None
        ):
        if self.local_rank != 0:
            futures = self.verify_model_runner.run(
                None, 
                None, 
                None,
                None
            )
            del futures
            return None
        
        is_verify_prefill_step = requests[0].is_verify_prefill_step
        if is_verify_prefill_step:
            requests = self.run_verify_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_speculation(requests)
            requests = self.model_io_processor.speculative_post_process_requests(requests, rejection_sampler_outputs)
            
        return requests

    @profile_nvtx("Worker._prepare_draft_input_tensors")
    def _prepare_draft_input_tensors(
        self,
        requests: Optional[List[SpeculativeRequest]] = None,
        prefill: bool = False
    ):
        if requests is not None:
            # driver worker
            input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = self.model_io_processor.build_draft_input_tensors(requests, prefill=prefill)
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = None, None, None, None

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states
    
    def run_verify_prefill(
            self,
            requests: Optional[List['Eagle3Request']] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = self.verify_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests]
        )
        if self.verify_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        requests = self.model_io_processor.append_hidden_states(requests, hidden_states)
        return requests
    
    @profile_nvtx("Eagle3Worker.run_draft")
    def run_draft(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List['Eagle3Request']] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = self._prepare_draft_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=None,
            hidden_states=hidden_states,
            num_logits_to_keep=1
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        requests = self.model_io_processor.append_eagle3_draft_outputs(
            requests, sampled_tokens_per_request, probs, probs_input_ids_indptr, target_model_runner.eagle3_vocab_mapping)
        requests = self.model_io_processor.set_hidden_states(requests, hidden_states)
        return requests
    
    @profile_nvtx("Eagle3Worker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List['Eagle3Request']] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps + 1
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        
        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        requests = self.model_io_processor.set_hidden_states(requests, hidden_states)

        return requests
    
    @profile_nvtx("Worker.run_speculation")
    def run_speculation(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        self.model_io_processor.setup_speculative_step(requests)
        for i in range(self.num_draft_steps):
            requests = self.run_draft(self.draft_model_runner, requests)
        requests = self.run_verify(self.verify_model_runner, requests)

        rejection_sampler_outputs = self.model_io_processor.evaluate_eagle3_posterior(requests)
        self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs
    
    def run(self):
        if self.local_rank == 0:
            requests = self.local_schedule()
            assert len(requests) > 0, "No requests to run"
            is_verify_prefill_step = requests[0].is_verify_prefill_step
            is_draft_prefill_step = requests[0].is_draft_prefill_step
            num_requests = len(requests)
        else:
            requests = None
            is_verify_prefill_step = None
            is_draft_prefill_step = None

        start_time = time.time()
        requests = self._run(requests)
    
        if self.local_rank > 0:
            return None, None


        if is_verify_prefill_step:
            num_generated_tokens = np.sum([request.verify_context_state.last_num_generated_tokens for request in requests])
            num_processed_tokens = np.sum([request.verify_context_state.scheduled_query_len for request in requests])
        elif is_draft_prefill_step:
            # num_generated_tokens = np.sum([request.draft_context_state.last_num_generated_tokens for request in requests])
            num_generated_tokens = 1
            # num_processed_tokens = np.sum([request.draft_context_state.scheduled_query_len for request in requests])
            num_processed_tokens = 0
        else:
            num_generated_tokens = np.sum([request.verify_context_state.last_num_generated_tokens for request in requests])
            num_processed_tokens = 0

        finished_requests = self._check_requests_finished(requests)
        end_time = time.time()
        time_elapsed = end_time - start_time
        stat = {
            'is_prefill_step': is_verify_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': num_generated_tokens,
            'num_processed_tokens': num_processed_tokens
        }
        return finished_requests, stat



from .model_io_processor import Eagle3SVModelIOProcessor
class Eagle3SVWorker(HierarchicalSpeculativeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _setup_worker_components(self):
        self.draft_model_runner = ModelRunner(registry=self.registry, type='draft', local_rank=self.local_rank)
        self.verify_model_runner = ModelRunner(registry=self.registry, type='verify', local_rank=self.local_rank)
        self.oracle_model_runner = ModelRunner(registry=self.registry, type=None, local_rank=self.local_rank)
        self.model_runner = self.oracle_model_runner

        self.model_io_processor = Eagle3SVModelIOProcessor(self.registry, self.local_rank)
        
        if self.registry.get('val.engine.use_ray_executor'):
            import ray
            ray_executors = [executor.ensure_init_done.remote() for executor in self.oracle_model_runner.model_executors]
            ray.get(ray_executors)


    @profile_nvtx("Worker._prepare_draft_input_tensors")
    def _prepare_draft_input_tensors(
        self,
        requests: Optional[List[SpeculativeRequest]] = None,
        prefill: bool = False
    ):
        if requests is not None:
            # driver worker
            input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = self.model_io_processor.build_draft_input_tensors(requests, prefill=prefill)
        else:
            input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = None, None, None, None

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states
    
    
    @profile_nvtx("Eagle3SVWorker.run_draft")
    def run_draft(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List['Eagle3SVRequest']] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states = self._prepare_draft_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=None,
            hidden_states=hidden_states,
            num_logits_to_keep=1
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)

        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        requests = self.model_io_processor.append_eagle3_draft_outputs(
            requests, sampled_tokens_per_request, probs, probs_input_ids_indptr, target_model_runner.eagle3_vocab_mapping)
        requests = self.model_io_processor.set_hidden_states(requests, hidden_states)
        return requests

    
    @profile_nvtx("Worker.run_verify")
    def run_verify(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):
        
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_verify_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps
        )

        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)

        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        del hidden_states

        requests = self.model_io_processor.append_verify_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)

        return requests
    
    @profile_nvtx("Eagle3SVWorker.run_oracle")
    def run_oracle(
            self,
            target_model_runner: ModelRunner,
            requests: Optional[List['Eagle3SVRequest']] = None,
        ):
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)
        
        runner_outputs = target_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=self.num_draft_steps + 1
        )
        
        if target_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)
        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs
        
        requests = self.model_io_processor.append_oracle_outputs(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        try:
            requests = self.model_io_processor.set_hidden_states(requests, hidden_states)
        except:
            import pdb; pdb.set_trace()
        return requests
    
    @profile_nvtx("Eagle3SVWorker.run_prefill")
    def run_prefill(
            self,
            requests: Optional[List[SpeculativeRequest]] = None,
        ):

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._prepare_oracle_input_tensors(requests)

        runner_outputs = self.verify_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests],
            num_logits_to_keep=0
        )
        
        del runner_outputs

        runner_outputs = self.oracle_model_runner.run(
            input_ids_tensor, 
            position_ids_tensor, 
            flashinfer_metadata,
            sampling_params=[request.sampling_params for request in requests]
        )


        if self.oracle_model_runner.is_ray_runner:
            runner_outputs = self.get_ray_model_results(runner_outputs)

        
        sampled_tokens_per_request, probs, probs_input_ids_indptr, hidden_states = runner_outputs

        
        requests = self.model_io_processor.post_process_requests(requests, sampled_tokens_per_request, probs, probs_input_ids_indptr)
        requests = self.model_io_processor.append_hidden_states(requests, hidden_states)


        for request in requests:
            # request.draft_context_state.context_len = request.oracle_context_state.context_len
            request.verify_context_state.context_len = request.oracle_context_state.context_len
        
        return requests
    
    @profile_nvtx("Eagle3SVWorker.run_hierarchical_speculation")
    def run_hierarchical_speculation(
        self,
        requests: List[HierarchicalSpeculativeRequest],
        num_tokens_to_execute: Optional[int] = None
        ):
        self.model_io_processor.setup_oracle_step(requests)
        self.model_io_processor.setup_speculative_step(requests)
        for i in range(self.num_draft_steps):
            requests = self.run_draft(self.draft_model_runner, requests)
        
        requests = self.run_verify(self.verify_model_runner, requests)
        
        speculative_verification_outputs, scheduling_stat = self.model_io_processor.verify_info_gain(requests, num_tokens_to_execute)
        self.scheduling_stats.append(scheduling_stat)

        requests, _ = self.model_io_processor.mid_process(requests, speculative_verification_outputs)

        requests = self.run_oracle(self.oracle_model_runner, requests)
        
        requests, rejection_sampler_outputs = self.model_io_processor.run_oracle_rejection_sampler(
            requests, return_beta=True, return_accept_prob=self.profile_acceptance_prob)
        self.model_io_processor.reset_rejection_probs()

        return requests, rejection_sampler_outputs

    
    @profile_nvtx("Eagle3SVWorker._run")
    def _run(
            self,
            requests: Optional[List[Eagle3SVRequest]] = None
        ):
        if self.local_rank != 0:
            futures = self.oracle_model_runner.run(
                None, 
                None, 
                None,
                None
            )
            del futures
            return None
        
        is_oracle_prefill_step = requests[0].is_oracle_prefill_step
        if is_oracle_prefill_step:
            requests = self.run_prefill(requests)
        else:
            requests, rejection_sampler_outputs = self.run_hierarchical_speculation(requests)
            requests = self.model_io_processor.hierarchical_speculative_post_process_requests(requests, rejection_sampler_outputs)
            
        return requests


    def run(self):
        if self.local_rank == 0:
            requests = self.local_schedule()
            assert len(requests) > 0, "No requests to run"
            is_oracle_prefill_step = requests[0].is_oracle_prefill_step
            is_verify_prefill_step = requests[0].is_verify_prefill_step
            is_draft_prefill_step = requests[0].is_draft_prefill_step
            num_requests = len(requests)
        else:
            requests = None
            is_oracle_prefill_step = None
            is_verify_prefill_step = None
            is_draft_prefill_step = None

        start_time = time.time()
        requests = self._run(requests)
    
        if self.local_rank > 0:
            return None, None

        if is_oracle_prefill_step:
            num_generated_tokens = np.sum([request.oracle_context_state.last_num_generated_tokens for request in requests])
            num_processed_tokens = np.sum([request.oracle_context_state.scheduled_query_len for request in requests])
        elif is_draft_prefill_step:
            # num_generated_tokens = np.sum([request.draft_context_state.last_num_generated_tokens for request in requests])
            num_generated_tokens = 1
            # num_processed_tokens = np.sum([request.draft_context_state.scheduled_query_len for request in requests])
            num_processed_tokens = 0
        else:
            num_generated_tokens = np.sum([request.oracle_context_state.last_num_generated_tokens for request in requests])
            num_processed_tokens = 0

        finished_requests = self._check_requests_finished(requests)
        end_time = time.time()
        time_elapsed = end_time - start_time
        stat = {
            'is_prefill_step': is_verify_prefill_step,
            'time_elapsed': time_elapsed,
            'num_requests': num_requests,
            'num_generated_tokens': num_generated_tokens,
            'num_processed_tokens': num_processed_tokens
        }
        return finished_requests, stat

    
    def _profile_latency(
            self,
            batch_size: int,
            prompt_len: int,
            num_compute_tokens: int
        ):
        request_cls = self.registry.get('cls.engine.request_class')
        block_size = self.registry.get('val.engine.kv_cache_block_size')
        vocab_size = self.registry.get('val.engine.vocab_size')
        hidden_size = self.registry.get('val.model.hf_config').hidden_size

        # warmup
        for i in range(3):
            if self.local_rank != 0:
                futures = self.oracle_model_runner.run(
                    None, 
                    None, 
                    None,
                    None
                )
                del futures
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                    
                for request in dummy_requests:
                    request.append_hidden_states(torch.randn(prompt_len, hidden_size, device=self.device))
                
                dummy_requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.hierarchical_speculative_post_process_requests(dummy_requests, rejection_sampler_outputs)

        latencies = []
        for i in range(12):
            if self.local_rank != 0:
                futures = self.oracle_model_runner.run(
                    None, 
                    None, 
                    None,
                    None
                )
                del futures
            else:
                dummy_requests = create_dummy_requests(request_cls, block_size, batch_size, prompt_len, vocab_size)
                
                for request in dummy_requests:
                    request.append_hidden_states(torch.randn(prompt_len, hidden_size, device=self.device))
                
                start_time = time.time()
                dummy_requests, rejection_sampler_outputs = self.run_hierarchical_speculation(dummy_requests, num_tokens_to_execute=num_compute_tokens)
                _ = self.model_io_processor.hierarchical_speculative_post_process_requests(dummy_requests, rejection_sampler_outputs)
                end_time = time.time()

                if i > 0:
                    latencies.append((end_time - start_time))
        
        if self.local_rank == 0:
            latencies = np.sort(latencies)[2:-2]
            latency = np.mean(latencies)
            return latency
        else:
            return None
    