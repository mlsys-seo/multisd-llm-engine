import inspect
import functools

from typing import List, Union, Optional, Tuple
from collections import deque

import numpy as np

from .registry import EngineRegistry
from .request import Request
from .block_manager import PagedAttentionBlockManager


from sdprofiler.utils.common import print_debug

DEBUG = False

class ScheduleProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self) -> List[Request]:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
    
class ScheduleProcessorList(list):
    """
    This class can be used to create a list of [`ScheduleProcessor`] to subsequently process a `queue` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`ScheduleProcessor`] to the
    inputs.`
    """

    def __call__(self, waiting_queue, scheduled_queue, **kwargs) -> List[Request]:
        r"""
        """

        cut_only = False
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                # if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                #     raise ValueError(
                #         f"Make sure that all the required parameters: {list(function_args.keys())} for "
                #         f"{processor.__class__} are passed to the schedule processor."
                #     )
                waiting_queue, scheduled_queue = processor(waiting_queue, scheduled_queue, cut_only=cut_only, **kwargs)
            else:
                waiting_queue, scheduled_queue = processor(waiting_queue, scheduled_queue, cut_only=cut_only)
            cut_only = True
        return waiting_queue, scheduled_queue


class FirstInFirstOutScheduler(ScheduleProcessor):
    def __call__(
            self, 
            waiting_queue, 
            scheduled_queue,
            **kwargs
        ) -> List[Request]:
        
        while waiting_queue:
            request = waiting_queue.pop(0)
            scheduled_queue.append(request)
        
        return waiting_queue, scheduled_queue

class PrefillFirstScheduler(ScheduleProcessor):
    def __init__(self, num_max_batch_requests: int, num_min_batch_requests: Optional[int]=None):
        self.num_max_batch_requests = num_max_batch_requests
        if num_min_batch_requests is None:
            self.num_min_batch_requests = num_max_batch_requests
        else:
            self.num_min_batch_requests = num_min_batch_requests
    
    
    def __call__(
            self,
            waiting_queue: List[Request],
            scheduled_queue: List[Request], 
            cut_only: bool=False,
            divide: int=1,
            **kwargs) -> Tuple[List[Request], List[Request]]:
        
        num_min_batch_requests = int(self.num_min_batch_requests // divide)
        num_max_batch_requests = int(self.num_max_batch_requests // divide)

        scheduled_prefill_requests = [ request for request in scheduled_queue if request.is_prefill_step ]
        scheduled_generation_requests = [ request for request in scheduled_queue if not request.is_prefill_step ]
        
        waiting_prefill_requests = [ request for request in waiting_queue if request.is_prefill_step ]
        waiting_generation_requests = [ request for request in waiting_queue if not request.is_prefill_step ]


        # target_num_scheduling_requests = self.num_max_batch_requests - len(scheduled_generation_requests)
        current_gen_num_requests = len(scheduled_generation_requests)
        if current_gen_num_requests < num_min_batch_requests:
            target_num_scheduling_requests = num_max_batch_requests - current_gen_num_requests
        else:
            target_num_scheduling_requests = 0

        if DEBUG:
            print_debug(
                function_name="PrefillFirstScheduler.__call__",
                self_num_max_batch_requests=self.num_max_batch_requests,
                self_min_batch_requests=self.num_min_batch_requests,
                divide=divide,
                num_min_batch_requests=num_min_batch_requests,
                num_max_batch_requests=num_max_batch_requests,
                current_gen_num_requests=current_gen_num_requests,
                target_num_scheduling_requests=target_num_scheduling_requests,
                len_waiting_prefill_requests=len(waiting_prefill_requests),
                len_waiting_generation_requests=len(waiting_generation_requests),
                len_scheduled_prefill_requests=len(scheduled_prefill_requests),
                len_scheduled_generation_requests=len(scheduled_generation_requests),
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
            )

        # Generation phase
        if target_num_scheduling_requests <= 0:
            # Generation with running requests
            scheduled_queue = scheduled_generation_requests
            waiting_queue.extend(scheduled_prefill_requests)

        elif len(waiting_generation_requests) >= target_num_scheduling_requests:
            # Generation with add some requests
            scheduled_queue = scheduled_generation_requests
            waiting_queue = []
        
            scheduled_queue.extend(waiting_generation_requests[:target_num_scheduling_requests])
            waiting_generation_requests = waiting_generation_requests[target_num_scheduling_requests:]
            
            waiting_queue.extend(waiting_generation_requests)
            waiting_queue.extend(scheduled_prefill_requests)
            waiting_queue.extend(waiting_prefill_requests)
        
        elif len(waiting_prefill_requests) + len(scheduled_prefill_requests) == 0:
            scheduled_queue.extend(waiting_generation_requests)
            waiting_queue = []

        # Prefill phase
        else:
            waiting_generation_requests.extend(scheduled_generation_requests)
            scheduled_generation_requests = []
            scheduled_queue = []
            waiting_queue = []
            
            if len(scheduled_prefill_requests) >= target_num_scheduling_requests:
                scheduled_queue.extend(scheduled_prefill_requests[:target_num_scheduling_requests])
                scheduled_prefill_requests = scheduled_prefill_requests[target_num_scheduling_requests:]
            else:
                scheduled_queue.extend(scheduled_prefill_requests)
                target_num_scheduling_requests -= len(scheduled_prefill_requests)
                scheduled_prefill_requests = []

                num_to_schedule = min(target_num_scheduling_requests, len(waiting_prefill_requests))
                scheduled_queue.extend(waiting_prefill_requests[:num_to_schedule])
                waiting_prefill_requests = waiting_prefill_requests[num_to_schedule:]
        
            waiting_queue.extend(scheduled_prefill_requests)
            waiting_queue.extend(waiting_generation_requests)
            waiting_queue.extend(waiting_prefill_requests)
        

        if DEBUG:
            print_debug(
                function_name="PrefillFirstScheduler.__end__",
                self_num_max_batch_requests=self.num_max_batch_requests,
                num_max_batch_requests=num_max_batch_requests,
                target_num_scheduling_requests=target_num_scheduling_requests,
                len_waiting_prefill_requests=len([ request for request in waiting_queue if request.is_prefill_step ]),
                len_waiting_generation_requests=len([ request for request in waiting_queue if not request.is_prefill_step ]),
                len_scheduled_prefill_requests=len([ request for request in scheduled_queue if request.is_prefill_step ]),
                len_scheduled_generation_requests=len([ request for request in scheduled_queue if not request.is_prefill_step ]),
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
            )

        return waiting_queue, scheduled_queue

class ConstantNumTokensScheduler(ScheduleProcessor):
    def __init__(self, num_tokens: int, use_chunked_prefill: bool=False):
        self.num_tokens = num_tokens
        self.use_chunked_prefill = use_chunked_prefill

    def __call__(
            self, 
            waiting_queue: List[Request], 
            scheduled_queue: List[Request],
            cut_only: bool=False,
            **kwargs
        ) -> List[Request]:
        
        num_tokens_left = self.num_tokens
        if DEBUG:
            print_debug(
                function_name="ConstantNumTokensScheduler.__call__",
                num_tokens_left=num_tokens_left,
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
                waiting_prefill_tokens=sum([request.get_required_num_tokens() for request in waiting_queue]),
                scheduled_prefill_tokens=sum([request.get_required_num_tokens() for request in scheduled_queue]),
            )
        
        # filter out scheduled_queue
        for idx, request in enumerate(scheduled_queue):
            required_num_tokens = request.get_required_num_tokens()
            if required_num_tokens > num_tokens_left: # move to waiting_queue
                if self.use_chunked_prefill:
                    # chunk the prefill
                    request.set_scheduled_query_len(num_prompt_tokens_limit=num_tokens_left)
                    waiting_queue = scheduled_queue[idx+1:] + waiting_queue
                    scheduled_queue = scheduled_queue[:idx+1]
                else:
                    waiting_queue = scheduled_queue[idx:] + waiting_queue
                    scheduled_queue = scheduled_queue[:idx]
                break
            else: # allocate
                request.set_scheduled_query_len()
                num_tokens_left -= required_num_tokens

        if not cut_only:
            idx = 0
            while waiting_queue and num_tokens_left > 0:
                request = waiting_queue[0]
                required_num_tokens = request.get_required_num_tokens()
                if required_num_tokens >= num_tokens_left:
                    if self.use_chunked_prefill:
                        # chunk the prefill
                        request.set_scheduled_query_len(num_prompt_tokens_limit=num_tokens_left)
                        request = waiting_queue.pop(0)
                        scheduled_queue.append(request)
                    break
                else:
                    request.set_scheduled_query_len()
                    request = waiting_queue.pop(0)
                    scheduled_queue.append(request)
                    num_tokens_left -= required_num_tokens
                idx += 1

        if DEBUG:
            print_debug(
                function_name="ConstantNumTokensScheduler.__end__",
                num_tokens_left=num_tokens_left,
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
                waiting_prefill_tokens=sum([request.get_required_num_tokens() for request in waiting_queue]),
                scheduled_prefill_tokens=sum([request.allocated_query_len for request in scheduled_queue]),
            )
        
        return waiting_queue, scheduled_queue

class KVCacheBlockAllocatingScheduler(ScheduleProcessor):
    def __init__(self, block_manager: PagedAttentionBlockManager):
        self.block_manager = block_manager
        self.kv_cache_block_size = self.block_manager.kv_cache_block_size

    def __call__(
            self,
            waiting_queue: List[Request], 
            scheduled_queue: List[Request],
            cut_only: bool=False,
            **kwargs
        ) -> List[Request]:
        num_kv_cache_blocks_left = self.block_manager.num_free_blocks
        

        if DEBUG:
            print_debug(
                function_name="KVCacheBlockAllocatingScheduler.__call__",
                num_kv_cache_blocks_left=num_kv_cache_blocks_left,
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
                waiting_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in waiting_queue]),
                scheduled_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in scheduled_queue]),
            )

        # filter out scheduled_queue
        for idx, request in enumerate(scheduled_queue):
            required_paged_kv_blocks = request.calculate_required_paged_kv_blocks(self.kv_cache_block_size)
            if required_paged_kv_blocks > num_kv_cache_blocks_left:
                if num_kv_cache_blocks_left > 0:
                    request.set_scheduled_query_len(num_prompt_tokens_limit=num_kv_cache_blocks_left*self.kv_cache_block_size)
                    request.set_kv_blocks_needed(num_kv_cache_blocks_left)
                    waiting_queue = scheduled_queue[idx+1:] + waiting_queue
                    scheduled_queue = scheduled_queue[:idx+1]
                    
                else:
                    waiting_queue = scheduled_queue[idx:] + waiting_queue
                    scheduled_queue = scheduled_queue[:idx]
                if DEBUG:
                    print_debug(
                        function_name="KVCacheBlockAllocatingScheduler.__end__",
                        num_kv_cache_blocks_left=num_kv_cache_blocks_left,
                        len_waiting_queue=len(waiting_queue),
                        len_scheduled_queue=len(scheduled_queue),
                        waiting_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in waiting_queue]),
                        scheduled_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in scheduled_queue]),
                    )
                return waiting_queue, scheduled_queue
            else:
                # request.set_scheduled_query_len()
                request.set_kv_blocks_needed(required_paged_kv_blocks)
                num_kv_cache_blocks_left -= required_paged_kv_blocks
        
        if not cut_only:
            while waiting_queue:
                request = waiting_queue[0]
                required_paged_kv_blocks = request.calculate_required_paged_kv_blocks(self.kv_cache_block_size)
                if required_paged_kv_blocks >= num_kv_cache_blocks_left:
                    num_available_blocks = required_paged_kv_blocks - abs(num_kv_cache_blocks_left)
                    request = waiting_queue.pop(0)
                    request.set_scheduled_query_len(num_prompt_tokens_limit=num_available_blocks*self.kv_cache_block_size)
                    request.set_kv_blocks_needed(num_available_blocks)
                    scheduled_queue.append(request)
                    if DEBUG:
                        print_debug(
                            function_name="KVCacheBlockAllocatingScheduler.__end__",
                            num_kv_cache_blocks_left=num_kv_cache_blocks_left,
                            len_waiting_queue=len(waiting_queue),
                            len_scheduled_queue=len(scheduled_queue),
                            waiting_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in waiting_queue]),
                            scheduled_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in scheduled_queue]),
                        )
                    return waiting_queue, scheduled_queue
                else:
                    request = waiting_queue.pop(0)
                    request.set_scheduled_query_len()
                    request.set_kv_blocks_needed(required_paged_kv_blocks)
                    scheduled_queue.append(request)
                    num_kv_cache_blocks_left -= required_paged_kv_blocks

        if DEBUG:
            print_debug(
                function_name="KVCacheBlockAllocatingScheduler.__end__",
                num_kv_cache_blocks_left=num_kv_cache_blocks_left,
                len_waiting_queue=len(waiting_queue),
                len_scheduled_queue=len(scheduled_queue),
                waiting_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in waiting_queue]),
                scheduled_required_paged_kv_blocks=sum([request.calculate_required_paged_kv_blocks(self.kv_cache_block_size) for request in scheduled_queue]),
            )
                
        return waiting_queue, scheduled_queue


# TODO: check if it works
class RemoveScheduledRequests(ScheduleProcessor):
    def __init__(self, queue: List[Request], scheduled_requests: List[Request]):
        self.queue = queue
        self.scheduled_requests = scheduled_requests
    
    def __call__(self, requests: List[Request]) -> List[Request]:
        for request in self.scheduled_requests:
            self.queue.remove(request)
        return requests


class PipelinedRequestQueue:
    def __init__(self, num_pipelining_steps: int):
        self.num_pipelining_steps = num_pipelining_steps
        self.buckets = {idx: [] for idx in range(num_pipelining_steps)}
        self.current_bucket_idx = 0

    def change_bucket(self, step: int):
        self.current_bucket_idx = step % self.num_pipelining_steps

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets.values())

    def __getattr__(self, name: str):
        if name == 'current_bucket':
            return self.buckets[self.current_bucket_idx]
        else:
            return getattr(self, name)
    
    def __setattr__(self, name, value):
        if name == 'current_bucket':
            self.buckets[self.current_bucket_idx] = value
        else:
            super().__setattr__(name, value)

class Scheduler:
    def __init__(
            self,
            registry: EngineRegistry,            
        ):
        self.registry = registry
        self.num_pipelining_steps = self.registry.get('val.engine.num_pipelining_steps')
        self.num_max_batch_requests = self.registry.get('val.engine.num_max_batch_requests')

        self.waiting_requests = []
        # self.scheduled_requests = []
        self.scheduled_requests = PipelinedRequestQueue(self.num_pipelining_steps)

        self.block_manager = PagedAttentionBlockManager(self.registry)

        self.schedule_processors = self._build_schedule_processors()

        self.cuda_graph_target_batch_sizes = self.registry.get('val.engine.cuda_graph_target_batch_sizes')

    def _build_schedule_processors(self):
        # schedule_processors = self.registry.get('val.scheduler.schedule_processors')
        schedule_processors = None
        if schedule_processors is None:
            schedule_processors = ScheduleProcessorList()
            schedule_processors.append(PrefillFirstScheduler(
                num_max_batch_requests=self.registry.get('val.engine.num_max_batch_requests_pipe'),
                num_min_batch_requests=self.registry.get('val.engine.num_min_batch_requests_pipe')
            ))
            schedule_processors.append(ConstantNumTokensScheduler(
                num_tokens=self.registry.get('val.engine.num_max_batch_tokens'),
                use_chunked_prefill=self.registry.get('val.engine.use_chunked_prefill')
            ))
            schedule_processors.append(KVCacheBlockAllocatingScheduler(self.block_manager))
            # schedule_processors.append(RemoveScheduledRequests(self.waiting_requests, self.scheduled_requests))
        return schedule_processors


    @property
    def num_requests(self):
        return len(self.waiting_requests) + len(self.scheduled_requests)

    def add_request(self, requests: Union[Request, List[Request]]):
        if isinstance(requests, Request):
            requests = [requests]
        self.waiting_requests.extend(requests)


    def remove_request(self, request: Request):
        self.block_manager.deallocate_blocks_on_request(request)
        if request in self.waiting_requests:
            self.waiting_requests.remove(request)

        for bucket_idx in self.scheduled_requests.buckets.keys():
            if request in self.scheduled_requests.buckets[bucket_idx]:
                self.scheduled_requests.buckets[bucket_idx].remove(request)

    def is_empty(self):
        return not self.waiting_requests and len(self.scheduled_requests) == 0


    def schedule(self, queue_idx: int=None):
        self.waiting_requests, self.scheduled_requests.current_bucket = self.schedule_processors(
            self.waiting_requests, 
            self.scheduled_requests.current_bucket,
            devide = self.num_pipelining_steps if self.num_requests < self.num_max_batch_requests else 1
        )

        # self.scheduled_requests.extend(running_requests)

        # # it will be one of ScheduleProcessor later.
        # for request in running_requests:
        #     self.waiting_requests.remove(request)
        for request in self.scheduled_requests.current_bucket:
            self.block_manager.allocate_blocks_on_request(request)


        # if len(self.scheduled_requests.current_bucket) == 0:
        #     all_context_len = sum([request.context_state.context_len for request in self.waiting_requests])
        #     all_request_allocated_blocks = sum([request.paged_kv_cache.current_num_kv_blocks for request in self.waiting_requests])

        #     error_message = f"No requests to schedule\n"

        #     error_message += f" :: num_waiting_requests: {len(self.waiting_requests)}, num_scheduled_requests: {len(self.scheduled_requests)}\n"
        #     error_message += f" :: num_pipelining_steps: {self.num_pipelining_steps}, num_max_batch_requests: {self.num_max_batch_requests}\n"
        #     error_message += f" :: num_max_batch_tokens: {self.registry.get('val.engine.num_max_batch_tokens')}, use_chunked_prefill: {self.registry.get('val.engine.use_chunked_prefill')}\n"
        #     error_message += f" :: num_kv_cache_blocks: {self.registry.get('val.engine.num_kv_cache_blocks')}, max_context_len: {self.registry.get('val.engine.max_context_len')}\n"
        #     error_message += f" :: allocated kv blocks: {self.block_manager.num_using_blocks}, remaining kv blocks: {self.block_manager.num_free_blocks}\n"
        #     error_message += f" :: all_context_len: {all_context_len}, all_request_allocated_blocks: {all_request_allocated_blocks}\n"
            
        #     assert False, error_message


        
        # request_len = len(self.scheduled_requests.current_bucket)
        # if request_len > 0 and len(self.cuda_graph_target_batch_sizes):
        #     max_graph_batch_size = max([x for x in self.cuda_graph_target_batch_sizes if x <= request_len])
        #     self.waiting_requests = self.scheduled_requests.current_bucket[max_graph_batch_size:] + self.waiting_requests
        #     self.scheduled_requests.current_bucket = self.scheduled_requests.current_bucket[:max_graph_batch_size]

        output = self.scheduled_requests.current_bucket.copy()
        if DEBUG:
            print(f"\n\n--------------- Scheduler.schedule ---------------")
            print(f"  len self.scheduled_requests: {len(self.scheduled_requests)}")
            print(f"  len self.waiting_requests: {len(self.waiting_requests)}")
            print(f"  self.scheduled_requests.current_bucket_idx: {self.scheduled_requests.current_bucket_idx}")
            print(f"  len self.scheduled_requests.current_bucket: {len(self.scheduled_requests.current_bucket)}")
            print(f"  current:: is_prefill: {output[0].is_prefill_step if len(output) > 0 else None}")
            print(f"--------------------------------------------------")

            print(f"\n\n--------------- self.scheduled_requests ---------------")
            for idx, bucket in enumerate(self.scheduled_requests.buckets.values()):
                print(f"  len bucket {idx}: {len(bucket)}")
            print(f"--------------------------------------------------")
        
        self.scheduled_requests.change_bucket(self.scheduled_requests.current_bucket_idx + 1)
        return output
    
    @property
    def current_bucket_idx(self):
        return self.scheduled_requests.current_bucket_idx
    
    def clear(self):
        for bucket_idx, bucket in self.scheduled_requests.buckets.items():
            for request in bucket:
                self.block_manager.deallocate_blocks_on_request(request)
                self.scheduled_requests.buckets[bucket_idx].remove(request)
                del request
        
        for request in self.waiting_requests:
            self.block_manager.deallocate_blocks_on_request(request)
            self.waiting_requests.remove(request)
            del request
            
    
    # def match_batchs_to_graph_size(self, scheduled_requests: List[Request]):
    #     request_len = len(scheduled_requests)
    #     max_graph_batch_size = max([x for x in self.cuda_graph_target_batch_sizes if x <= request_len])
    #     self.waiting_requests.extend(scheduled_requests[max_graph_batch_size:])
    #     scheduled_requests = scheduled_requests[:max_graph_batch_size]
    #     return scheduled_requests