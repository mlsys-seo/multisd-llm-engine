import uuid
import time

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import flashinfer
from flashinfer.triton.page import get_batch_indices_positions_kernel

import numpy as np

from .utils.common import print_debug
from .sampler import SamplingParams


DEBUG = False

@dataclass
class FlashInferMetadata:
    input_ids_indptr: torch.Tensor
    input_ids_lengths: torch.Tensor
    batch_indices: torch.Tensor
    positions: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_last_page_len: torch.Tensor

    input_ids_indptr_cpu: Optional[np.ndarray] = None
    input_ids_lengths_cpu: Optional[np.ndarray] = None

    paged_kv_indices_len: Optional[int] = None
    kv_cache_block_size: Optional[int] = None

    num_all_tokens: Optional[int] = None
    num_all_tokens_tensor: Optional[torch.Tensor] = None
    num_requests: Optional[int] = None

    def __post_init__(self):
        if self.num_all_tokens is None:
            self.num_all_tokens = self.input_ids_indptr[-1].item() if len(self.input_ids_indptr) > 0 else 0
            self.num_all_tokens_tensor = self.input_ids_indptr[-1].detach().clone() if len(self.input_ids_indptr) > 0 else torch.tensor(0)
    
        if self.num_requests is None:
            self.num_requests = self.input_ids_indptr.size(0) - 1 if len(self.input_ids_indptr) > 0 else 0

        if isinstance(self.input_ids_indptr, torch.Tensor):
            self.input_ids_indptr_cpu = self.input_ids_indptr.numpy()
            self.input_ids_lengths_cpu = self.input_ids_lengths.numpy()
    
    @classmethod
    def create_empty(cls, kv_cache_block_size: int):
        return cls(
            input_ids_indptr=torch.tensor([], dtype=torch.int32),
            input_ids_lengths=torch.tensor([], dtype=torch.int32),
            batch_indices=None,
            positions=None,
            paged_kv_indices=torch.tensor([], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([], dtype=torch.int32),
            kv_cache_block_size=kv_cache_block_size
        )
    
    def to_cpu(self):
        self.input_ids_indptr = self.input_ids_indptr.cpu()
        self.input_ids_lengths = self.input_ids_lengths.cpu()
        self.paged_kv_indices = self.paged_kv_indices.cpu()
        self.paged_kv_indptr = self.paged_kv_indptr.cpu()
        self.paged_kv_last_page_len = self.paged_kv_last_page_len.cpu()
    
    def serialize(self):
        return [
            self.input_ids_indptr,
            self.input_ids_lengths,
            self.paged_kv_indices,
            self.paged_kv_indptr,
            self.paged_kv_last_page_len,
            self.kv_cache_block_size,
        ]
    
    @classmethod
    def deserialize(cls, object_list: List[torch.Tensor], to_tensor: bool = False):
        print(f"FlashInferMetadata.deserialize: {object_list}")
        
        return cls(
            input_ids_indptr=object_list[0],
            input_ids_lengths=object_list[1],
            paged_kv_indices=object_list[2],
            paged_kv_indptr=object_list[3],
            paged_kv_last_page_len=object_list[4],
            kv_cache_block_size=object_list[5],
            batch_indices=None,
            positions=None,
        )

    @classmethod
    def create_dummy(
        cls, 
        kv_cache_block_size: int,
    ):
        input_ids_indptr = np.array([])
        input_ids_lengths = np.array([])
        paged_kv_indices = np.array([])
        paged_kv_indptr = np.array([])

        input_ids_indptr = torch.tensor(input_ids_indptr, dtype=torch.int32)
        input_ids_lengths = torch.tensor(input_ids_lengths, dtype=torch.int32)
        paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32)
        paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32)
        paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32)
        
        return cls(
            input_ids_indptr=input_ids_indptr,
            input_ids_lengths=input_ids_lengths,
            batch_indices=None,
            positions=None,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            kv_cache_block_size=kv_cache_block_size
        )
    

    # only for graph execution
    @torch.no_grad()
    def copy_(
        self, 
        flashinfer_metadata: 'FlashInferMetadata',
        prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper] = None,
        kv_cache_block_size: Optional[int] = None,
    ):

        self.input_ids_indptr_cpu = flashinfer_metadata.input_ids_indptr_cpu
        self.input_ids_lengths = flashinfer_metadata.input_ids_lengths

        if self.num_all_tokens != flashinfer_metadata.num_all_tokens:
            self._copy_unsized(flashinfer_metadata, prefill_wrapper, kv_cache_block_size)
            return

        self.num_all_tokens_tensor.copy_(flashinfer_metadata.num_all_tokens_tensor, non_blocking=True)
        seq_lens = torch.clamp(prefill_wrapper._paged_kv_indptr_buf[1:] - prefill_wrapper._paged_kv_indptr_buf[:-1] - 1, min=0) * kv_cache_block_size + prefill_wrapper._paged_kv_last_page_len_buf
        get_batch_indices_positions_kernel[(self.num_requests,)](
            prefill_wrapper._qo_indptr_buf, seq_lens, self.batch_indices, self.positions, num_stages=2
        )


    @torch.no_grad()
    def _copy_unsized(
        self, 
        flashinfer_metadata: 'FlashInferMetadata',
        prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper] = None,
        kv_cache_block_size: Optional[int] = None,
    ):
        current_num_tokens = flashinfer_metadata.num_all_tokens
        self.num_all_tokens_tensor.copy_(flashinfer_metadata.num_all_tokens_tensor, non_blocking=True)

        seq_lens = torch.clamp(prefill_wrapper._paged_kv_indptr_buf[1:] - prefill_wrapper._paged_kv_indptr_buf[:-1] - 1, min=0) * kv_cache_block_size + prefill_wrapper._paged_kv_last_page_len_buf
        get_batch_indices_positions_kernel[(self.num_requests,)](
            prefill_wrapper._qo_indptr_buf, seq_lens, self.batch_indices[:current_num_tokens], self.positions[:current_num_tokens], num_stages=2
        )

    @torch.no_grad()
    def set_append_metadata(
            self,
            prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
            kv_cache_block_size: int,
    ):
        if self.input_ids_indptr_cpu is None:
            self.input_ids_indptr_cpu = self.input_ids_indptr.numpy()
            self.input_ids_lengths_cpu = self.input_ids_lengths.numpy()

        device = prefill_wrapper.device
        self.input_ids_indptr = self.input_ids_indptr.to(device, non_blocking=True)
        self.num_all_tokens_tensor = self.num_all_tokens_tensor.to(device, non_blocking=True)

        seq_lens = torch.clamp(prefill_wrapper._paged_kv_indptr_buf[1:] - prefill_wrapper._paged_kv_indptr_buf[:-1] - 1, min=0) * kv_cache_block_size + prefill_wrapper._paged_kv_last_page_len_buf
        self.batch_indices = torch.empty((self.num_all_tokens,), device=device, dtype=torch.int32)
        self.positions = torch.empty((self.num_all_tokens,), device=device, dtype=torch.int32)
        
        get_batch_indices_positions_kernel[(self.num_requests,)](
            prefill_wrapper._qo_indptr_buf, seq_lens, self.batch_indices, self.positions, num_stages=2
        )

        torch.cuda.synchronize()



    
    def __str__(self):
        return (
            f"input_ids_indptr: {self.input_ids_indptr}\n"
            f"input_ids_indptr.shape: {self.input_ids_indptr.shape}\n"
            f"input_ids_lengths: {self.input_ids_lengths}\n"
            f"input_ids_lengths.shape: {self.input_ids_lengths.shape}\n"
            f"batch_indices: {self.batch_indices}\n"
            f"batch_indices.shape: {self.batch_indices.shape if self.batch_indices is not None else None}\n"
            f"positions: {self.positions}\n"
            f"positions.shape: {self.positions.shape if self.positions is not None else None}\n"
            f"paged_kv_indices: {self.paged_kv_indices}\n"
            f"paged_kv_indices.shape: {self.paged_kv_indices.shape}\n"
            f"paged_kv_indptr: {self.paged_kv_indptr}\n"
            f"paged_kv_indptr.shape: {self.paged_kv_indptr.shape}\n"
            f"paged_kv_last_page_len: {self.paged_kv_last_page_len}\n"
            f"paged_kv_last_page_len.shape: {self.paged_kv_last_page_len.shape}\n"
            
            f"input_ids_indptr_cpu: {self.input_ids_indptr_cpu}\n"
            f"input_ids_lengths_cpu: {self.input_ids_lengths_cpu}\n"
            f"paged_kv_indices_len: {self.paged_kv_indices_len}\n"
            f"kv_cache_block_size: {self.kv_cache_block_size}\n"
            f"num_all_tokens: {self.num_all_tokens}\n"
            f"num_requests: {self.num_requests}"
        )



@dataclass
class RequestOutput:
    request_id: str

    prompt_text: str
    generated_text: str
    generated_ids: np.ndarray

    prompt_len: int
    generated_len: int

    queue_wait_time: float
    prefill_latency: float
    generation_latency: float
    overall_latency: float

    prompt_log_probs: np.ndarray = field(default_factory=lambda: None)
    generated_log_probs: np.ndarray = field(default_factory=lambda: None)
    
    
class RequestState(Enum):
    NEW = "new"
    TOKENIZED = "tokenized"  
    PREFILL = "prefill"
    GENERATING = "generating"
    FINISHED = "finished"


@dataclass
class RequestKVCache:
    paged_kv_block_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    current_num_required_blocks: int = 0
    kv_blocks_needed: int = 0


    def __str__(self):
        return (
            f"paged_kv_block_indices: {self.paged_kv_block_indices}"
        )

    @property
    def current_num_kv_blocks(self):
        return self.paged_kv_block_indices.shape[0]
    
    def calculate_required_paged_kv_blocks(self, target_len: int, page_size: int):
        num_required_blocks = np.ceil((target_len+1) / page_size) - self.current_num_kv_blocks
        num_required_blocks = np.maximum(0, int(num_required_blocks))
        # if DEBUG:
        #     print_debug(
        #         function_name="RequestKVCache.calculate_required_paged_kv_blocks",
        #         target_len=target_len,
        #         page_size=page_size,
        #         len_self_paged_kv_block_indices=len(self.paged_kv_block_indices),
        #         num_required_blocks=num_required_blocks
        #     )
        return num_required_blocks
        

    def set_kv_blocks_needed(self, kv_blocks_needed: int):
        self.kv_blocks_needed = kv_blocks_needed


    def append_paged_kv_blocks(self, allocating_paged_kv_blocks: np.ndarray):
        assert self.kv_blocks_needed == len(allocating_paged_kv_blocks), "The number of blocks needed does not match the number of blocks allocated"
        
        self.paged_kv_block_indices = np.concatenate([self.paged_kv_block_indices, allocating_paged_kv_blocks], axis=0)
        
        if DEBUG:
            print_debug(
                function_name="RequestKVCache.append_paged_kv_blocks",
                self_current_num_required_blocks=self.current_num_required_blocks,
                allocating_paged_kv_blocks=allocating_paged_kv_blocks,
                self_paged_kv_block_indices=self.paged_kv_block_indices
            )
        self.kv_blocks_needed = 0

    
    def prepare_paged_kv_blocks(self, context_len: int, page_size: int) -> Tuple[np.ndarray, np.ndarray]:
        num_blocks = np.ceil(context_len / page_size).astype(int)
        
        last_page_len = context_len % page_size
        if last_page_len == 0 and context_len > 0:
            last_page_len = page_size
        last_page_len = np.array([last_page_len])

        block_indices = self.paged_kv_block_indices[:num_blocks] 

        return block_indices, last_page_len


@dataclass
class ContextState:
    context_len: int = 0
    scheduled_query_len: int = 0
    num_generation_tokens: int = 1
    last_num_generated_tokens: int = 0

    @property
    def scheduled_context_len(self):
        return self.context_len + self.scheduled_query_len

    @property
    def reset_eagle3_context_len(self):
        self.context_len = 0
        self.scheduled_query_len = 0
        self.num_generation_tokens = 1
        self.last_num_generated_tokens = 0


@dataclass
class Request:
    request_id: Optional[str] = None
    prompt_text: Optional[str] = None
    generated_text: Optional[str] = None
    prompt_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    generated_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sampling_params: Optional[SamplingParams] = None

    prompt_log_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    generated_log_probs: np.ndarray = field(default_factory=lambda: np.array([]))

    is_finished: bool = False
    last_generated_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    num_generation_tokens: int = 1

    # for execution
    context_state: ContextState = field(default_factory=lambda: ContextState())
    allocated_query_len: int = 0
    
    # for paged kv cache
    # paged_kv_block_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # current_num_required_blocks: int = 0
    paged_kv_cache: RequestKVCache = field(default_factory=lambda: RequestKVCache())
    
    gpu_id: Optional[int] = None

    # for statistics
    arrival_time: Optional[float] = None
    tokenized_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    generation_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None

    def __post_init__(self, **kwargs):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

        if self.prompt_text is None and self.prompt_ids is None:
            raise ValueError("Either prompt_text or prompt_ids must be provided.")
        
        if self.prompt_ids is None:
            self.prompt_ids = np.array([], dtype=np.int64)

        if self.sampling_params is None:
            self.sampling_params = SamplingParams()
        
        if self.is_tokenized:
            self.arrival_time = self.tokenized_time = time.time()
        else:
            self.arrival_time = time.time()

    @property
    def prompt_len(self):
        return self.prompt_ids.shape[0]
    
    @property
    def remain_prompt_len(self):
        return self.prompt_len - self.context_state.context_len
    
    @property
    def generated_len(self):
        return self.generated_ids.shape[0]

    @property
    def is_tokenized(self):
        if self.prompt_ids is None:
            return False
        if self.prompt_ids.shape[0] == 0:
            return False
        return True
    
    @property
    def is_prefill_step(self):
        return self.context_state.context_len < self.prompt_len
    
    @property
    def paged_kv_block_indices(self):
        return self.paged_kv_cache.paged_kv_block_indices
    
    @property
    def current_num_kv_blocks(self):
        return self.paged_kv_cache.current_num_kv_blocks
    
    @property
    def kv_blocks_needed(self):
        return self.paged_kv_cache.kv_blocks_needed
    
    @property
    def overall_latency(self):
        if self.is_finished:
            return self.generation_end_time - self.prefill_start_time
        else:
            return None
    
    @property
    def queue_wait_time(self):
        if self.prefill_start_time is None:
            return None
        return self.prefill_start_time - self.tokenized_time

    @property
    def prefill_latency(self):
        if self.is_finished:
            return self.prefill_end_time - self.prefill_start_time
        else:
            return None
    
    @property
    def generation_latency(self):
        if self.is_finished:
            if self.generation_start_time is None:
                return None
            else:
                return self.generation_end_time - self.generation_start_time
        else:
            return None
    @property
    def awaiting_latency(self):
        if self.is_finished:
            if self.generation_start_time is None:
                return None
            else:
                return self.generation_start_time - self.prefill_end_time
        else:
            return None


    def tokenize(self, tokenizer):
        self.prompt_ids = tokenizer.encode(self.prompt_text, return_tensors='np')[0]
        self.context_state.scheduled_query_len = self.prompt_len
        self.tokenized_time = time.time()
    
    def get_prompt_generated_len(self, generated_ids_len: int, remain_prompt_len: int, is_prefill_step: bool):
        if is_prefill_step:
            if remain_prompt_len > generated_ids_len:
                prompt_generated_idx = generated_ids_len
            else:
                prompt_generated_idx = remain_prompt_len - 1
        else:
            prompt_generated_idx = 0
        return prompt_generated_idx
    
    def convert_to_request_output(self):
        return RequestOutput(
            request_id=self.request_id,
            prompt_text=self.prompt_text,
            generated_text=self.generated_text,
            generated_ids=self.generated_ids,
            prompt_len=self.prompt_len,
            generated_len=self.generated_len,
            queue_wait_time=self.queue_wait_time,
            prefill_latency=self.prefill_latency,
            generation_latency=self.generation_latency,
            overall_latency=self.overall_latency,
            prompt_log_probs=self.prompt_log_probs,
            generated_log_probs=self.generated_log_probs,
        )
    

    """
    for Scheduler (adjust memory/num_running_tokens)
    """
    def get_required_num_tokens(self):
        if self.is_prefill_step:
            return self.prompt_len - self.context_state.context_len
        else:
            return self.num_generation_tokens


    def set_scheduled_query_len(
            self,
            num_prompt_tokens_limit: Optional[int] = None):
        
        if self.is_prefill_step:
            allocated_query_len = self.remain_prompt_len + self.num_generation_tokens - 1
        else:
            allocated_query_len = self.num_generation_tokens

        if num_prompt_tokens_limit is not None:
            allocated_query_len = np.minimum(allocated_query_len, num_prompt_tokens_limit)
        
        self.allocated_query_len = allocated_query_len
        self.context_state.scheduled_query_len = allocated_query_len
        
        if DEBUG:
            print_debug(
                function_name="Request.set_scheduled_query_len",
                self_is_prefill_step=self.is_prefill_step,
                num_prompt_tokens_limit=num_prompt_tokens_limit,
                prompt_len=self.prompt_len,
                context_len=self.context_state.context_len,
                num_generation_tokens=self.num_generation_tokens,
                allocated_query_len=allocated_query_len,
                self_context_state_scheduled_query_len=self.context_state.scheduled_query_len,
                )
                    
    def calculate_required_paged_kv_blocks(self, page_size: int):
        return self.paged_kv_cache.calculate_required_paged_kv_blocks(self.context_state.scheduled_context_len, page_size)

    def set_kv_blocks_needed(self, kv_blocks_needed: int):
        self.paged_kv_cache.set_kv_blocks_needed(kv_blocks_needed)

    def append_paged_kv_blocks(self, allocating_paged_kv_blocks: np.ndarray):
        self.paged_kv_cache.append_paged_kv_blocks(allocating_paged_kv_blocks)


    """
    for ModelIOProcessor (build input tensors)
    """
    def prepare_input_ids(self):
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.context_state.context_len:self.context_state.scheduled_context_len]
        else:
            input_ids = self.generated_ids[-1:]
            if self.generation_start_time is None:
                self.generation_start_time = time.time()

        position_ids = np.arange(self.context_state.context_len, self.context_state.context_len + input_ids.shape[0])

        
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_draft_input_ids",
                request_id=self.request_id,
                input_ids=input_ids,
                position_ids=position_ids,
                len_input_ids=input_ids.shape[0],
                context_len=self.context_state.context_len,
            )
        return input_ids, position_ids
    

    def prepare_paged_kv_blocks(self, page_size: int):
        block_indices, last_page_len = self.paged_kv_cache.prepare_paged_kv_blocks(self.context_state.scheduled_context_len, page_size)
        if DEBUG:
            print_debug(
                function_name="Request.prepare_paged_kv_blocks",
                block_indices=block_indices,
                last_page_len=last_page_len,
            )
        return block_indices, last_page_len
    
    def append_outputs(self, generated_ids: np.ndarray, log_probs: np.ndarray):
        # define prompt / generated ids
        if self.is_prefill_step:
            generated_ids_len = generated_ids.shape[0]
            prompt_generated_idx = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)
            prompt_generated_ids = generated_ids[:prompt_generated_idx]
            generated_ids = generated_ids[prompt_generated_idx:]
        else:
            prompt_generated_ids = []
            self.last_generated_ids = generated_ids

        # if will be generating step, set decoding
        if self.is_prefill_step:
            if self.context_state.scheduled_context_len >= self.prompt_len:
                self.prefill_end_time = time.time()
                self.sampling_params.is_prefill = False
                if self.generation_start_time is None:
                    self.generation_start_time = time.time()

        finished = False
        if len(generated_ids) > 0:
            if not self.sampling_params.disable_eos_token:
                if np.any(np.isin(generated_ids, self.sampling_params.eos_token_ids)):
                    finished = True
                    self.record_finished()
        self.generated_ids = np.concatenate([self.generated_ids, generated_ids], axis=0)
        if self.is_prefill_step:
            self.context_state.last_num_generated_tokens = generated_ids.shape[0]
            # self.context_state.last_num_generated_tokens = self.context_state.scheduled_query_len
            self.context_state.context_len += self.context_state.scheduled_query_len
        else:

            self.context_state.last_num_generated_tokens = generated_ids.shape[0]
            self.context_state.context_len += generated_ids.shape[0]

        if DEBUG:
            print_debug(
                function_name="Request.append_outputs",
                request_id=self.request_id,
                generated_ids=generated_ids,
                context_len=self.context_state.context_len,
            )
        self.reset_scheduled_query_len()
        return finished


    def reset_scheduled_query_len(self):
        if self.is_prefill_step:
            self.context_state.scheduled_query_len = self.prompt_len - self.context_state.context_len
        else:
            self.context_state.scheduled_query_len = self.num_generation_tokens

    def record_finished(self):
        self.is_finished = True
        self.generation_end_time = time.time()


    @classmethod
    def create_dummy(cls, prompt_len: int, vocab_size: int, paged_kv_block_indices: np.ndarray, is_prefill_step: bool = False):
        request = cls(
            prompt_ids=np.random.randint(0, vocab_size, (prompt_len,)),
            sampling_params=SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05)
        )
        request.paged_kv_cache.paged_kv_block_indices = paged_kv_block_indices
        if not is_prefill_step:
            request.generated_ids = np.random.randint(0, vocab_size, (1,))
            for attr in request.__dict__:
                if type(getattr(request, attr)) == ContextState:
                    getattr(request, attr).context_len = prompt_len
        return request


@dataclass
class RejectionSamplerOutput():
    sampled_tokens: np.ndarray
    emitted_token_num: int
    num_draft_tokens: int = 0
    num_alive_tokens: Optional[int] = None
    failed_token_num: Optional[int] = None
    not_use_resampled_tokens: bool = False
    beta_list: np.ndarray = field(default_factory=lambda: np.array([]))
    accept_prob_list: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    full_expected_tokens: Optional[int] = None

    def __post_init__(self):
        self.failed_token_num = self.num_draft_tokens - self.emitted_token_num
        self.not_use_resampled_tokens = bool(self.emitted_token_num == self.num_alive_tokens)


@dataclass
class RejectionSamplerState():
    token_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # probs: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float))
    num_appended_tokens: int = 0
    len_token_ids: Optional[int] = None
    device: Optional[str] = None
    vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.len_token_ids is not None:
            self.token_ids = np.zeros((self.len_token_ids,), dtype=np.int64)
            # self.probs = torch.zeros((self.len_token_ids, self.vocab_size), dtype=torch.float, device=self.device)
            self.is_len_token_ids_fixed = True
        else:
            self.token_ids = np.array([], dtype=np.int64)
            # self.probs = torch.tensor([], dtype=torch.float, device=self.device)
        
        self.num_appended_tokens = 0

    @property
    def is_len_token_ids_fixed(self):
        return self.len_token_ids is not None

    def append_tokens(self, token_ids: np.ndarray):
        self.token_ids = np.concatenate([self.token_ids, token_ids], axis=0)
        self.num_appended_tokens += token_ids.shape[0]

    def cut_tokens(self, rejection_sampler_output: RejectionSamplerOutput):
        len_sampled_tokens = rejection_sampler_output.sampled_tokens.shape[0]
        self.token_ids = rejection_sampler_output.sampled_tokens
        self.num_appended_tokens = len_sampled_tokens

    def cut_tokens_by_length(self, length: int):
        self.token_ids = self.token_ids[:length]
        self.num_appended_tokens = length

    def log_probs(self):
        return None # torch.log(self.probs.sum(dim=-1)).cpu().numpy()

    def clear(self):
        self.token_ids = np.array([], dtype=np.int64)
        # if self.is_len_token_ids_fixed:
        #     self.probs.zero_()
        # else:
        #     self.probs = torch.tensor([], dtype=torch.float, device=self.device)
        self.num_appended_tokens = 0


@dataclass
class RejectionSamplerStat:
    num_iterations: int = 0
    num_compute_tokens: List[int] = field(default_factory=lambda: [])
    num_alive_tokens: List[int] = field(default_factory=lambda: [])
    emitted_token_num: List[int] = field(default_factory=lambda: [])
    beta_list: List[List[float]] = field(default_factory=lambda: [])
    accept_prob_list: List[List[float]] = field(default_factory=lambda: [])
    expected_tokens: List[List[float]] = field(default_factory=lambda: [])
    full_expected_tokens: List[List[float]] = field(default_factory=lambda: [])

    def append(
            self,
            num_compute_tokens: int,
            num_alive_tokens: int,
            emitted_token_num: int,
            beta_list: List[float],
            accept_prob_list: List[float],
            expected_tokens: Optional[float] = None,
            full_expected_tokens: Optional[float] = None
        ):
        self.num_iterations += 1
        self.num_compute_tokens.append(num_compute_tokens)
        self.num_alive_tokens.append(num_alive_tokens)
        self.emitted_token_num.append(emitted_token_num)
        self.beta_list.append(beta_list)
        self.accept_prob_list.append(accept_prob_list)
        if expected_tokens is not None:
            self.expected_tokens.append(expected_tokens)
        if full_expected_tokens is not None:
            self.full_expected_tokens.append(full_expected_tokens)



@dataclass
class SpeculativeRequestOutput(RequestOutput):
    num_draft_steps: int = 0
    verify_stat: RejectionSamplerStat = field(default_factory=lambda: RejectionSamplerStat())



@dataclass
class SpeculativeRequest(Request):
    num_draft_steps: int = 0
    vocab_size: int = 0
    
    draft_context_state: ContextState = field(default_factory=lambda: ContextState())
    verify_context_state: ContextState = field(default_factory=lambda: ContextState())
    draft_output_state: Optional[RejectionSamplerState] = None
    verify_output_state: Optional[RejectionSamplerState] = None

    # statistics
    verify_stat: RejectionSamplerStat = field(default_factory=lambda: RejectionSamplerStat())


    @property
    def draft_len_generated(self):
        return self.draft_output_state.num_appended_tokens
    

    @property
    def is_draft_prefill_step(self):
        return self.draft_context_state.context_len < self.prompt_len
    

    @property
    def remain_draft_prompt_len(self):
        return self.prompt_len - self.draft_context_state.context_len


    @property
    def remain_draft_len(self):
        return self.num_draft_steps - self.draft_len_generated
    

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.num_generation_tokens = self.num_draft_steps + 1
        self.use_verify_context()


    def use_draft_context(self):
        self.context_state = self.draft_context_state


    def use_verify_context(self):
        self.context_state = self.verify_context_state


    def __str__(self):
        return super().__str__()
    

    def __repr__(self):
        return super().__repr__()
    

    def setup_speculative_step(self, device: str):
        if self.draft_output_state is None:
            self.draft_output_state = RejectionSamplerState(device=device, vocab_size=self.vocab_size)
        else:
            self.draft_output_state.clear()
        if self.verify_output_state is None:
            self.verify_output_state = RejectionSamplerState(device=device, vocab_size=self.vocab_size)
        else:
            self.verify_output_state.clear()

    
    def prepare_draft_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
            else:
                process_token_len = self.verify_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
            
            if self.generation_start_time is None:
                self.generation_start_time = time.time()

        position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
        
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_draft_input_ids",
                request_id=self.request_id,
                draft_input_ids=input_ids,
                draft_position_ids=position_ids,
                len_draft_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )
        
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids

    def append_draft_outputs(
            self,
            generated_ids: np.ndarray
        ):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_idx = self.get_prompt_generated_len(generated_ids.shape[0], self.remain_draft_prompt_len, self.is_draft_prefill_step)
        
        if self.is_draft_prefill_step:
            draft_generated_ids = generated_ids[prompt_generated_idx:]
        else:
            draft_generated_ids = generated_ids[-1:]

        self.draft_output_state.append_tokens(draft_generated_ids)
        self.draft_context_state.context_len += self.draft_context_state.scheduled_query_len
        
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.append_draft_outputs",
                self_is_draft_prefill_step=self.is_draft_prefill_step,
                generated_ids=generated_ids,
                draft_generated_ids=self.draft_output_state.token_ids,
                draft_len_generated=self.draft_output_state.num_appended_tokens,
            )

    def prepare_verify_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.verify_context_state.context_len:self.verify_context_state.scheduled_context_len]
        else:
            input_ids = np.concatenate([self.generated_ids[-1:], self.draft_output_state.token_ids], axis=0)
        
        position_ids = np.arange(self.verify_context_state.context_len, self.verify_context_state.context_len + input_ids.shape[0])

        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_verify_input_ids",
                request_id=self.request_id,
                verify_input_ids=input_ids,
                verify_position_ids=position_ids,
                len_verify_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )
        self.verify_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids
    
    
    def append_verify_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)

        if self.is_prefill_step:
            verify_generated_ids = generated_ids[prompt_generated_len:]
        else:
            verify_generated_ids = generated_ids[-(self.draft_output_state.num_appended_tokens + 1):]

        if self.is_prefill_step:
            self.verify_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()

        self.verify_output_state.append_tokens(verify_generated_ids)
                
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.append_verify_outputs",
                generated_ids=generated_ids,
                self_verify_generated_ids=self.verify_output_state.token_ids,
                self_prompt_len=self.prompt_len,
            )
    def prepare_verify_rejection_tensors(self) -> Tuple[np.ndarray]:
        return self.draft_output_state.token_ids
    
    def cut_speculative_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        # rejection_sampler_output.cut_tokens(self.num_draft_steps)
        num_compute_tokens = self.draft_output_state.num_appended_tokens
        self.verify_output_state.cut_tokens(rejection_sampler_output)

        self.verify_stat.append(
            num_compute_tokens=rejection_sampler_output.num_draft_tokens,
            num_alive_tokens=rejection_sampler_output.num_alive_tokens,
            emitted_token_num=rejection_sampler_output.emitted_token_num,
            beta_list=rejection_sampler_output.beta_list,
            accept_prob_list=rejection_sampler_output.accept_prob_list,
        )
        return rejection_sampler_output

    def speculative_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        if DEBUG:
            print_debug(
                function_name="speculative_append_outputs_before_cut",
                request_id=self.request_id,
                verify_output_state_token_ids=self.verify_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )

        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.use_verify_context()
        super().append_outputs(rejection_sampler_output.sampled_tokens, self.verify_output_state.log_probs())
        self.draft_context_state.context_len = min(self.verify_context_state.context_len, self.draft_context_state.context_len)
        
        if DEBUG:
            print_debug(
                function_name="speculative_append_outputs_after_cut",
                request_id=self.request_id,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )

    def convert_to_request_output(self):
        return SpeculativeRequestOutput(
            request_id=self.request_id,
            prompt_text=self.prompt_text,
            generated_text=self.generated_text,
            generated_ids=self.generated_ids,
            prompt_len=self.prompt_len,
            generated_len=self.generated_len,
            queue_wait_time=self.queue_wait_time,
            prefill_latency=self.prefill_latency,
            generation_latency=self.generation_latency,
            overall_latency=self.overall_latency,
            prompt_log_probs=self.prompt_log_probs,
            generated_log_probs=self.generated_log_probs,
            num_draft_steps=self.num_draft_steps,
            verify_stat=self.verify_stat,
        )

@dataclass
class HierarchicalSpeculativeRequestOutput(SpeculativeRequestOutput):
    num_verify_steps: int = 0
    oracle_stat: RejectionSamplerStat = field(default_factory=lambda: RejectionSamplerStat())

@dataclass
class HierarchicalSpeculativeRequest(SpeculativeRequest):
    num_verify_steps: int = 1

    oracle_context_state: ContextState = field(default_factory=lambda: ContextState())
    oracle_output_state: Optional[RejectionSamplerState] = None
    oracle_stat: RejectionSamplerStat = field(default_factory=lambda: RejectionSamplerStat())


    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.oracle_context_state.num_generation_tokens = self.num_draft_steps + 1
        self.verify_context_state.num_generation_tokens = self.num_draft_steps
        self.use_oracle_context()


    @property
    def is_verify_prefill_step(self):
        return self.verify_context_state.context_len < self.prompt_len


    @property
    def is_oracle_prefill_step(self):
        return self.oracle_context_state.context_len < self.prompt_len

    def prepare_draft_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        # if self.is_draft_prefill_step:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
            else:
                process_token_len = self.oracle_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
        
            if self.generation_start_time is None:
                self.generation_start_time = time.time()


        position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_draft_input_ids",
                request_id=self.request_id,
                self_is_prefill_step=self.is_prefill_step,
                draft_input_ids=input_ids,
                draft_position_ids=position_ids,
                len_draft_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len
            )
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids

    def prepare_verify_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.verify_context_state.context_len:self.verify_context_state.scheduled_context_len]
        else:
            process_token_len = self.oracle_context_state.context_len - self.verify_context_state.context_len + 1
            input_ids = np.concatenate([self.generated_ids[-process_token_len:], self.draft_output_state.token_ids[:-1]], axis=0)
        position_ids = np.arange(self.verify_context_state.context_len, self.verify_context_state.context_len + input_ids.shape[0])

        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.prepare_verify_input_ids",
                request_id=self.request_id,
                verify_input_ids=input_ids,
                verify_position_ids=position_ids,
                len_verify_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        self.verify_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids

    def append_verify_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)

        if self.is_prefill_step:
            verify_generated_ids = generated_ids[prompt_generated_len:]
        else:
            verify_generated_ids = generated_ids[-(self.draft_output_state.num_appended_tokens):]

        if self.is_prefill_step:
            self.verify_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()
                
        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.append_verify_outputs",
                generated_ids=generated_ids,
                self_verify_generated_ids=self.verify_output_state.token_ids,
                self_prompt_len=self.prompt_len,
                self_vocab_size=self.vocab_size,
            )
        self.verify_output_state.append_tokens(verify_generated_ids)

    def mid_process(self, rejection_sampler_output: RejectionSamplerOutput):
        self.verify_context_state.context_len += self.verify_context_state.scheduled_query_len
        if DEBUG:
            print_debug(
                function_name="mid_process_first",
                request_id=self.request_id,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                beta_list=rejection_sampler_output.beta_list,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        num_compute_tokens = self.draft_output_state.num_appended_tokens
        self.verify_output_state.cut_tokens(rejection_sampler_output)
        self.draft_output_state.cut_tokens(rejection_sampler_output)
        
        self.verify_stat.append(
            num_compute_tokens=rejection_sampler_output.num_draft_tokens,
            num_alive_tokens=rejection_sampler_output.num_alive_tokens,
            emitted_token_num=rejection_sampler_output.emitted_token_num,
            beta_list=rejection_sampler_output.beta_list,
            accept_prob_list=rejection_sampler_output.accept_prob_list,
            expected_tokens=rejection_sampler_output.expected_tokens,
            full_expected_tokens=rejection_sampler_output.full_expected_tokens,
        )
        
        if DEBUG:
            print_debug(
                function_name="mid_process_last",
                request_id=self.request_id,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                beta_list=rejection_sampler_output.beta_list,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )

    def use_oracle_context(self):
        self.context_state = self.oracle_context_state


    def setup_oracle_step(self, device: str):
        if self.oracle_output_state is None:
            self.oracle_output_state = RejectionSamplerState(device=device, vocab_size=self.vocab_size)
        else:
            self.oracle_output_state.clear()
    

    def prepare_oracle_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_oracle_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.oracle_context_state.context_len:self.oracle_context_state.scheduled_context_len]
        else:
            input_ids = np.concatenate([self.generated_ids[-1:], self.draft_output_state.token_ids], axis=0)
        position_ids = np.arange(self.oracle_context_state.context_len, self.oracle_context_state.context_len + input_ids.shape[0])

        self.oracle_context_state.scheduled_query_len = input_ids.shape[0]

        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.prepare_oracle_input_ids",
                request_id=self.request_id,
                oracle_input_ids=input_ids,
                oracle_position_ids=position_ids,
                len_oracle_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        return input_ids, position_ids
    
    
    def append_oracle_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)

        if self.is_oracle_prefill_step:
            oracle_gen_ids = generated_ids[prompt_generated_len:]
        else:
            oracle_gen_ids = generated_ids[-(self.verify_output_state.num_appended_tokens + 1):]

        if self.is_prefill_step:
            self.oracle_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()

        self.oracle_output_state.append_tokens(oracle_gen_ids)

    def prepare_oracle_rejection_tensors(self) -> Tuple[np.ndarray]:
        return self.verify_output_state.token_ids

    def cut_speculative_outputs(
        self,
        rejection_sampler_output: RejectionSamplerOutput
    ):
        num_compute_tokens = self.oracle_output_state.num_appended_tokens
        self.oracle_output_state.cut_tokens(rejection_sampler_output)

        self.oracle_stat.append(
            num_compute_tokens=rejection_sampler_output.num_draft_tokens,
            num_alive_tokens=rejection_sampler_output.num_alive_tokens,
            emitted_token_num=rejection_sampler_output.emitted_token_num,
            beta_list=rejection_sampler_output.beta_list,
            accept_prob_list=rejection_sampler_output.accept_prob_list,
        )
        return rejection_sampler_output

    def hierarchical_speculative_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.use_oracle_context()
        super().append_outputs(rejection_sampler_output.sampled_tokens, self.oracle_output_state.log_probs())
        self.draft_context_state.context_len = min(self.oracle_context_state.context_len, self.draft_context_state.context_len)
        self.verify_context_state.context_len = self.draft_context_state.context_len

        if DEBUG:
            print_debug(
                function_name="hierarchical_speculative_append_outputs",
                request_id=self.request_id,
                oracle_output_state_token_ids=self.oracle_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
            
        if DEBUG:
            print_debug(
                function_name="hierarchical_speculative_append_outputs",
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )

    
    def convert_to_request_output(self):
        return HierarchicalSpeculativeRequestOutput(
            request_id=self.request_id,
            prompt_text=self.prompt_text,
            generated_text=self.generated_text,
            generated_ids=self.generated_ids,
            prompt_len=self.prompt_len,
            generated_len=self.generated_len,
            queue_wait_time=self.queue_wait_time,
            prefill_latency=self.prefill_latency,
            generation_latency=self.generation_latency,
            overall_latency=self.overall_latency,
            prompt_log_probs=self.prompt_log_probs,
            generated_log_probs=self.generated_log_probs,
            num_draft_steps=self.num_draft_steps,
            num_verify_steps=self.num_verify_steps,
            verify_stat=self.verify_stat,
            oracle_stat=self.oracle_stat,
        )
    



class LayerSkipRequest(SpeculativeRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.num_generation_tokens = self.num_draft_steps + 1
        self.use_verify_context()
        # self.oracle_context_state.num_generation_tokens = self.num_draft_steps
        # self.verify_context_state.num_generation_tokens = self.num_draft_steps
        # self.use_oracle_context()

    # 모델 forward 전 input 준비
    def prepare_draft_input_ids(self):
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
                position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
            else:
                # print("----------------------여긴가??-------------------")
                process_token_len = self.verify_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
                position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
            if self.generation_start_time is None:
                self.generation_start_time = time.time()

        assert self.generation_start_time is not None

        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_draft_input_ids",
                request_id=self.request_id,
                draft_input_ids=input_ids,
                draft_position_ids=position_ids,
                len_draft_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                process_token_len=process_token_len,
            )
        
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]

        return input_ids, position_ids


    # 모델 output 확률값 / 현재 seq len 업데이트
    def append_draft_outputs(
            self,
            generated_ids: np.ndarray
        ):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_idx = self.get_prompt_generated_len(generated_ids.shape[0], self.remain_draft_prompt_len, self.is_draft_prefill_step)
        
        if self.is_draft_prefill_step:
            draft_generated_ids = generated_ids[prompt_generated_idx:]
        else:
            draft_generated_ids = generated_ids[-1:]
        self.draft_output_state.append_tokens(draft_generated_ids)
        self.draft_context_state.context_len += self.draft_context_state.scheduled_query_len
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.append_draft_outputs",
                self_is_draft_prefill_step=self.is_draft_prefill_step,
                generated_ids=generated_ids,
                draft_generated_ids=self.draft_output_state.token_ids,
                draft_len_generated=self.draft_output_state.num_appended_tokens,
            )
        
    # 모델 forward 전 input 준비
    def prepare_verify_input_ids(self):
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.verify_context_state.context_len:self.verify_context_state.scheduled_context_len]
        else:
            input_ids = np.concatenate([self.generated_ids[-1:], self.draft_output_state.token_ids], axis=0)
        
        position_ids = np.arange(self.verify_context_state.context_len, self.verify_context_state.context_len + input_ids.shape[0])

        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_verify_input_ids",
                request_id=self.request_id,
                verify_input_ids=input_ids,
                verify_position_ids=position_ids,
                len_verify_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )
        self.verify_context_state.scheduled_query_len = input_ids.shape[0]
    
        return input_ids, position_ids
    
    # 모델 output 확률값 / 현재 seq len 업데이트
    def append_verify_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)

        if self.is_prefill_step:
            verify_generated_ids = generated_ids[prompt_generated_len:]
        else:
            verify_generated_ids = generated_ids[-(self.draft_output_state.num_appended_tokens + 1):]

        if self.is_prefill_step:
            self.verify_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()
        self.verify_output_state.append_tokens(verify_generated_ids)
        self.draft_context_state.context_len += 1
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.append_verify_outputs",
                generated_ids=generated_ids,
                self_verify_generated_ids=self.verify_output_state.token_ids,
                self_prompt_len=self.prompt_len,
            )

    # append_draft_outputs, append_verify_outputs 가지고 rejection 준비
    def prepare_verify_rejection_tensors(self) -> Tuple[np.ndarray]:
        return self.draft_output_state.token_ids

    def layerskip_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        if DEBUG:
            print_debug(
                function_name="speculative_append_outputs_before_cut",
                request_id=self.request_id,
                verify_output_state_token_ids=self.verify_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )

        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.use_verify_context()
        super().append_outputs(rejection_sampler_output.sampled_tokens, self.verify_output_state.log_probs())
        self.draft_context_state.context_len = min(self.verify_context_state.context_len, self.draft_context_state.context_len)
        
        
        
class HierarchicalLayerSkipRequest(HierarchicalSpeculativeRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_draft_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
            else:
                process_token_len = self.oracle_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
        
            if self.generation_start_time is None:
                self.generation_start_time = time.time()


        position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.prepare_draft_input_ids",
                request_id=self.request_id,
                self_is_prefill_step=self.is_prefill_step,
                draft_input_ids=input_ids,
                draft_position_ids=position_ids,
                len_draft_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
                process_token_len=process_token_len,
            )
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids
    

    def append_draft_outputs(
            self,
            generated_ids: np.ndarray
        ):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_idx = self.get_prompt_generated_len(generated_ids.shape[0], self.remain_draft_prompt_len, self.is_draft_prefill_step)
        
        if self.is_draft_prefill_step:
            draft_generated_ids = generated_ids[prompt_generated_idx:]
        else:
            draft_generated_ids = generated_ids[-1:]

        self.draft_output_state.append_tokens(draft_generated_ids)
        self.draft_context_state.context_len += self.draft_context_state.scheduled_query_len
        
        if DEBUG:
            print_debug(
                function_name="SpeculativeRequest.append_draft_outputs",
                self_is_draft_prefill_step=self.is_draft_prefill_step,
                generated_ids=generated_ids,
                draft_generated_ids=self.draft_output_state.token_ids,
                draft_len_generated=self.draft_output_state.num_appended_tokens,
            )
            
    def prepare_verify_input_ids(self, draft_step) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.verify_context_state.context_len:self.verify_context_state.scheduled_context_len]
        else:
            process_token_len = self.oracle_context_state.context_len - self.verify_context_state.context_len + 1
            input_ids = np.concatenate([self.generated_ids[-process_token_len:], self.draft_output_state.token_ids[-draft_step:]], axis=0)
        position_ids = np.arange(self.verify_context_state.context_len, self.verify_context_state.context_len + input_ids.shape[0])

        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.prepare_verify_input_ids",
                request_id=self.request_id,
                verify_input_ids=input_ids,
                verify_position_ids=position_ids,
                len_verify_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        self.verify_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids
    # def prepare_verify_input_ids(self):
    #     if self.is_prefill_step:
    #         if self.prefill_start_time is None:
    #             self.prefill_start_time = time.time()
    #         input_ids = self.prompt_ids[self.verify_context_state.context_len:self.verify_context_state.scheduled_context_len]
    #     else:
    #         input_ids = np.concatenate([self.generated_ids[-1:], self.draft_output_state.token_ids], axis=0)
        
    #     position_ids = np.arange(self.verify_context_state.context_len, self.verify_context_state.context_len + input_ids.shape[0])

    #     if DEBUG:
    #         print_debug(
    #             function_name="SpeculativeRequest.prepare_verify_input_ids",
    #             request_id=self.request_id,
    #             verify_input_ids=input_ids,
    #             verify_position_ids=position_ids,
    #             len_verify_input_ids=input_ids.shape[0],
    #             draft_context_len=self.draft_context_state.context_len,
    #             verify_context_len=self.verify_context_state.context_len,
    #         )
    #     self.verify_context_state.scheduled_query_len = input_ids.shape[0]
    #     import pdb; pdb.set_trace()
    #     return input_ids, position_ids
    
    def append_verify_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)
        # import pdb; pdb.set_trace()
        if self.is_prefill_step:
            verify_generated_ids = generated_ids[prompt_generated_len:]
        else:
            # verify_generated_ids = generated_ids[-(self.draft_output_state.num_appended_tokens):]
            verify_generated_ids = generated_ids[:self.draft_output_state.num_appended_tokens]

        if self.is_prefill_step:
            self.verify_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()
                
        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.append_verify_outputs",
                generated_ids=generated_ids,
                self_verify_generated_ids=self.verify_output_state.token_ids,
                self_prompt_len=self.prompt_len,
                self_vocab_size=self.vocab_size,
            )
        self.verify_output_state.append_tokens(verify_generated_ids)

    def mid_process(self, rejection_sampler_output: RejectionSamplerOutput):
        self.verify_context_state.context_len += self.verify_context_state.scheduled_query_len

        if DEBUG:
            print_debug(
                function_name="mid_process_first",
                request_id=self.request_id,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                beta_list=rejection_sampler_output.beta_list,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        num_compute_tokens = self.draft_output_state.num_appended_tokens
        self.verify_output_state.cut_tokens(rejection_sampler_output)
        self.draft_output_state.cut_tokens(rejection_sampler_output)
        
        self.verify_stat.append(
            num_compute_tokens=rejection_sampler_output.num_draft_tokens,
            num_alive_tokens=rejection_sampler_output.num_alive_tokens,
            emitted_token_num=rejection_sampler_output.emitted_token_num,
            beta_list=rejection_sampler_output.beta_list,
            accept_prob_list=rejection_sampler_output.accept_prob_list,
        )
        
        if DEBUG:
            print_debug(
                function_name="mid_process_last",
                request_id=self.request_id,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                beta_list=rejection_sampler_output.beta_list,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )

    def prepare_oracle_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_oracle_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.oracle_context_state.context_len:self.oracle_context_state.scheduled_context_len]
        else:
            input_ids = np.concatenate([self.generated_ids[-1:], self.draft_output_state.token_ids], axis=0)
        position_ids = np.arange(self.oracle_context_state.context_len, self.oracle_context_state.context_len + input_ids.shape[0])

        self.oracle_context_state.scheduled_query_len = input_ids.shape[0]

        if DEBUG:
            print_debug(
                function_name="HierarchicalSpeculativeRequest.prepare_oracle_input_ids",
                request_id=self.request_id,
                oracle_input_ids=input_ids,
                oracle_position_ids=position_ids,
                len_oracle_input_ids=input_ids.shape[0],
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        return input_ids, position_ids
    
    
    def append_oracle_outputs(self, generated_ids: np.ndarray):
        generated_ids_len = generated_ids.shape[0]
        prompt_generated_len = self.get_prompt_generated_len(generated_ids_len, self.remain_prompt_len, self.is_prefill_step)

        if self.is_oracle_prefill_step:
            oracle_gen_ids = generated_ids[prompt_generated_len:]
        else:
            oracle_gen_ids = generated_ids[-(self.verify_output_state.num_appended_tokens + 1):]

        if self.is_prefill_step:
            self.oracle_context_state.context_len += (prompt_generated_len)
            if not self.is_prefill_step:
                self.prefill_end_time = time.time()

        self.oracle_output_state.append_tokens(oracle_gen_ids)
        
    def hierarchical_speculative_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        
        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.use_oracle_context()
        super().append_outputs(rejection_sampler_output.sampled_tokens, self.oracle_output_state.log_probs())
        self.draft_context_state.context_len = min(self.oracle_context_state.context_len, self.draft_context_state.context_len)
        self.verify_context_state.context_len = self.draft_context_state.context_len

        if DEBUG:
            print_debug(
                function_name="hierarchical_self_speculative_append_outputs",
                request_id=self.request_id,
                oracle_output_state_token_ids=self.oracle_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
            
        if DEBUG:
            print_debug(
                function_name="hierarchical_self_speculative_append_outputs",
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )

class SmartSpecRequest(SpeculativeRequest):
    def __init__(self, *args, **kwargs):
        self.moving_average_window = kwargs.pop('moving_average_window', 20) # I know this is not good
        super().__init__(*args, **kwargs)
        self.acceptance_rates = []
        self.original_draft_length = self.num_draft_steps
        
    def update_acceptance_rate(self, accepted_tokens: int):
        current_rate = accepted_tokens / self.num_draft_steps
        self.acceptance_rates.append(current_rate)
        if len(self.acceptance_rates) > self.moving_average_window:
            self.acceptance_rates = self.acceptance_rates[-self.moving_average_window:]
    
    def get_acceptance_rate(self) -> float:
        if len(self.acceptance_rates) == 0:
            return 1.0
        return round(sum(self.acceptance_rates) / len(self.acceptance_rates), 4)
    
    def get_draft_length(self) -> int:
        return self.num_draft_steps

    def should_cut_draft(self) -> bool:
        return self.get_draft_length() < self.original_draft_length
    
    def mid_process(self):
        self.draft_output_state.cut_tokens_by_length(self.get_draft_length())
        
        
class SVIPRequest(SpeculativeRequest):
    def __init__(self, *args, **kwargs):
        self.threshold_for_svip = kwargs.pop('threshold_for_svip', 0.5)
        super().__init__(*args, **kwargs)
        self.original_draft_length = self.num_draft_steps
        self.should_stop_drafting = False
    
    def setup_speculative_step(self, device: str):
        self.should_stop_drafting = False
        super().setup_speculative_step(device)

    def get_draft_length(self, ) -> int:
        return self.num_draft_steps

    def should_cut_draft(self) -> bool:
        return self.get_draft_length() < self.original_draft_length
    
    def update_request(self, should_stop_drafting: bool, entropy_flags: bool, num_draft_steps: int):
        self.should_stop_drafting = should_stop_drafting
        if self.should_stop_drafting:
            self.num_draft_steps = num_draft_steps + 1
            if not entropy_flags:
                self.num_generation_tokens = num_draft_steps + 2
    
    def mid_process(self):
        self.draft_output_state.cut_tokens_by_length(self.get_draft_length())
        

class Eagle3Request(SpeculativeRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_states = None
    
    @property
    def is_draft_prefill_step(self):
        return self.draft_context_state.context_len < self.prompt_len

    @property
    def is_verify_prefill_step(self):
        return self.verify_context_state.context_len < self.prompt_len

    def set_scheduled_query_len(
            self,
            num_prompt_tokens_limit: Optional[int] = None):
        
        if self.is_draft_prefill_step:
            self.use_draft_context()
            allocated_query_len = self.remain_prompt_len + self.num_generation_tokens - 1
        elif self.is_verify_prefill_step:
            self.use_verify_context()
            allocated_query_len = self.remain_prompt_len + self.num_generation_tokens - 1
        else:
            allocated_query_len = self.num_generation_tokens

        if num_prompt_tokens_limit is not None:
            allocated_query_len = np.minimum(allocated_query_len, num_prompt_tokens_limit)
        
        self.allocated_query_len = allocated_query_len
        self.draft_context_state.scheduled_query_len = allocated_query_len
        self.verify_context_state.scheduled_query_len = allocated_query_len
    
    def prepare_draft_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
            input_ids = input_ids[1:]
            if self.draft_context_state.scheduled_context_len >= self.prompt_len:
                input_ids = np.concatenate([input_ids, self.generated_ids[-1:]])
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
            else:
                process_token_len = self.verify_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
            
            if self.generation_start_time is None:
                self.generation_start_time = time.time()

        position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
        
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids
    
    def append_hidden_states(self, hidden_state: torch.Tensor):
        if self.hidden_states is None:
            self.hidden_states = hidden_state
        else:
            self.hidden_states = torch.cat([self.hidden_states, hidden_state], dim=0)
    
    def cut_hidden_states(self, num_emitted_tokens: int):
        self.hidden_states = self.hidden_states[:num_emitted_tokens+1]
    
    def append_draft_outputs(self, generated_ids: np.ndarray):
        draft_generated_ids = generated_ids[-1:]

        self.draft_output_state.append_tokens(draft_generated_ids)
        self.draft_context_state.context_len += self.draft_context_state.scheduled_query_len
        if self.is_draft_prefill_step:
            self.draft_context_state.scheduled_query_len = self.prompt_len - self.draft_context_state.context_len

    def prepare_draft_hidden_states(self, prompt_len: int) -> torch.Tensor:
        if self.is_draft_prefill_step:
            from_idx = self.draft_context_state.context_len
            to_idx = self.draft_context_state.context_len+prompt_len
            hidden_states = self.hidden_states[from_idx:to_idx]
        else:
            hidden_states = self.hidden_states[-prompt_len:]
        return hidden_states

    def speculative_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        if DEBUG:
            print_debug(
                function_name="speculative_append_outputs_before_cut",
                request_id=self.request_id,
                verify_output_state_token_ids=self.verify_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
            )
        
        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.cut_hidden_states(rejection_sampler_output.emitted_token_num)
        self.use_verify_context()
        finished = super().append_outputs(rejection_sampler_output.sampled_tokens, self.verify_output_state.log_probs())
        self.draft_context_state.context_len = min(self.verify_context_state.context_len, self.draft_context_state.context_len)
        if finished:
            self.hidden_state = None



class Eagle3SVRequest(HierarchicalSpeculativeRequest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_states = None
    
    @property
    def is_draft_prefill_step(self):
        return self.draft_context_state.context_len < self.prompt_len

    @property
    def is_verify_prefill_step(self):
        return self.verify_context_state.context_len < self.prompt_len

    @property
    def is_oracle_prefill_step(self):
        return self.oracle_context_state.context_len < self.prompt_len

    def set_scheduled_query_len(
            self,
            num_prompt_tokens_limit: Optional[int] = None):
        
        if self.is_draft_prefill_step:
            self.use_draft_context()
            allocated_query_len = self.remain_prompt_len + self.num_generation_tokens - 1
        elif self.is_verify_prefill_step:
            self.use_verify_context()
            allocated_query_len = self.remain_prompt_len + self.num_generation_tokens - 1
        else:
            allocated_query_len = self.num_generation_tokens

        if num_prompt_tokens_limit is not None:
            allocated_query_len = np.minimum(allocated_query_len, num_prompt_tokens_limit)
        
        self.allocated_query_len = allocated_query_len
        self.draft_context_state.scheduled_query_len = allocated_query_len
        self.oracle_context_state.scheduled_query_len = allocated_query_len
    
    def prepare_draft_input_ids(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_prefill_step:
            if self.prefill_start_time is None:
                self.prefill_start_time = time.time()
            input_ids = self.prompt_ids[self.draft_context_state.context_len:self.draft_context_state.scheduled_context_len]
            input_ids = input_ids[1:]
            if self.draft_context_state.scheduled_context_len >= self.prompt_len:
                input_ids = np.concatenate([input_ids, self.generated_ids[-1:]])
        else:
            if self.draft_output_state.num_appended_tokens > 0:
                process_token_len = 1
                input_ids = self.draft_output_state.token_ids[-process_token_len:]
            else:
                process_token_len = self.oracle_context_state.context_len - self.draft_context_state.context_len + 1
                input_ids = self.generated_ids[-process_token_len:]
            
            if self.generation_start_time is None:
                self.generation_start_time = time.time()

        position_ids = np.arange(self.draft_context_state.context_len, self.draft_context_state.context_len + input_ids.shape[0])
        
        self.draft_context_state.scheduled_query_len = input_ids.shape[0]
        return input_ids, position_ids
    
    def append_hidden_states(self, hidden_state: torch.Tensor):
        if self.hidden_states is None:
            self.hidden_states = hidden_state
        else:
            self.hidden_states = torch.cat([self.hidden_states, hidden_state], dim=0)
    
    def cut_hidden_states(self, num_emitted_tokens: int):
        self.hidden_states = self.hidden_states[:num_emitted_tokens+1]
    
    def append_draft_outputs(self, generated_ids: np.ndarray):
        draft_generated_ids = generated_ids[-1:]

        self.draft_output_state.append_tokens(draft_generated_ids)
        self.draft_context_state.context_len += self.draft_context_state.scheduled_query_len
        if self.is_draft_prefill_step:
            self.draft_context_state.scheduled_query_len = self.prompt_len - self.draft_context_state.context_len

    def prepare_draft_hidden_states(self, prompt_len: int) -> torch.Tensor:
        if self.is_draft_prefill_step:
            from_idx = self.draft_context_state.context_len
            to_idx = self.draft_context_state.context_len+prompt_len
            hidden_states = self.hidden_states[from_idx:to_idx]
        else:
            hidden_states = self.hidden_states[-prompt_len:]
        return hidden_states
    
    def hierarchical_speculative_append_outputs(
            self,
            rejection_sampler_output: RejectionSamplerOutput
        ):
        rejection_sampler_output = self.cut_speculative_outputs(rejection_sampler_output)
        self.cut_hidden_states(rejection_sampler_output.emitted_token_num)
        self.use_oracle_context()
        finished = super().append_outputs(rejection_sampler_output.sampled_tokens, self.verify_output_state.log_probs())
        self.draft_context_state.context_len = min(self.oracle_context_state.context_len, self.draft_context_state.context_len)
        self.verify_context_state.context_len = self.draft_context_state.context_len
        
        if finished:
            self.hidden_state = None
        
        if DEBUG:
            print_debug(
                function_name="hierarchical_speculative_append_outputs",
                request_id=self.request_id,
                oracle_output_state_token_ids=self.oracle_output_state.token_ids,
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
            
        if DEBUG:
            print_debug(
                function_name="hierarchical_speculative_append_outputs",
                sampled_tokens=rejection_sampler_output.sampled_tokens,
                alive_token_num=rejection_sampler_output.num_alive_tokens,
                emitted_token_num=rejection_sampler_output.emitted_token_num,
                failed_token_num=rejection_sampler_output.failed_token_num,
                draft_context_len=self.draft_context_state.context_len,
                verify_context_len=self.verify_context_state.context_len,
                oracle_context_len=self.oracle_context_state.context_len,
            )
        