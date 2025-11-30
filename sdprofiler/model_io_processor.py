from typing import List, Optional, Tuple, Union, Dict, Callable
from collections import deque
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


from .request import Request, FlashInferMetadata, SmartSpecRequest, SVIPRequest, Eagle3Request
from .registry import EngineRegistry
from .sampler import Sampler, RejectionSampler
from .utils.common import profile_nvtx, print_debug
from .utils import ops
import ray.util.collective as collective
from .utils.ops.sync_probs import sync_probs_df, exist_method

DEBUG = False

class ModelIOProcessor:
    def __init__(
            self,
            registry: EngineRegistry,
        ):
        self.registry = registry

        self.global_rank = 0 # get_global_rank(worker_idx)
        self.local_rank = 0 # get_local_rank(worker_idx)
        self.device = f'cuda:{self.local_rank}'

        self.hf_config = self.registry.get('val.model.hf_config')
        self.sampler = Sampler(self.hf_config.vocab_size).eval()

        self.past_key_values = None
        
        self.num_kv_cache_blocks = self.registry.get('val.engine.num_kv_cache_blocks')
        self.kv_cache_block_size = self.registry.get('val.engine.kv_cache_block_size')

        self.eos_token_ids = self.registry.get('val.model.eos_token_ids')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.registry.get('val.model.model_name_or_path'))
        
        self.beta_threshold = self.registry.get('val.engine.beta_threshold')
        self.engine_stat = self.registry.get('val.engine.stat')

    
    def update_registry(self, registry: EngineRegistry):
        self.registry = registry

    @profile_nvtx("ModelIOProcessor._build_flashinfer_model_inputs")
    def _build_flashinfer_model_inputs(
            self,
            input_ids_list: List[np.ndarray],
            position_ids_list: List[np.ndarray],
            paged_kv_indices_list: List[np.ndarray],
            paged_kv_last_page_len_list: List[np.ndarray],
            return_np: bool = False
        ):
        input_ids_nested = np.concatenate(input_ids_list, axis=0)
        position_ids_nested = np.concatenate(position_ids_list, axis=0)

        # lengths = [input_ids.shape[0] for input_ids in input_ids_list]
        # input_idx_start_indices = np.cumsum([0] + lengths, dtype=np.int32)
        # input_idx_length = np.array(lengths, dtype=np.int32)

        input_idx_length = np.array([input_ids.shape[0] for input_ids in input_ids_list], dtype=np.int32)
        input_idx_start_indices = np.concatenate(([0], np.cumsum(input_idx_length)), dtype=np.int32)

        # input_idx_start_indices = np.cumsum([0] + [input_ids.shape[0] for input_ids in input_ids_list], dtype=np.int32)
        # input_idx_length = np.array([input_ids.shape[0] for input_ids in input_ids_list], dtype=np.int32)

        if DEBUG:
            print_debug(
                function_name="ModelIOProcessor._build_flashinfer_model_inputs",
                input_ids_list=[input_ids for input_ids in input_ids_list],
                position_ids_list=[position_ids for position_ids in position_ids_list],
                input_ids_nested=input_ids_nested.shape,
                position_ids_nested=position_ids_nested.shape,
                input_idx_start_indices=input_idx_start_indices.shape,
                input_idx_length=input_idx_length,
            )
        paged_kv_indptr = np.cumsum([0] + [paged_kv_indices.shape[0] for paged_kv_indices in paged_kv_indices_list], dtype=np.int32)
        paged_kv_indices = np.concatenate(paged_kv_indices_list, axis=0, dtype=np.int32)
        paged_kv_last_page_len = np.concatenate(paged_kv_last_page_len_list, axis=0, dtype=np.int32)

        if return_np:
            return input_ids_nested, position_ids_nested, FlashInferMetadata(
                input_ids_indptr=input_idx_start_indices,
                input_ids_lengths=input_idx_length,
                batch_indices=None,
                positions=None,
                paged_kv_indices=paged_kv_indices,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_last_page_len=paged_kv_last_page_len,
                kv_cache_block_size=self.kv_cache_block_size
            )
        
        input_ids_tensor = torch.from_numpy(input_ids_nested)
        position_ids_tensor = torch.from_numpy(position_ids_nested)
        input_ids_indptr = torch.from_numpy(input_idx_start_indices)
        input_ids_lengths = torch.from_numpy(input_idx_length)

        paged_kv_indices = torch.from_numpy(paged_kv_indices)
        paged_kv_indptr = torch.from_numpy(paged_kv_indptr)
        paged_kv_last_page_len = torch.from_numpy(paged_kv_last_page_len)

        # return input_ids_tensor, position_ids_tensor, FlashInferMetadata(
        #     input_ids_indptr=input_ids_indptr,
        #     input_ids_lengths=input_ids_lengths,
        #     batch_indices=None,
        #     positions=None,
        #     paged_kv_indices=paged_kv_indices,
        #     paged_kv_indptr=paged_kv_indptr,
        #     paged_kv_last_page_len=paged_kv_last_page_len,
        #     kv_cache_block_size=self.kv_cache_block_size
        # )
        flashinfer_metadata = FlashInferMetadata(
            input_ids_indptr=input_ids_indptr,
            input_ids_lengths=input_ids_lengths,
            batch_indices=None,
            positions=None,
            paged_kv_indices=paged_kv_indices,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_last_page_len=paged_kv_last_page_len,
            kv_cache_block_size=self.kv_cache_block_size
        )
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    
    
    @profile_nvtx("ModelIOProcessor.build_input_tensors")
    def build_input_tensors(self, requests: List[Request], return_np=False):
        if len(requests) == 0:
            return [], [], FlashInferMetadata.create_zero_length(self.kv_cache_block_size)
        np_arrays = [request.prepare_input_ids() for request in requests]
        input_ids_list, position_ids_list = zip(*np_arrays)
        np_arrays = [request.prepare_paged_kv_blocks(self.kv_cache_block_size) for request in requests]
        paged_kv_indices_list, paged_kv_last_page_len_list = zip(*np_arrays)

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
    
        if DEBUG:
            print_debug("ModelIOProcessor.build_input_tensors", 
                batch_size=len(requests),
                input_ids_tensor=input_ids_tensor.shape,
                position_ids_tensor=position_ids_tensor.shape,
            )
        
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

    
    @profile_nvtx("ModelIOProcessor.check_finish_criteria")
    def check_finish_criteria(self, request: Request):
        is_stop = False

        if request.sampling_params.min_new_tokens is not None:
            if request.generated_len < request.sampling_params.min_new_tokens:
                is_stop = False

        if request.sampling_params.max_new_tokens is not None:
            if request.generated_len >= request.sampling_params.max_new_tokens:
                is_stop = True

        if self.hf_config.max_position_embeddings is not None:
            if request.context_state.context_len >= self.hf_config.max_position_embeddings:
                is_stop = True

        if request.context_state.context_len >= self.registry.get('val.engine.max_context_len'):
            is_stop = True
        
        if not request.sampling_params.disable_eos_token:
            if np.any(np.isin(request.last_generated_ids, self.eos_token_ids)):
                is_stop = True
        
        
        return is_stop

    @profile_nvtx("ModelIOProcessor.post_process_requests")
    def post_process_requests(
            self,
            requests: List[Request],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray
        ):
        
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            log_probs = None
            request.append_outputs(sampled_tokens, log_probs)
        return requests


from .request import SpeculativeRequest, RejectionSamplerOutput
class RejectionProbs:
    def __init__(self, num_max_batch_requests, max_token_len, vocab_size, use_sparse_probs=False):
        self.num_max_batch_requests = num_max_batch_requests
        self.max_token_len = max_token_len
        self.vocab_size = vocab_size
        self.use_sparse_probs = use_sparse_probs
        self.appended_request_idx = 0
        self.appended_token_idx = 0

        if self.use_sparse_probs:
            self.probs_indptr = torch.zeros([num_max_batch_requests, max_token_len, 0], dtype=torch.int32, device='cuda')
            self.probs = torch.zeros([num_max_batch_requests, max_token_len, 0], dtype=torch.float32, device='cuda')
        else:
            self.probs = torch.zeros(num_max_batch_requests, max_token_len, vocab_size, dtype=torch.float32, device='cuda')

        self.sparse_probs = []
        self.base_shape = None

    def reset(self):
        self.appended_request_idx = 0
        self.appended_token_idx = 0
        if self.use_sparse_probs:
            self.probs_indptr = torch.zeros([self.num_max_batch_requests, self.max_token_len, 0], dtype=torch.int32, device='cuda')
            self.probs = torch.zeros([self.num_max_batch_requests, self.max_token_len, 0], dtype=torch.float32, device='cuda')
        else:
            self.probs.zero_()

    def cutoff(self, mask):
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask
        else:
            mask_tensor = torch.tensor(mask, device=self.probs.device)

        padded_mask = torch.zeros((self.probs.shape[0], mask_tensor.shape[1]), dtype=torch.bool, device=self.probs.device)
        padded_mask[:mask_tensor.shape[0], :] = mask_tensor

        if self.use_sparse_probs:
            self.probs_indptr[padded_mask] = -1
            self.probs[padded_mask] = 0
        else:
            self.probs[padded_mask] = 0
    # prob: [batch_size, vocab_size]
    @profile_nvtx("RejectionProbs.multi_request_single_append")
    def multi_request_single_append(self, num_requests, probs):

        if self.use_sparse_probs:
            nonzero_indices = probs.nonzero()
            batch_indices = nonzero_indices[:, 0]
            token_indices = nonzero_indices[:, 1]
            batch_counts = torch.bincount(batch_indices, minlength=num_requests)
            max_nonzero = batch_counts.max().item()
            batch_offsets = torch.arange(max_nonzero, device=batch_indices.device)[None, :] 
            valid_mask = batch_offsets < batch_counts[:, None]
            
            if max_nonzero > self.probs_indptr.shape[-1]:
                pad_size = (0, max_nonzero - self.probs_indptr.shape[-1], 0, 0, 0, 0)
                self.probs_indptr = torch.nn.functional.pad(self.probs_indptr, pad_size, mode='constant', value=-1)
                self.probs = torch.nn.functional.pad(self.probs, pad_size, mode='constant', value=0)

            indices = torch.full_like(valid_mask, -1, dtype=torch.int64, device=probs.device)
            values = torch.zeros_like(valid_mask, dtype=probs.dtype, device=probs.device)
            indices.masked_scatter_(valid_mask, token_indices)
            values.masked_scatter_(valid_mask, probs[batch_indices, token_indices])
            
            self.probs_indptr[:num_requests, self.appended_token_idx, :valid_mask.shape[1]] = indices
            self.probs[:num_requests, self.appended_token_idx, :valid_mask.shape[1]] = values
        else:
            self.probs[:num_requests, self.appended_token_idx, :].copy_(probs)
        self.appended_request_idx = np.maximum(self.appended_request_idx, num_requests-1)
    
    @profile_nvtx("RejectionProbs.multi_request_batch_append")
    def multi_request_batch_append(self, num_requests, probs, probs_input_ids_indptr: Optional[np.ndarray] = None):
        if probs_input_ids_indptr is None: # assume that requests are all of same length
            length = probs.shape[0] // num_requests
            self.probs[:num_requests,:length, :].copy_(probs.view(num_requests, length, -1))
            self.appended_request_idx = np.maximum(self.appended_request_idx, num_requests-1)
            
            self.appended_token_idx = np.maximum(self.appended_token_idx, length-1)
            return
            
        elif self.use_sparse_probs:
            lengths = np.diff(probs_input_ids_indptr)
            nonzero_indices = probs.nonzero()
            batch_counts = torch.bincount(nonzero_indices[:, 0], minlength=num_requests)
            max_nonzero = batch_counts.max().item()
            batch_offsets = torch.arange(max_nonzero, device=nonzero_indices.device)[None, :] 
            valid_mask = batch_offsets < batch_counts[:, None]

            indices = torch.full_like(valid_mask, -1, dtype=torch.int64, device=probs.device)
            values = torch.zeros_like(valid_mask, dtype=probs.dtype, device=probs.device)
            indices.masked_scatter_(valid_mask, nonzero_indices[:, 1])
            values.masked_scatter_(valid_mask, probs[nonzero_indices[:, 0], nonzero_indices[:, 1]])

            self.probs_indptr = torch.full((self.num_max_batch_requests, self.max_token_len, indices.shape[-1]), -1, dtype=torch.int32, device=probs.device)
            self.probs = torch.zeros((self.num_max_batch_requests, self.max_token_len, indices.shape[-1]), dtype=probs.dtype, device=probs.device)

            probs_ids_per_request_delta_tensor = torch.from_numpy(probs_input_ids_indptr[1:] - probs_input_ids_indptr[:-1])
            valid_mask = torch.arange(self.max_token_len) < probs_ids_per_request_delta_tensor.unsqueeze(1)
                
            self.probs_indptr[:num_requests][valid_mask] = indices.to(torch.int32)
            self.probs[:num_requests][valid_mask] = values
            #self.probs_indptr[:num_requests].copy_(indices.to(torch.int32).view(num_requests, length, -1))
            #self.probs[:num_requests].copy_(values.view(num_requests, length, -1))
            
            self.appended_request_idx = np.maximum(self.appended_request_idx, num_requests-1)
            self.appended_token_idx = np.maximum(self.appended_token_idx, np.max(lengths)-1)
        else:
            ops.copy_ragged_probs(
                ragged_probs=probs,
                probs_indptr=torch.from_numpy(probs_input_ids_indptr),
                probs=self.probs
            )
        lengths = np.diff(probs_input_ids_indptr)
        self.appended_request_idx = np.maximum(self.appended_request_idx, num_requests-1)
        self.appended_token_idx = np.maximum(self.appended_token_idx, np.max(lengths)-1)

    def get_probs(self, request_idx: Optional[torch.Tensor] = None):
        if request_idx is None:
            out = self.probs[:self.appended_request_idx+1]
        else:
            out = self.probs.index_select(0, request_idx)

        # print_debug(
        #     function_name="RejectionProbs.get_probs",
        #     request_idx=request_idx,
        #     self_probs_nonzero=self.probs.nonzero(),
        # )
        return out

    def get_probs_from_token_idx(self, token_idx: torch.Tensor):
        token_indices = token_idx.unsqueeze(-1)
        if self.use_sparse_probs:
            out = torch.zeros_like(token_idx, dtype=self.probs.dtype)
            
            probs_indptr = self.probs_indptr[:, :token_idx.shape[1], :]
            
            valid_mask = (probs_indptr != -1)
            batch_indices, seq_indices, pos_indices = valid_mask.nonzero(as_tuple=True)
            token_values = probs_indptr[valid_mask]
            
            matches = (token_values.unsqueeze(-1) == token_indices[batch_indices, seq_indices])
            match_indices = matches.nonzero()[:,0]
            
            out[batch_indices[match_indices], seq_indices[match_indices]] = self.probs[
                batch_indices[match_indices],
                seq_indices[match_indices],
                pos_indices[match_indices]
            ]
        else:
            valid_mask = token_indices != -1
            out = torch.zeros_like(token_indices, dtype=self.probs.dtype, device=self.probs.device)
            if valid_mask.any():
                out[valid_mask] = self.probs.gather(2, token_indices.masked_fill(~valid_mask, 0))[valid_mask]
                out = out.squeeze(-1)
        return out

    def get_index_from_token_idx(self, token_indices: torch.Tensor):
        if self.use_sparse_probs:
            nonzero_batch_indices = self.probs.sum(dim=(1,2)).nonzero()[:, 0]
            probs_indptr = self.probs_indptr[nonzero_batch_indices]

            out = torch.full_like(token_indices, -1, dtype=torch.int64, device=token_indices.device)
            valid_tokens = token_indices != -1
            batch_indices, seq_indices = valid_tokens.nonzero(as_tuple=True)
            matches = probs_indptr[batch_indices, seq_indices] == token_indices[batch_indices, seq_indices].unsqueeze(-1)
            match_indices = matches.nonzero()[:, 1]
            out[batch_indices, seq_indices] = match_indices
        else:
            out = token_indices
        return out

    def get_token_idx_from_index(self, indices: torch.Tensor):
        if self.use_sparse_probs:
            nonzero_batch_indices = self.probs.sum(dim=(1,2)).nonzero()[:, 0]
            probs_indptr = self.probs_indptr[nonzero_batch_indices]

            out = torch.full_like(indices, -1, dtype=torch.int32, device=indices.device)
            valid_mask = indices != -1
            batch_indices, seq_indices = valid_mask.nonzero(as_tuple=True)
            out[batch_indices, seq_indices] = probs_indptr[batch_indices, seq_indices, indices[batch_indices, seq_indices]]
        else:
            out = indices
        return out


class SpeculativeModelIOProcessor(ModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None,  **kwargs):
        super().__init__(*args, **kwargs)
        self.rejection_sampler = RejectionSampler()
        self.vocab_size = self.registry.get('val.engine.vocab_size')
        self.num_max_batch_requests = self.registry.get('val.engine.num_max_batch_requests')
        self.num_draft_steps = self.registry.get('val.engine.num_draft_steps')
        self.use_sparse_probs = self.registry.get('val.engine.use_sparse_probs')
        self.local_rank = local_rank if local_rank is not None else 0
        
        self.draft_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )

        self.verify_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps + 1,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )

    @profile_nvtx("SpeculativeModelIOProcessor.sync_rejection_probs")
    def sync_rejection_probs(
        self,
        rejection_probs1: RejectionProbs,
        rejection_probs2: RejectionProbs
    ):

        if not self.use_sparse_probs:
            return rejection_probs1.get_probs(), rejection_probs2.get_probs()
        
        dp, dpi, vp, vpi = sync_probs_df(
            rejection_probs1.probs, rejection_probs2.probs,
            rejection_probs1.probs_indptr, rejection_probs2.probs_indptr
        )

        rejection_probs1.probs = dp
        rejection_probs2.probs = vp
        rejection_probs1.probs_indptr = dpi
        rejection_probs2.probs_indptr = vpi

        return rejection_probs1.probs, rejection_probs2.probs


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.get_beta_list")
    def get_beta_tensor(
        self,
        draft_probs: torch.Tensor,
        verify_probs: torch.Tensor,
        num_requests: Optional[int] = None
    ) -> np.ndarray:
        # draft_probs, verify_probs = self.sync_rejection_probs(self.draft_rejection_probs, self.verify_rejection_probs)
        if num_requests is None:
            return ops.calculate_beta_torch(draft_probs, verify_probs)
        return ops.calculate_beta_torch(draft_probs[:num_requests], verify_probs[:num_requests])
    

        
    @profile_nvtx("SpeculativeModelIOProcessor.setup_speculative_step")
    def setup_speculative_step(self, requests: List[SpeculativeRequest]):
        self.reset_rejection_probs()
        for request in requests:
            request.setup_speculative_step(device=self.device)

    @profile_nvtx("SpeculativeModelIOProcessor.build_draft_input_tensors")
    def build_draft_input_tensors(self, requests: List[SpeculativeRequest]):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        for i, request in enumerate(requests):
            request.use_draft_context()
            input_ids, position_ids = request.prepare_draft_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        
        if DEBUG:
            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            for idx in range(len(requests)):
                print_debug(
                    function_name="SpeculativeModelIOProcessor.build_draft_input_tensors",
                    input_ids=input_ids_list[idx],
                    position_ids=position_ids_list[idx],
                    string=self.tokenizer.decode(input_ids_list[idx]),
                    paged_kv_indices=paged_kv_indices_list[idx],
                    paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
                )
            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")

        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

    
    @profile_nvtx("SpeculativeModelIOProcessor.append_draft_outputs")
    def append_draft_outputs(
            self,
            requests: List[SpeculativeRequest],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray
        ):
        # print("---------------------------append_draft_outputs---------------------------")
        
        # print_debug(
        #     function_name="SpeculativeModelIOProcessor.append_draft_outputs",
        #     sampled_tokens_per_request=sampled_tokens_per_request,
        #     probs_nonzero=probs.nonzero(),
        #     probs_input_ids_indptr=probs_input_ids_indptr,
        # )
        
        self.draft_rejection_probs.multi_request_single_append(len(requests), probs)
        for request_idx, (request, sampled_tokens) in enumerate(zip(requests, sampled_tokens_per_request)):
            request.append_draft_outputs(sampled_tokens)
        self.draft_rejection_probs.appended_token_idx += 1
        return requests
    

    @profile_nvtx("SpeculativeModelIOProcessor.build_verify_input_tensors")
    def build_verify_input_tensors(
            self,
            requests: List[SpeculativeRequest]
        ):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        for i, request in enumerate(requests):
            request.use_verify_context()

            input_ids, position_ids = request.prepare_verify_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        if DEBUG:
            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            for idx in range(len(requests)):
                print_debug(
                    function_name="SpeculativeModelIOProcessor.build_verify_input_tensors",
                    input_ids=input_ids_list[idx],
                    position_ids=position_ids_list[idx],
                    string=self.tokenizer.decode(input_ids_list[idx]),
                    paged_kv_indices=paged_kv_indices_list[idx],
                    paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
                    flashinfer_metadata=flashinfer_metadata,
                )
            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    

    
    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.append_verify_outputs")
    def append_verify_outputs(
        self,
        requests: List[SpeculativeRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs: torch.Tensor,
        probs_input_ids_indptr: np.ndarray
    ):

        # print_debug(
        #     function_name="SpeculativeModelIOProcessor.append_verify_outputs",
        #     sampled_tokens_per_request=sampled_tokens_per_request,
        #     probs_nonzero=probs.nonzero(),
        #     probs_input_ids_indptr=probs_input_ids_indptr,
        # )
        if self.use_sparse_probs:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs, probs_input_ids_indptr)
        else:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs)
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            request.append_verify_outputs(sampled_tokens)
        return requests
    

    def _construct_rejection_sampler_output(
        self,
        batch_token_ids,
        batch_num_draft_tokens,
        batch_emitted_token_num,
        batch_alive_token_num,
        beta_list_per_request: Optional[np.ndarray] = None,
        accept_prob_list_per_request: Optional[np.ndarray] = None,
        batch_expected_tokens: Optional[np.ndarray] = None,
        batch_full_expected_tokens: Optional[np.ndarray] = None
    ):
        num_requests = batch_token_ids.shape[0]
        
        rejection_sampler_outputs = []
        for idx in range(num_requests):
            num_alive_tokens = batch_alive_token_num[idx]
            sampled_tokens = batch_token_ids[idx][:num_alive_tokens]

            rejection_sampler_output = RejectionSamplerOutput(
                sampled_tokens=sampled_tokens,
                emitted_token_num=batch_emitted_token_num[idx],
                num_draft_tokens=batch_num_draft_tokens[idx],
                num_alive_tokens=num_alive_tokens,
                beta_list=beta_list_per_request[idx] if beta_list_per_request is not None else None,
                accept_prob_list=accept_prob_list_per_request[idx] if accept_prob_list_per_request is not None else None,
                expected_tokens=batch_expected_tokens[idx] if batch_expected_tokens is not None else None,
                full_expected_tokens=batch_full_expected_tokens[idx] if batch_full_expected_tokens is not None else None,
            )
            rejection_sampler_outputs.append(rejection_sampler_output)

        return rejection_sampler_outputs

    def _run_rejection_sampler(
        self,
        draft_token_ids_list,
        draft_probs_tensor,
        verify_probs_tensor,
        return_log_probs: bool = False,
        return_beta: bool = False,
        return_accept_prob: bool = False
    ):
        draft_token_ids_list = [torch.from_numpy(draft_token_ids) for draft_token_ids in draft_token_ids_list]
        draft_token_ids_tensor = pad_sequence(draft_token_ids_list, batch_first=True, padding_value=-1).to(self.device)
        draft_token_ids = self.draft_rejection_probs.get_index_from_token_idx(draft_token_ids_tensor)

        # draft_token_ids_tensor_np = np.concatenate(draft_token_ids_tensor.cpu().numpy(), axis=0)
        # nonzero_indices = draft_probs_tensor.nonzero()[:,-1].cpu().numpy()
        # if np.any(draft_token_ids_tensor_np != nonzero_indices):
        #     print(f"    Mismatch found between draft_token_ids and nonzero_indices")
        #     print(f"    draft_token_ids shape: {draft_token_ids.shape}")
        #     print(f"    draft_token_ids_tensor_np shape: {draft_token_ids_tensor_np.shape}")
        #     print(f"    nonzero_indices shape: {nonzero_indices.shape}")
        #     print(f"    draft_token_ids_tensor_np: {draft_token_ids_tensor_np}")
        #     print(f"    nonzero_indices: {nonzero_indices}")

        #     import pdb; pdb.set_trace()
        
        batch_token_ids, batch_accepted_token_num, batch_emitted_token_num = self.rejection_sampler.forward(
            draft_token_ids,
            draft_probs_tensor,
            verify_probs_tensor
        )

        batch_token_ids = self.draft_rejection_probs.get_token_idx_from_index(batch_token_ids)

        batch_token_ids = batch_token_ids.cpu().numpy()
        batch_num_draft_tokens = [draft_token_ids.shape[0] for draft_token_ids in draft_token_ids_list]
        batch_emitted_token_num = batch_emitted_token_num.cpu().numpy()
        batch_alive_token_num = batch_emitted_token_num + 1

        beta_list_per_request = self.get_beta_tensor(draft_probs_tensor, verify_probs_tensor).cpu().numpy() if return_beta else None
        
        if return_accept_prob:
            accept_prob_list_per_request = self.get_accept_prob(draft_token_ids_tensor, self.draft_rejection_probs, self.verify_rejection_probs)
        else:
            accept_prob_list_per_request = None
        rejection_sampler_outputs = self._construct_rejection_sampler_output(
            batch_token_ids=batch_token_ids,
            batch_num_draft_tokens=batch_num_draft_tokens,
            batch_emitted_token_num=batch_emitted_token_num,
            batch_alive_token_num=batch_alive_token_num,
            beta_list_per_request=beta_list_per_request,
            accept_prob_list_per_request=accept_prob_list_per_request,
        )
        return rejection_sampler_outputs
    

    @profile_nvtx("SpeculativeModelIOProcessor.run_reject_sampler")
    def run_rejection_sampler(
            self, 
            requests: List[SpeculativeRequest],
            return_log_probs: bool = False,
            return_beta: bool = False,
            return_accept_prob: bool = False,
        ) -> List[RejectionSamplerOutput]:
        
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]
        
        # 다름
        # print_debug(
        #     function_name="SpeculativeModelIOProcessor.run_reject_sampler | before sync_rejection_probs",
        #     draft_token_ids_list=draft_token_ids_list,
        #     draft_probs_tensor_nonzero=self.draft_rejection_probs.get_probs().nonzero()[:,-1].cpu().numpy(),
        #     verify_probs_tensor_nonzero=self.verify_rejection_probs.get_probs().nonzero()[:,-1].cpu().numpy(),
        # )
        
        draft_probs_tensor, verify_probs_tensor = self.sync_rejection_probs(self.draft_rejection_probs, self.verify_rejection_probs)

        rejection_sampler_outputs = self._run_rejection_sampler(
            draft_token_ids_list,
            draft_probs_tensor,
            verify_probs_tensor,
            return_log_probs=return_log_probs,
            return_beta=return_beta,
            return_accept_prob=return_accept_prob,
        )
        
        # if DEBUG:
            # print_debug(
            #     function_name="SpeculativeModelIOProcessor.run_reject_sampler",
            #     draft_token_ids_list=draft_token_ids_list,
            #     draft_probs_tensor_nonzero=self.draft_rejection_probs.get_probs().nonzero()[:,-1].cpu().numpy(),
            #     verify_probs_tensor_nonzero=self.verify_rejection_probs.get_probs().nonzero()[:,-1].cpu().numpy(),
            #     rejection_sampler_outputs=rejection_sampler_outputs,
            # )
        return rejection_sampler_outputs


    @profile_nvtx("SpeculativeModelIOProcessor.reset_rejection_probs")
    def reset_rejection_probs(self):
        self.draft_rejection_probs.reset()
        self.verify_rejection_probs.reset()

    @profile_nvtx("SpeculativeModelIOProcessor.speculative_post_process_requests")
    def speculative_post_process_requests(
            self,
            requests: List[SpeculativeRequest],
            rejection_sampler_outputs: List[RejectionSamplerOutput]
        ):

        for request, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            request.speculative_append_outputs(rejection_sampler_output)
        return requests


import pandas as pd
from .request import HierarchicalSpeculativeRequest
class HierarchicalSpeculativeModelIOProcessor(SpeculativeModelIOProcessor):
    def __init__(self, registry: EngineRegistry, local_rank: Optional[int] = None):
        super().__init__(registry)

        self.draft_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )
        
        self.verify_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps + 1 if self.registry.get('val.engine.use_only_beta_cutoff') else self.num_draft_steps,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )
        
        self.oracle_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps + 1,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )

        self.profile_acceptance_prob = self.registry.get('val.engine.run_profile_acceptance_prob')
        self.use_only_beta_cutoff = self.registry.get('val.engine.use_only_beta_cutoff')
        self.accept_prob_profiles = self.registry.get('val.engine.accept_prob_profiles')
        if not self.profile_acceptance_prob and not self.use_only_beta_cutoff:
            assert self.accept_prob_profiles is not None, "accept_prob_profiles is not set"
        self.profiled_batch_sizes = None
        self.local_rank = local_rank if local_rank is not None else 0
        self.world_size = self.registry.get('val.model.world_size')

    def setup_oracle_step(self, requests: List[HierarchicalSpeculativeRequest]):
        for request in requests:
            request.setup_oracle_step(device=self.device)


        


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.verify_info_gain_with_rejection_sampling")
    def verify_info_gain_with_rejection_sampling(
            self, 
            requests: List[SpeculativeRequest],
        ) -> List[RejectionSamplerOutput]:
        
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]

        max_token_len = max(token_ids.shape[0] for token_ids in draft_token_ids_list)

        draft_probs_tensor = self.draft_rejection_probs.get_probs()[:, :max_token_len, :]
        verify_probs_tensor = self.verify_rejection_probs.get_probs()[:, :max_token_len + 1, :]

        batch_num_draft_tokens = [len(draft_token_ids) for draft_token_ids in draft_token_ids_list]
        batch_token_ids_tensor = pad_sequence([torch.from_numpy(tokens) for tokens in draft_token_ids_list], batch_first=True, padding_value=-1).to(self.device)

        batch_token_ids, _, batch_emitted_token_num = self.rejection_sampler.forward(
            batch_token_ids_tensor,
            draft_probs_tensor,
            verify_probs_tensor
        )

        batch_token_ids = batch_token_ids[:, :max_token_len]
        batch_emitted_token_num = batch_emitted_token_num.clamp(max=max_token_len-1) + 1
        
        # col_idx = torch.arange(draft_probs_tensor.shape[1])[None, :].cuda()
        # mask = col_idx >= batch_emitted_token_num[:, None]
        # self.draft_rejection_probs.cutoff(mask)
        
        col_idx = torch.arange(verify_probs_tensor.shape[1])[None, :].cuda()
        mask = col_idx >= batch_emitted_token_num[:, None]
        self.verify_rejection_probs.cutoff(mask)
        
        rejection_sampler_outputs = self._construct_rejection_sampler_output(
            batch_token_ids=batch_token_ids.cpu().numpy(),
            batch_num_draft_tokens=batch_num_draft_tokens,
            batch_emitted_token_num=batch_emitted_token_num.cpu().numpy(),
            batch_alive_token_num=batch_emitted_token_num.cpu().numpy(),
        )
        
        # Beta Cut
        """
        beta_list_per_request_tensor = self.get_beta_tensor(draft_probs_tensor, verify_probs_tensor)
        second_mask = beta_list_per_request_tensor < self.beta_threshold
        
        has_value = second_mask.any(axis=1)
        beta_cut_idx = torch.argmax(second_mask.to(torch.int32), dim=1)
        beta_cut_idx[~has_value] = beta_list_per_request_tensor.shape[1]
        
        row = torch.arange(second_mask.shape[0], device=self.device)[:, None]
        col = torch.arange(second_mask.shape[1], device=self.device)
        second_mask[row, col] |= (col >= beta_cut_idx[:, None])
        second_mask = torch.cat([second_mask, torch.ones((second_mask.shape[0], 1), dtype=bool, device=self.device)], dim=1)
        
        mask = second_mask
        batch_token_ids_tensor[mask[:, :-1]] = -1
        max_num_drafting_tokens = beta_cut_idx.max().item()
        
        self.draft_rejection_probs.cutoff(mask[:, :-1])
        self.verify_rejection_probs.cutoff(mask[:, :-1])
        
        # emitted_token_num = torch.minimum(batch_emitted_token_num, beta_cut_idx)

        emitted_token_num = beta_cut_idx

        rejection_sampler_outputs = self._construct_rejection_sampler_output(
            batch_token_ids=batch_token_ids.cpu().numpy(),
            batch_num_draft_tokens=batch_num_draft_tokens,
            batch_emitted_token_num=emitted_token_num.cpu().numpy(),
            batch_alive_token_num=emitted_token_num.cpu().numpy(),
            beta_list_per_request=beta_list_per_request_tensor.cpu().numpy(),
        """
        return rejection_sampler_outputs
    
    @np.vectorize
    def _custom_category(x):
        # if x == 0:
        #     return "0"
        if x >= 1:
            return "1"
        elif x < 0.8:
            lower = np.floor(x / 0.1) * 0.1
            upper = lower + 0.1
            return float(f"{lower:.1f}")
        else:
            lower = 0.8 + np.floor((x - 0.8) / 0.02) * 0.02
            upper = lower + 0.02
            return float(f"{lower:.2f}")
        
    def _get_category_indices(self, x):
        indices = np.where(
            x >= 1,
            18,
            np.where(
                x < 0.8,
                np.floor(x / 0.1),
                8 + np.floor((x - 0.8) / 0.02)
            )
        ).astype(int)
        return indices
    
    def _run_rejection_sampler(
        self,
        draft_token_ids_list,
        draft_probs_tensor,
        verify_probs_tensor,
        return_log_probs: bool = False,
        return_beta: bool = False,
        return_accept_prob: bool = False,
    ):
        draft_token_ids_list = [torch.from_numpy(draft_token_ids) for draft_token_ids in draft_token_ids_list]
        draft_token_ids_tensor = pad_sequence(draft_token_ids_list, batch_first=True, padding_value=-1).to(self.device)
        
        draft_token_ids = self.draft_rejection_probs.get_index_from_token_idx(draft_token_ids_tensor)
        
        batch_token_ids_tmp, batch_accepted_token_num, batch_emitted_token_num = self.rejection_sampler.forward(
            draft_token_ids,
            draft_probs_tensor,
            verify_probs_tensor
        )

        batch_token_ids = self.draft_rejection_probs.get_token_idx_from_index(batch_token_ids_tmp)

        # # TODO THERE IS A BUG IN FLASHINFER REJECTION SAMPLER!!!
        # if (batch_token_ids[torch.arange(batch_token_ids.shape[0]), batch_emitted_token_num] == -1).any():
        #     token_ids = torch.tensor([[ 1,  0,  1, -1, -1]], device=self.device)
        #     draft_probs = torch.tensor([[[0.5883, 0.4117, 0.0000, 0.0000],
        #                                  [1.0000, 0.0000, 0.0000, 0.0000],
        #                                  [0.3803, 0.5435, 0.0762, 0.0000],
        #                                  [0.0000, 0.0000, 0.0000, 0.0000],
        #                                  [0.0000, 0.0000, 0.0000, 0.0000]]], device=self.device)
        #     verify_probs = torch.tensor([[[0.4555, 0.5445, 0.0000, 0.0000],
        #                                   [1.0000, 0.0000, 0.0000, 0.0000],
        #                                   [0.5783, 0.2831, 0.0000, 0.1386],
        #                                   [0.2346, 0.6850, 0.0804, 0.0000],
        #                                   [0.0000, 0.0000, 0.0000, 0.0000],
        #                                   [0.0000, 0.0000, 0.0000, 0.0000]]], device=self.device)
        #     output_ids, _, emitted_tmp = self.rejection_sampler.forward(token_ids, draft_probs, verify_probs)
        #     import pdb; pdb.set_trace()

        batch_token_ids = batch_token_ids.cpu().numpy()
        batch_num_draft_tokens = [draft_token_ids.shape[0] for draft_token_ids in draft_token_ids_list]
        batch_emitted_token_num = batch_emitted_token_num.cpu().numpy()
        batch_alive_token_num = batch_emitted_token_num + 1

        beta_list_per_request = self.get_beta_tensor(draft_probs_tensor, verify_probs_tensor).cpu().numpy() if return_beta else None

        accept_prob_list_per_request = None
        
        rejection_sampler_outputs = self._construct_rejection_sampler_output(
            batch_token_ids=batch_token_ids,
            batch_num_draft_tokens=batch_num_draft_tokens,
            batch_emitted_token_num=batch_emitted_token_num,
            batch_alive_token_num=batch_alive_token_num,
            beta_list_per_request=beta_list_per_request,
            accept_prob_list_per_request=accept_prob_list_per_request,
        )
        return rejection_sampler_outputs
    

    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.mid_process")
    def mid_process(
        self,
        requests: List[HierarchicalSpeculativeRequest],
        rejection_sampler_outputs: List[RejectionSamplerOutput]
    ) -> Tuple[List[HierarchicalSpeculativeRequest], List[HierarchicalSpeculativeRequest]]:
        continue_requests = []
        stop_requests = []

        for req, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            req.mid_process(rejection_sampler_output)
            continue_requests.append(req)
        return continue_requests, stop_requests


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.build_oracle_input_tensors")
    def build_oracle_input_tensors(
        self,
        requests: List[HierarchicalSpeculativeRequest]
    ) -> Tuple[torch.Tensor, torch.Tensor, FlashInferMetadata]:
        
        input_ids_list = []
        position_ids_list = []
        paged_kv_indices_list = []
        paged_kv_last_page_len_list = []

        
        for req in requests:
            req.use_oracle_context()
            input_ids, position_ids = req.prepare_oracle_input_ids()
            input_ids_list.append(input_ids)
            position_ids_list.append(position_ids)

            kv_indices, last_page_len = req.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list.append(kv_indices)
            paged_kv_last_page_len_list.append(last_page_len)

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list,
            position_ids_list,
            paged_kv_indices_list,
            paged_kv_last_page_len_list
        )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_draft_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.append_oracle_outputs")
    def append_oracle_outputs(
        self,
        requests: List[HierarchicalSpeculativeRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs_per_request: List[torch.Tensor],
        probs_input_ids_indptr: np.ndarray
    ):
        self.oracle_rejection_probs.multi_request_batch_append(len(requests), probs_per_request, probs_input_ids_indptr)
        
        for request_idx, (request, sampled_tokens, probs) in enumerate(zip(requests, sampled_tokens_per_request, probs_per_request)):
            request.append_oracle_outputs(sampled_tokens)
        torch.cuda.synchronize()
        return requests


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.run_oracle_rejection_sampler")
    def run_oracle_rejection_sampler(
        self, 
        requests: List[HierarchicalSpeculativeRequest],
        return_log_probs: bool = False,
        return_beta: bool = False,
        return_accept_prob: bool = False,
    ):
        rejection_sampler_requests = {i : request for i, request in enumerate(requests) if request.verify_output_state.num_appended_tokens > 0}
        pass_rejection_sampler_requests = {i : request for i, request in enumerate(requests) if request.verify_output_state.num_appended_tokens == 0}
        
        if len(rejection_sampler_requests):
            # Prepare tensors for rejection sampler
            draft_token_ids_list = [request.prepare_oracle_rejection_tensors() for request in rejection_sampler_requests.values()]
            draft_request_indices = list(rejection_sampler_requests.keys())
            draft_request_indices = torch.tensor(draft_request_indices, dtype=torch.int32)
            draft_request_indices = draft_request_indices.to(self.device)

            max_token_len = max(token_ids.shape[0] for token_ids in draft_token_ids_list)

            if self.use_only_beta_cutoff:
                draft_probs_tensor, verify_probs_tensor = self.sync_rejection_probs(self.verify_rejection_probs, self.oracle_rejection_probs)
            else:
                draft_probs_tensor, verify_probs_tensor = self.sync_rejection_probs(self.draft_rejection_probs, self.oracle_rejection_probs)

            if DEBUG:
                print_debug(
                    function_name="HierarchicalSpeculativeModelIOProcessor.run_oracle_rejection_sampler",
                    draft_probs_tensor=draft_probs_tensor.shape,
                    verify_probs_tensor=verify_probs_tensor.shape,
                    draft_token_ids_list=draft_token_ids_list,
                )
            rejection_sampler_outputs = self._run_rejection_sampler(
                draft_token_ids_list,
                draft_probs_tensor[draft_request_indices, :max_token_len, :].contiguous(),
                verify_probs_tensor[draft_request_indices, :max_token_len+1, :].contiguous(),
                return_log_probs=return_log_probs,
                return_beta=return_beta,
                return_accept_prob=return_accept_prob
            )
        else:
            rejection_sampler_outputs = []
        
        if len(pass_rejection_sampler_requests):
            pass_rejection_sampler_outputs = [
                RejectionSamplerOutput(
                    sampled_tokens=request.oracle_output_state.token_ids,
                    emitted_token_num=request.oracle_output_state.num_appended_tokens,
                    num_draft_tokens=request.oracle_output_state.num_appended_tokens,
                    num_alive_tokens=request.oracle_output_state.num_appended_tokens,
                ) for request in pass_rejection_sampler_requests.values()]
        else:
            pass_rejection_sampler_outputs = []
        
        total_requests = len(rejection_sampler_requests) + len(pass_rejection_sampler_requests)
        all_outputs = [None] * total_requests
        all_requests = [None] * total_requests
    
        # Fill rejection sampler outputs and requests
        for idx, request_idx in enumerate(rejection_sampler_requests):
            all_outputs[request_idx] = rejection_sampler_outputs[idx]
            all_requests[request_idx] = rejection_sampler_requests[request_idx]
            
        # Fill pass-through outputs and requests
        for idx, request_idx in enumerate(pass_rejection_sampler_requests):
            all_outputs[request_idx] = pass_rejection_sampler_outputs[idx]
            all_requests[request_idx] = pass_rejection_sampler_requests[request_idx]
            
        rejection_sampler_outputs = all_outputs
        requests = all_requests

        # print(f"\n - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx, request in enumerate(requests):
        #     print_debug(
        #         function_name="HierarchicalSpeculativeModelIOProcessor.run_oracle_rejection_sampler",
        #         generated_ids_list=request.generated_ids,
        #         generated_ids_string=f"\n{self.tokenizer.decode(request.generated_ids)}",
        #         verify_token_ids_list=request.verify_output_state.token_ids,
        #         verify_token_ids_string=f"\n{self.tokenizer.decode(request.verify_output_state.token_ids)}",
        #         oracle_token_ids_list=request.oracle_output_state.token_ids,
        #         oracle_token_ids_string=f"\n{self.tokenizer.decode(request.oracle_output_state.token_ids)}",
        #         # sampled_tokens=rejection_sampler_outputs[idx].sampled_tokens,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")

        return requests, rejection_sampler_outputs


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.reset_rejection_probs")
    def reset_rejection_probs(self):
        self.draft_rejection_probs.reset()
        self.verify_rejection_probs.reset()
        self.oracle_rejection_probs.reset()


    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.hierarchical_speculative_post_process_requests")
    def hierarchical_speculative_post_process_requests(
            self,
            requests: List[HierarchicalSpeculativeRequest],
            rejection_sampler_outputs: List[RejectionSamplerOutput]
        ):

        for request, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            request.hierarchical_speculative_append_outputs(rejection_sampler_output)
        return requests



from .request import LayerSkipRequest
class LayerSkipIOProcessor(SpeculativeModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rejection_sampler = RejectionSampler()
        self.vocab_size = self.registry.get('val.engine.vocab_size')
        self.num_max_batch_requests = self.registry.get('val.engine.num_max_batch_requests')
        self.num_draft_steps = self.registry.get('val.engine.num_draft_steps')
        self.use_sparse_probs = self.registry.get('val.engine.use_sparse_probs')
        self.local_rank = local_rank if local_rank is not None else 0
        
        self.verify_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps+1,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )
    @profile_nvtx("SpeculativeModelIOProcessor.build_draft_input_tensors")
    def build_draft_input_tensors(self, requests: List[LayerSkipRequest]):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len


        for i, request in enumerate(requests):
            request.use_draft_context()
            input_ids, position_ids = request.prepare_draft_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_draft_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

    @profile_nvtx("LayerSkipIOProcessor.build_verify_input_tensors")
    def build_verify_input_tensors(self, requests: List[LayerSkipRequest]):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        for i, request in enumerate(requests):
            request.use_verify_context()

            input_ids, position_ids = request.prepare_verify_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_verify_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #         flashinfer_metadata=flashinfer_metadata,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        # import pdb; pdb.set_trace()
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    
    # def build_oracle_input_tensors(self, requests: List[LayerSkipRequest]):
    #     import pdb; pdb.set_trace()
    #     pass
    @profile_nvtx("LayerSkipIOProcessor.append_draft_outputs")
    def append_draft_outputs(
            self,
            requests: List[LayerSkipRequest],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray
        ):
        # print("---------------------------append_draft_outputs---------------------------")
        self.draft_rejection_probs.multi_request_single_append(len(requests), probs)
        for request_idx, (request, sampled_tokens) in enumerate(zip(requests, sampled_tokens_per_request)):
            request.append_draft_outputs(sampled_tokens)
        self.draft_rejection_probs.appended_token_idx += 1
        return requests
    
    @profile_nvtx("LayerSkipIOProcessor.append_verify_outputs")
    def append_verify_outputs(
        self,
        requests: List[LayerSkipRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs: torch.Tensor,
        probs_input_ids_indptr: np.ndarray
    ):
        if self.use_sparse_probs:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs, probs_input_ids_indptr)
        else:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs)
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            request.append_verify_outputs(sampled_tokens)
        return requests
    
    # def append_oracle_outputs(self, requests: List[LayerSkipRequest], sampled_tokens_per_request: List[np.ndarray], probs_per_request: List[torch.Tensor], probs_input_ids_indptr: np.ndarray):
    #     import pdb; pdb.set_trace()
    #     pass
    
    @profile_nvtx("LayerSkipIOProcessor.hierarchical_speculative_post_process_requests")
    def layerskip_post_process_requests(
            self,
            requests: List[LayerSkipRequest],
            rejection_sampler_outputs: List[RejectionSamplerOutput]
        ):
        for request, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            request.speculative_append_outputs(rejection_sampler_output)
        return requests


   
from .request import HierarchicalLayerSkipRequest
class HiarchicalLayerSkipIOProcessor(HierarchicalSpeculativeModelIOProcessor):
    def __init__(self, registry: EngineRegistry, local_rank: int):
        super().__init__(registry, local_rank)
        ## FIXIT
        self.verify_rejection_probs = RejectionProbs(
            num_max_batch_requests=self.num_max_batch_requests,
            max_token_len=self.num_draft_steps,
            vocab_size=self.vocab_size,
            use_sparse_probs=self.use_sparse_probs
        )
        
    @profile_nvtx("HiarchicalLayerSkipIOProcessor.build_draft_input_tensors")
    def build_draft_input_tensors(self, requests: List[HierarchicalLayerSkipRequest]):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        for i, request in enumerate(requests):
            request.use_draft_context()
            input_ids, position_ids = request.prepare_draft_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )

        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_draft_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        # import pdb; pdb.set_trace()
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    
    @profile_nvtx("HiarchicalLayerSkipIOProcessor.build_verify_input_tensors")
    def build_verify_input_tensors(self, requests: List[HierarchicalLayerSkipRequest]):
        request_len = len(requests)
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        for i, request in enumerate(requests):
            request.use_verify_context()
            input_ids, position_ids = request.prepare_verify_input_ids(draft_step=self.num_draft_steps)
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_verify_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #         flashinfer_metadata=flashinfer_metadata,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
    
    @profile_nvtx("HiarchicalLayerSkipIOProcessor.build_oracle_input_tensors")
    def build_oracle_input_tensors(
        self,
        requests: List[HierarchicalLayerSkipRequest]
    ) -> Tuple[torch.Tensor, torch.Tensor, FlashInferMetadata]:
        
        input_ids_list = []
        position_ids_list = []
        paged_kv_indices_list = []
        paged_kv_last_page_len_list = []
        
        for req in requests:
            req.use_oracle_context()
            input_ids, position_ids = req.prepare_oracle_input_ids()
            input_ids_list.append(input_ids)
            position_ids_list.append(position_ids)

            kv_indices, last_page_len = req.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list.append(kv_indices)
            paged_kv_last_page_len_list.append(last_page_len)

        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list,
            position_ids_list,
            paged_kv_indices_list,
            paged_kv_last_page_len_list
        )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        # for idx in range(len(requests)):
        #     print_debug(
        #         function_name="SpeculativeModelIOProcessor.build_oracle_input_tensors",
        #         input_ids=input_ids_list[idx],
        #         position_ids=position_ids_list[idx],
        #         string=self.tokenizer.decode(input_ids_list[idx]),
        #         paged_kv_indices=paged_kv_indices_list[idx],
        #         paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        #     )
        # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        # import pdb; pdb.set_trace()
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata

    @profile_nvtx("HiarchicalLayerSkipIOProcessor.append_draft_outputs")
    def append_draft_outputs(
            self,
            requests: List[HierarchicalLayerSkipRequest],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray
        ):
        self.draft_rejection_probs.multi_request_single_append(len(requests), probs)
        for request_idx, (request, sampled_tokens) in enumerate(zip(requests, sampled_tokens_per_request)):
            request.append_draft_outputs(sampled_tokens)
        self.draft_rejection_probs.appended_token_idx += 1
        return requests
    
    @profile_nvtx("HiarchicalLayerSkipIOProcessor.append_verify_outputs")
    def append_verify_outputs(
        self,
        requests: List[HierarchicalLayerSkipRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs: torch.Tensor,
        probs_input_ids_indptr: np.ndarray
    ):

        if self.use_sparse_probs:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs, probs_input_ids_indptr)
        else:
            self.verify_rejection_probs.multi_request_batch_append(len(requests), probs)
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            request.append_verify_outputs(sampled_tokens)
        return requests
    
    @profile_nvtx("HiarchicalLayerSkipIOProcessor.append_oracle_outputs")
    def append_oracle_outputs(
        self,
        requests: List[HierarchicalLayerSkipRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs_per_request: List[torch.Tensor],
        probs_input_ids_indptr: np.ndarray
    ):
        self.oracle_rejection_probs.multi_request_batch_append(len(requests), probs_per_request, probs_input_ids_indptr)
        
        for request_idx, (request, sampled_tokens, probs) in enumerate(zip(requests, sampled_tokens_per_request, probs_per_request)):
            request.append_oracle_outputs(sampled_tokens)
        torch.cuda.synchronize()
        return requests
    
    @profile_nvtx("LayerSkipIOProcessor.hierarchical_speculative_post_process_requests")
    def layerskip_post_process_requests(
            self,
            requests: List[HierarchicalLayerSkipRequest],
            rejection_sampler_outputs: List[RejectionSamplerOutput]
        ):

        for request, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            request.hierarchical_speculative_append_outputs(rejection_sampler_output)
        return requests


class SmartSpecModelIOProcessor(SpeculativeModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.moving_average_window = self.registry.get('val.engine.moving_average_window')
        
    @profile_nvtx("SmartSpecModelIOProcessor.mid_process")
    def mid_process(
            self,
            requests: List['SmartSpecRequest'],
        ):
        
        for req in requests:
            if req.should_cut_draft():
                req.mid_process()
        return requests
    
    @profile_nvtx("SmartSpecModelIOProcessor.append_verify_outputs")
    def append_verify_outputs(
        self,
        requests: List[SmartSpecRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs: torch.Tensor,
        probs_input_ids_indptr: np.ndarray
    ):
        self.verify_rejection_probs.multi_request_batch_append(len(requests), probs, probs_input_ids_indptr)
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            request.append_verify_outputs(sampled_tokens)
        return requests
    
    @profile_nvtx("SmartSpecModelIOProcessor.run_rejection_sampler")
    def run_rejection_sampler(
            self, 
            requests: List['SmartSpecRequest'],
            return_log_probs: bool = False,
            return_beta: bool = False,
            return_accept_prob: bool = False,
        ) -> List[RejectionSamplerOutput]:
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]
        
        draft_probs_tensor, verify_probs_tensor = self.sync_rejection_probs(self.draft_rejection_probs, self.verify_rejection_probs)

        max_token_len = max(request.num_draft_steps for request in requests)
        if draft_probs_tensor.shape[1] > max_token_len:
            rejection_sampler_outputs = self._run_rejection_sampler(
                draft_token_ids_list,
                draft_probs_tensor[:, :max_token_len].contiguous(),
                verify_probs_tensor[:, :max_token_len+1].contiguous(),
                return_log_probs=return_log_probs,
                return_beta=return_beta,
                return_accept_prob=return_accept_prob,
            )
        else:
            rejection_sampler_outputs = self._run_rejection_sampler(
                draft_token_ids_list,
                draft_probs_tensor,
                verify_probs_tensor,
                return_log_probs=return_log_probs,
                return_beta=return_beta,
                return_accept_prob=return_accept_prob,
            )
        for i, request in enumerate(requests):
            accepted_tokens = rejection_sampler_outputs[i].emitted_token_num
            request.update_acceptance_rate(accepted_tokens)
        return rejection_sampler_outputs

    @profile_nvtx("SmartSpecModelIOProcessor.get_dynamic_draft_length")
    def get_dynamic_draft_length(self, requests: List['SmartSpecRequest']) -> int:
        acceptance_rates = np.array([req.get_acceptance_rate() for req in requests])
        is_one_mask = acceptance_rates == 1.0
        is_zero_mask = acceptance_rates == 0.0
        
        powers = np.arange(1, self.num_draft_steps + 1)
        acceptance_rates = acceptance_rates.reshape(-1, 1)
        powered_rates = acceptance_rates ** powers
        
        expected_tokens = np.where(
            is_one_mask.reshape(-1, 1),
            powers + 1,
            np.where(
                is_zero_mask.reshape(-1, 1),
                0.0,
                (1 - powered_rates) / (1 - acceptance_rates)
            )
        )
        
        efficiency = expected_tokens / (powers.reshape(1, -1) + 1)
        target_sum = self.num_draft_steps * len(requests) * 0.5
        num_requests = len(requests)
        
        all_efficiencies = []
        for req_idx in range(num_requests):
            for idx in range(self.num_draft_steps):
                all_efficiencies.append((efficiency[req_idx, idx], idx + 1, req_idx))
        
        all_efficiencies.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        
        selected_indices = [0] * num_requests
        current_sum = 0
        
        for eff, idx, req_idx in all_efficiencies:
            if eff <= 0.8 and current_sum >= target_sum:
                break
            if selected_indices[req_idx] < idx:
                selected_indices[req_idx] = idx
                current_sum += 1
        
        for req_idx in range(num_requests):
            if selected_indices[req_idx] == 0:
                selected_indices[req_idx] = 1
                current_sum += 1
        
        for request_idx, request in enumerate(requests):
            request.num_draft_steps = selected_indices[request_idx]
        return selected_indices, max(selected_indices)
    
    
class SVIPModelIOProcessor(SpeculativeModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_for_svip = self.registry.get('val.engine.threshold_for_svip')

    # def setup_speculative_step(self, requests: List['SVIPRequest']):
    #     for req in requests:
    #         req.should_stop_drafting = False
    #     return requests
        
    @profile_nvtx("SVIPModelIOProcessor.mid_process")
    def mid_process(
            self,
            requests: List['SVIPRequest'],
        ):
        
        for req in requests:
            if req.should_cut_draft():
                req.mid_process()
        return requests
    
    @profile_nvtx("SVIPModelIOProcessor.append_verify_outputs")
    def append_verify_outputs(
        self,
        requests: List[SVIPRequest],
        sampled_tokens_per_request: List[np.ndarray],
        probs: torch.Tensor,
        probs_input_ids_indptr: np.ndarray
    ):
        self.verify_rejection_probs.multi_request_batch_append(len(requests), probs, probs_input_ids_indptr)
        for request, sampled_tokens in zip(requests, sampled_tokens_per_request):
            request.append_verify_outputs(sampled_tokens)
        return requests
    
    @profile_nvtx("SVIPModelIOProcessor.run_rejection_sampler")
    def run_rejection_sampler(
            self, 
            requests: List['SVIPRequest'],
            return_log_probs: bool = False,
            return_beta: bool = False,
            return_accept_prob: bool = False,
        ) -> List[RejectionSamplerOutput]:
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]
        
        draft_probs_tensor, verify_probs_tensor = self.sync_rejection_probs(self.draft_rejection_probs, self.verify_rejection_probs)

        max_token_len = max(request.num_draft_steps for request in requests)
        # max_token_len = self.num_draft_steps
        if draft_probs_tensor.shape[1] > max_token_len:
            rejection_sampler_outputs = self._run_rejection_sampler(
                draft_token_ids_list,
                draft_probs_tensor[:, :max_token_len].contiguous(),
                verify_probs_tensor[:, :max_token_len+1].contiguous(),
                return_log_probs=return_log_probs,
                return_beta=return_beta,
                return_accept_prob=return_accept_prob,
            )
        else:
            rejection_sampler_outputs = self._run_rejection_sampler(
                draft_token_ids_list,
                draft_probs_tensor,
                verify_probs_tensor,
                return_log_probs=return_log_probs,
                return_beta=return_beta,
                return_accept_prob=return_accept_prob,
            )
        return rejection_sampler_outputs

    @profile_nvtx("SVIPModelIOProcessor.should_stop_drafting")
    def should_stop_drafting(self, requests: List['SVIPRequest'], idx: int) -> int:
        should_stop_drafting = np.array([req.should_stop_drafting for req in requests])
        
        with torch.no_grad():
            probs = self.draft_rejection_probs.get_probs().select(1, idx)
            entropy = torch.special.entr(probs).sum(dim=-1)
            entropy_flags = (entropy > self.threshold_for_svip ** 2).view(-1).cpu().numpy()
        
        should_stop_drafting = should_stop_drafting | entropy_flags
        
        for i, req in enumerate(requests):
            req.update_request(should_stop_drafting[i], entropy_flags[i], idx)
        return all(should_stop_drafting)
    
    
class Eagle3ModelIOProcessor(SpeculativeModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
    
    @profile_nvtx("Eagle3ModelIOProcessor.build_draft_input_tensors")
    def build_draft_input_tensors(self, requests: List[Eagle3Request], prefill: bool = False):
        request_len = len([request for request in requests if not prefill or request.is_draft_prefill_step])
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        hidden_states = []
        i = 0
        for request in requests:
            if prefill and not request.is_draft_prefill_step:
                continue
            request.use_draft_context()
            input_ids, position_ids = request.prepare_draft_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len
            hidden_states.append(request.prepare_draft_hidden_states(len(input_ids)))
            i += 1

        hidden_states = torch.cat(hidden_states, dim=0)
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states
    
    def append_hidden_states(self, requests: List[Eagle3Request], hidden_states: torch.Tensor):
        for i, request in enumerate(requests):
            hidden_state = hidden_states[i]
            request.append_hidden_states(hidden_state)
        return requests
    
    @profile_nvtx("SpeculativeModelIOProcessor.append_draft_outputs")
    def append_eagle3_draft_outputs(
            self,
            requests: List[Eagle3Request],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray,
            mapping_fn: Callable
        ):
        
        for request_idx, (request, sampled_tokens) in enumerate(zip(requests, sampled_tokens_per_request)):
            mapped_token_ids = mapping_fn(torch.tensor(sampled_tokens))
            self.draft_rejection_probs.probs[request_idx, self.draft_rejection_probs.appended_token_idx, mapped_token_ids] = 1
            request.append_draft_outputs(mapped_token_ids)
        self.draft_rejection_probs.appended_request_idx = np.maximum(self.draft_rejection_probs.appended_request_idx, len(requests))
        self.draft_rejection_probs.appended_token_idx += 1
        return requests
    
    def set_hidden_states(self, requests: List[Eagle3Request], hidden_states: torch.Tensor):
        for i, request in enumerate(requests):
            request.hidden_states = hidden_states[i]
        return requests
    
    def evaluate_eagle3_posterior(
        self,
        requests: List[Eagle3Request]
    ):
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]
        
        rejection_sampler_outputs = self._run_rejection_sampler(
            draft_token_ids_list,
            self.draft_rejection_probs.get_probs(),
            self.verify_rejection_probs.get_probs(),
            return_log_probs=False,
            return_beta=False,
            return_accept_prob=False,
        )
        
        return rejection_sampler_outputs


class Eagle3SVModelIOProcessor(HierarchicalSpeculativeModelIOProcessor):
    def __init__(self, *args, local_rank: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
    
    @profile_nvtx("Eagle3SVModelIOProcessor.build_draft_input_tensors")
    def build_draft_input_tensors(self, requests: List[Eagle3Request], prefill: bool = False):
        request_len = len([request for request in requests if not prefill or request.is_draft_prefill_step])
        input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list = [None] * request_len, [None] * request_len, [None] * request_len, [None] * request_len

        hidden_states = []
        i = 0
        for request in requests:
            if prefill and not request.is_draft_prefill_step:
                continue
            request.use_draft_context()
            input_ids, position_ids = request.prepare_draft_input_ids()
            input_ids_list[i] = input_ids
            position_ids_list[i] = position_ids

            kv_indices, last_page_len = request.prepare_paged_kv_blocks(self.kv_cache_block_size)
            paged_kv_indices_list[i] = kv_indices
            paged_kv_last_page_len_list[i] = last_page_len
            hidden_states.append(request.prepare_draft_hidden_states(len(input_ids)))
            i += 1

        hidden_states = torch.cat(hidden_states, dim=0)
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = self._build_flashinfer_model_inputs(
            input_ids_list, position_ids_list, paged_kv_indices_list, paged_kv_last_page_len_list
        )
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata, hidden_states
    
    def append_hidden_states(self, requests: List[Eagle3Request], hidden_states: torch.Tensor):
        for i, request in enumerate(requests):
            hidden_state = hidden_states[i]
            request.append_hidden_states(hidden_state)
        return requests

    @profile_nvtx("HierarchicalSpeculativeModelIOProcessor.mid_process")
    def mid_process(
        self,
        requests: List[HierarchicalSpeculativeRequest],
        rejection_sampler_outputs: List[RejectionSamplerOutput]
    ) -> Tuple[List[HierarchicalSpeculativeRequest], List[HierarchicalSpeculativeRequest]]:
        continue_requests = []
        stop_requests = []

        for req, rejection_sampler_output in zip(requests, rejection_sampler_outputs):
            req.mid_process(rejection_sampler_output)
            continue_requests.append(req)
        return continue_requests, stop_requests
    
    @profile_nvtx("SpeculativeModelIOProcessor.append_draft_outputs")
    def append_eagle3_draft_outputs(
            self,
            requests: List[Eagle3Request],
            sampled_tokens_per_request: List[np.ndarray],
            probs: torch.Tensor,
            probs_input_ids_indptr: np.ndarray,
            mapping_fn: Callable
        ):
        
        for request_idx, (request, sampled_tokens) in enumerate(zip(requests, sampled_tokens_per_request)):
            mapped_token_ids = mapping_fn(torch.tensor(sampled_tokens))
            self.draft_rejection_probs.probs[request_idx, self.draft_rejection_probs.appended_token_idx, mapped_token_ids] = 1
            request.append_draft_outputs(mapped_token_ids)
        self.draft_rejection_probs.appended_request_idx = np.maximum(self.draft_rejection_probs.appended_request_idx, len(requests))
        self.draft_rejection_probs.appended_token_idx += 1
        return requests
    
    def set_hidden_states(self, requests: List[Eagle3Request], hidden_states: torch.Tensor):
        for i, request in enumerate(requests):
            request.hidden_states = hidden_states[i]
        return requests
    
    def evaluate_eagle3_posterior(
        self,
        requests: List[Eagle3Request]
    ):
        draft_token_ids_list = [request.prepare_verify_rejection_tensors() for request in requests]
        
        rejection_sampler_outputs = self._run_rejection_sampler(
            draft_token_ids_list,
            self.draft_rejection_probs.get_probs(),
            self.oracle_rejection_probs.get_probs(),
            return_log_probs=False,
            return_beta=False,
            return_accept_prob=False,
        )
        
        return rejection_sampler_outputs