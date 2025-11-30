import os, sys
from functools import cached_property
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from .utils.ops.rejection_sampling import speculative_sampling


import torch
import numpy as np

from torch import nn
from torch.nn import functional as F


from flashinfer.sampling import (sampling_from_probs, chain_speculative_sampling, top_k_mask_logits, top_p_renorm_probs)

from .utils.common import profile_nvtx, print_debug


DEBUG = False

@dataclass
class SamplingParams:
    do_sample: bool = False
    top_k: int = 0
    top_p: float = 0.0
    temperature: float = 0.0
    repetition_penalty: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    disable_eos_token: bool = False
    is_prefill: bool = True
    eos_token_ids: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.do_sample = False if self.temperature == 0.0 else self.do_sample
        

def renormalize_fn(probs: torch.Tensor) -> torch.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True)


class Sampler(nn.Module):
    def __init__(
            self,
            max_vocab_size: int = None,
            min_vocab_size: int = None):
        
        super().__init__()
        assert max_vocab_size is not None or min_vocab_size is not None
        min_vocab_size = max_vocab_size if min_vocab_size is None else min_vocab_size
        self.target_vocab_size = min(max_vocab_size, min_vocab_size)


    def _get_bin_counts_and_mask(
        self, 
        token_ids: torch.Tensor,
        vocab_size: int,
        num_seqs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_counts = torch.zeros(
            vocab_size, 
            dtype=torch.long,
            device=token_ids.device)
        bin_counts.scatter_add_(-1, token_ids, torch.ones_like(token_ids))
        bin_counts = bin_counts.expand(num_seqs, -1)
        return bin_counts, bin_counts > 0


    def _apply_repetition_penalty(self, logits, prompt_mask, output_mask, penalty) -> torch.Tensor:
        repetition_penalty = torch.tensor([penalty], device=logits.device)[:, None].repeat(logits.shape)
        repetition_penalty[~(prompt_mask | output_mask)] = 1.0
        return torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)


    def _apply_frequency_penalty(self, logits, output_bin_counts, penalty) -> torch.Tensor:
        return logits - penalty * output_bin_counts


    def _apply_presence_penalty(self, logits, output_mask, penalty) -> torch.Tensor:
        return logits - penalty * output_mask


    def _apply_top_k(self, logits_sort, top_k) -> torch.Tensor:
        top_k_mask = logits_sort < logits_sort[:, -top_k].unsqueeze(-1)
        return logits_sort.masked_fill_(top_k_mask, -float('inf'))


    def _apply_top_p(self, logits_sort, top_p) -> torch.Tensor:
        probs_sort = logits_sort.softmax(dim=-1)
        top_p_mask = probs_sort.cumsum(dim=-1) <= (1 - top_p)
        return logits_sort.masked_fill_(top_p_mask, -float('inf'))


    def apply_min_tokens_penalty(
        self, 
        logits: torch.Tensor,
        output_token_ids: torch.Tensor,
        sampling_params: SamplingParams,
        eos_token_ids: List[int]
    ) -> torch.Tensor:
        if (sampling_params.min_output_length is not None) and (output_token_ids.shape[-1] < sampling_params.min_output_length):
            for eos_token_id in eos_token_ids:
                logits[0, :, eos_token_id] = -float('inf')
        return logits
    
    def apply_penalties(self, logits, prompt_token_ids, output_token_ids, sampling_params) -> torch.Tensor:
        num_seqs, vocab_size = logits[0].shape
        _, prompt_mask = self._get_bin_counts_and_mask(prompt_token_ids[0], vocab_size, num_seqs)
        output_bin_counts, output_mask = self._get_bin_counts_and_mask(output_token_ids, vocab_size, num_seqs)
        
        # Apply repetition penalty
        if sampling_params.repetition_penalty != None and sampling_params.repetition_penalty != 1.0:
            logits[0] = self._apply_repetition_penalty(logits[0], prompt_mask, output_mask, sampling_params.repetition_penalty)
        
        # Apply frequency penalty
        if sampling_params.frequency_penalty != None and sampling_params.frequency_penalty != 0.0:
            logits[0] = self._apply_frequency_penalty(logits[0], output_bin_counts, sampling_params.frequency_penalty)
        
        # Apply presence penalty
        if sampling_params.presence_penalty != None and sampling_params.presence_penalty != 0.0:
            logits[0] = self._apply_presence_penalty(logits[0], output_mask, sampling_params.presence_penalty)
        
        return logits


    def apply_top_k_top_p(self, logits, sampling_params) -> torch.Tensor:
        logits, logits_indices = torch.sort(logits, dim=-1, descending=False)
        if sampling_params.top_k > 0:
            logits = self._apply_top_k(logits, sampling_params.top_k)
        if sampling_params.top_p < 1.0:
            logits = self._apply_top_p(logits, sampling_params.top_p)
        logits = torch.empty_like(logits).scatter_(-1, logits_indices, logits)
        return logits
    

    @profile_nvtx("Sampler.forward")
    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        eos_token_ids: List[int],
        sampling_params: SamplingParams = None,
        prompt_token_ids: torch.Tensor = None,
        output_token_ids: torch.Tensor = None,
        return_log_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        current_vocab_size = logits.shape[-1]
        if current_vocab_size > self.target_vocab_size:
            logits = logits[..., :self.target_vocab_size]
    
        if DEBUG:
            print_debug(
                function_name="Sampler.forward",
                logits_shape=logits.shape,
                sampling_params=sampling_params,
                eos_token_ids=eos_token_ids,
                prompt_token_ids=prompt_token_ids,
                output_token_ids=output_token_ids,
                return_log_probs=return_log_probs
            )
        if sampling_params is None or (not sampling_params.do_sample) or (sampling_params.temperature == 0.0) or sampling_params.is_prefill:
            sampled_tokens = logits.argmax(dim=-1)
            num_classes = logits.size(-1)
            # print(f"sample_tokens.shape {sampled_tokens.shape}")    
            # print(f"torch.cuda.memory(current): {torch.cuda.memory_stats(self.device)['allocated_bytes.all.current']/1024/1024} MB")
            # print(f"                       torch.cuda.memory(reserved current): {torch.cuda.memory_stats(self.device)['reserved_bytes.all.current']/1024/1024} MB")
            
            probs = F.one_hot(sampled_tokens, num_classes=num_classes).float()
            if return_log_probs:
                log_probs = torch.log(probs).sum(dim=-1)
                return sampled_tokens, log_probs
            else:
                return sampled_tokens, probs

        if sampling_params.temperature != 1.0:
            logits.div_(sampling_params.temperature)

        # if sampling_params.top_k > 0:
        #     logits = top_k_mask_logits(logits, sampling_params.top_k)

        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        if sampling_params.top_p < 1.0:
            probs = top_p_renorm_probs(probs, sampling_params.top_p)
        
        sampled_tokens = sampling_from_probs(probs)

        if return_log_probs:
            log_probs = torch.log(probs).sum(dim=-1)
            return sampled_tokens, log_probs
        else:
            return sampled_tokens, probs

class RejectionSampler(nn.Module):
    def __init__(self):
        super().__init__()

    @profile_nvtx("RejectionSampler.forward")
    @torch.no_grad()
    def forward(
        self,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        verify_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        

        '''
        Outputs:
        - output_token_ids (torch.Tensor)
            - The output token indices verified by the target model, rejected samples are padded with -1. 
            - Compared to input draft_token_ids, the output tensor has an additional token index at the end for the final token, 
            - if all previous tokens are accepted, another "bonus" token will be sampled from the target model's probability. 
            - Shape: (batch_size, num_specutate_tokens + 1)

        - output_accepted_token_num (torch.Tensor) 
            - The number of tokens that can be accepted if each token is considered independently for each request. 
            - This metric does not consider the fact that rejection sampling will stop at the first token that does not satisfy the probablity requirement r < p/q. 
            - It only evaluates the alignment of draft model and target model. s
            - Shape: (batch_size)

        output_emitted_token_num (torch.Tensor)
            - The number of tokens that are finally emitted/generated for each request. 
            - Shape: (batch_size)
        '''
        # print(f"draft_token_ids.shape: {draft_token_ids.shape}")
        # print(f"draft_probs.shape: {draft_probs.shape}")
        # print(f"verify_probs.shape: {verify_probs.shape}")


        batch_size, num_draft_tokens = draft_token_ids.shape
        
        #print(f"\n------------- chain_speculative_sampling -------------")
        #print(f"batch_size: {batch_size}")
        ## print(f"uniform_samples.shape: {uniform_samples.shape}")
        #print(f"draft_token_ids: {draft_token_ids}")
        #print(f"draft_probs.shape: {draft_probs.shape}")
        #print(f"draft_token_ids.shape: {draft_token_ids.shape}")
        #print(f"verify_probs.shape: {verify_probs.shape}")
        #print(f"------------------------------------------------------")
        
        accepted_token_ids, output_accepted_token_num, output_emitted_token_num =\
            chain_speculative_sampling(draft_probs[:batch_size, : ,:], draft_token_ids, verify_probs[:batch_size, : ,:])
            
        #accepted_token_ids, output_accepted_token_num, output_emitted_token_num =\
        #    speculative_sampling(draft_probs, draft_token_ids, verify_probs)
        

        return accepted_token_ids, output_accepted_token_num, output_emitted_token_num
