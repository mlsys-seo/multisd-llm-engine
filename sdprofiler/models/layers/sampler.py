import os, sys
from functools import cached_property
from typing import Optional, Tuple, Dict, List

import torch
from torch import nn
from functools import wraps

from flashinfer.sampling import (top_k_top_p_sampling_from_probs, sampling_from_probs)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hierarchical_spec.utils import SamplingParams


class Sampler(nn.Module):
    def __init__(self,
                 max_vocab_size: int = None,
                 min_vocab_size: int = None):
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.min_vocab_size = min_vocab_size

    def _get_bin_counts_and_mask(
        self, 
        token_ids: torch.Tensor,
        vocab_size: int,
        num_seqs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_counts = torch.zeros(vocab_size, 
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
        return logits_sort.masked_fill(top_k_mask, -float('inf'))
    
    def _apply_top_p(self, logits_sort, top_p) -> torch.Tensor:
        probs_sort = logits_sort.softmax(dim=-1)
        top_p_mask = probs_sort.cumsum(dim=-1) <= (1 - top_p)
        return logits_sort.masked_fill(top_p_mask, -float('inf'))
    
    def forward(
        self,
        logits: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        output_token_ids: torch.Tensor,
        sampling_params: SamplingParams,
        eos_token_ids: List[int]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        current_vocab_size = logits.shape[-1]
        if self.min_vocab_size is not None and current_vocab_size > self.min_vocab_size:
            logits = logits[:, :, :self.min_vocab_size].clone()
        
        if not sampling_params.do_sample or sampling_params.temperature == 0.0:
            sampled_tokens = logits.argmax(dim=-1)
            return sampled_tokens, None

        logits = self.apply_min_tokens_penalty(logits, output_token_ids, sampling_params, eos_token_ids)
        logits = self.apply_penalties(logits, prompt_token_ids, output_token_ids, sampling_params)

        if sampling_params.temperature != 1.0:
            logits = logits / sampling_params.temperature

        logits = self.apply_top_k_top_p(logits, sampling_params)

        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

        sampled_tokens = torch.zeros((probs.shape[0], probs.shape[1]), dtype=torch.int32, device=probs.device)
        for sample_idx in range(probs.shape[0]):
            single_probs = probs[sample_idx,]
            uniform_samples = torch.rand(single_probs.shape[0], device=probs.device).view(1, -1)
            token_ids, _ = top_k_top_p_sampling_from_probs(
                single_probs, 
                uniform_samples, 
                sampling_params.top_k, 
                sampling_params.top_p)
            sampled_tokens[sample_idx,] = token_ids

        return sampled_tokens, probs


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
        logits_sort, logits_indices = torch.sort(logits[0], dim=-1, descending=False)
        if sampling_params.top_k > 0:
            logits_sort = self._apply_top_k(logits_sort, sampling_params.top_k)
        if sampling_params.top_p < 1.0:
            logits_sort = self._apply_top_p(logits_sort, sampling_params.top_p)
        logits[0] = torch.empty_like(logits[0]).scatter_(-1, logits_indices, logits_sort)
        return logits


def renormalize_fn(probs: torch.Tensor) -> torch.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True)


class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler

    def _rejection_greedy(
        self,
        draft_tokens: torch.Tensor, # [num_draft_tokens]
        oracle_tokens: torch.Tensor, # [num_draft_tokens]
        sampling_params: SamplingParams,
        num_draft_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        all_accepted = True
        num_accepted_tokens = 0
        new_token = None
        
        # find first idx where oracle_token != draft_token
        for draft_idx, (draft_token, oracle_token) in enumerate(zip(draft_tokens, oracle_tokens)):
            if draft_token != oracle_token:
                break

        num_accepted_tokens = draft_idx
        accepted_tokens = draft_tokens[:num_accepted_tokens]

        return accepted_tokens, num_accepted_tokens

    def forward(
        self,
        draft_tokens: torch.Tensor, # [num_draft_tokens]
        draft_probs: torch.Tensor, # [num_draft_tokens, vocab_size]
        oracle_tokens: torch.Tensor, # [num_draft_tokens]
        oracle_probs: torch.Tensor, # [num_draft_tokens, vocab_size]
        sampling_params: SamplingParams,
        num_draft_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if not sampling_params.do_sample or sampling_params.temperature == 0.0:
            return self._rejection_greedy(
                draft_tokens=draft_tokens,
                oracle_tokens=oracle_tokens,
                sampling_params=sampling_params,
                num_draft_tokens=num_draft_tokens
            )

        all_accepted = True
        num_accepted_tokens = 0
        new_token = None
        for draft_idx in range(num_draft_tokens):
            draft_token = draft_tokens[draft_idx]
            acceptance_prob = min(1, oracle_probs[draft_idx, draft_token] / draft_probs[draft_idx, draft_token])
            if torch.rand(1).item() < acceptance_prob:  # accepted
                num_accepted_tokens += 1
            else:  # rejected
                new_probs = oracle_probs[draft_idx] - draft_probs[draft_idx]
                new_probs = renormalize_fn(new_probs)
                new_token, _ = self.sampler(new_probs, sampling_params, return_tokens=True)  # resample
                all_accepted = False

                break
        
        # assert (num_accepted_tokens == (num_draft_tokens-1)) == all_accepted


        accepted_tokens = draft_tokens[:num_accepted_tokens]

        if new_token is not None:
            accepted_tokens = torch.cat([accepted_tokens, new_token], dim=0)
        # if all draft tokens were accepted, sample a final token
        if all_accepted:
            bonus_token, _ = self.sampler(oracle_probs[-1,], sampling_params, return_tokens=True)
            accepted_tokens = torch.cat([accepted_tokens, bonus_token], dim=0)
        
        assert accepted_tokens.shape[-1] == num_accepted_tokens + 1
        return accepted_tokens, num_accepted_tokens