import torch
import triton
import triton.language as tl

@triton.jit
def speculative_sampling_kernel(draft_probs_ptr, draft_token_ids_ptr, uniform_samples_ptr, target_probs_ptr, output_ptr, output_accepted_token_num_ptr, output_emitted_draft_token_num_ptr,
                                num_speculate_tokens, vocab_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # i는 while 문을 빨리 끝내기 위한 반복문임.
    i = 0
    while i < num_speculate_tokens:
        token_id = tl.load(draft_token_ids_ptr + pid * num_speculate_tokens + i)
        reject_flag = True
        if token_id != -1:
            draft_prob = tl.load(draft_probs_ptr + pid * num_speculate_tokens * vocab_size + i * vocab_size + token_id)
            target_prob = tl.load(target_probs_ptr + pid * (num_speculate_tokens + 1) * vocab_size + i * vocab_size + token_id)
            uniform_sample = tl.load(uniform_samples_ptr + pid * (num_speculate_tokens + 1) + i)

            if (target_prob / draft_prob) >= uniform_sample:
                tl.store(output_ptr + pid * (num_speculate_tokens + 1) + i, token_id)
                reject_flag = False

        # True => reject
        if reject_flag:
            uniform_sample = tl.load(uniform_samples_ptr + pid * (num_speculate_tokens + 1) + num_speculate_tokens)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            #target_mask = tl.load(target_probs_ptr + pid * (num_speculate_tokens + 1) * vocab_size + i * vocab_size + offsets, mask=mask)
            #mask = mask & (target_mask > 1e-8)
            
            target_p = tl.load(target_probs_ptr + pid * (num_speculate_tokens + 1) * vocab_size + i * vocab_size + offsets, mask=mask)
            draft_p = tl.load(draft_probs_ptr + pid * num_speculate_tokens * vocab_size + i * vocab_size + offsets, mask=mask)

            new_p = target_p - draft_p
            new_p = tl.maximum(new_p, 0.0)
            new_p = new_p / tl.sum(new_p)
            new_p = tl.cumsum(new_p)

            # 랜덤 확률 분포에서 샘플링
            new_p = tl.where(new_p <= uniform_sample, 2.0, new_p)
            sampled_token_id = tl.argmin(new_p, axis=-1, tie_break_left=True)

            # 디버깅 결과, tl.cumsum에서 아주 작은 오차가 발생함. argmin이 마지막 토큰에 대해선 잘 작동하지 않는 문제. 예외 상황 정의 후 해결
            if sampled_token_id >= vocab_size:
                sampled_token_id = vocab_size - 1

            # 토큰 저장
            tl.store(output_ptr + pid * (num_speculate_tokens + 1) + i, sampled_token_id)
            tl.store(output_accepted_token_num_ptr + pid, i)
            tl.store(output_emitted_draft_token_num_ptr + pid, i)
            i = 100
        i += 1

    # 모든 토큰이 Accept 된 경우, 마지막 토큰 샘플링
    if i != 101:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        #target_mask = tl.load(target_probs_ptr + pid * (num_speculate_tokens + 1) * vocab_size + num_speculate_tokens * vocab_size + offsets, mask=mask)
        #mask = mask & (target_mask > 1e-8)
        target_p = tl.load(target_probs_ptr + pid * (num_speculate_tokens + 1) * vocab_size + i * vocab_size + offsets, mask=mask)

        uniform_sample = tl.load(uniform_samples_ptr + pid * (num_speculate_tokens + 1) + num_speculate_tokens)
        new_p = tl.cumsum(target_p)
        new_p = tl.where(new_p <= uniform_sample, 2.0, new_p)
        sampled_token_id = tl.argmin(new_p, axis=-1, tie_break_left=True)

        # 디버깅 결과, tl.cumsum에서 아주 작은 오차가 발생함. argmin이 마지막 토큰에 대해선 잘 작동하지 않는 문제. 예외 상황 정의 후 해결
        if sampled_token_id >= vocab_size:
            sampled_token_id = vocab_size - 1

        # 토큰 저장
        tl.store(output_ptr + pid * (num_speculate_tokens + 1) + num_speculate_tokens, sampled_token_id)
        tl.store(output_accepted_token_num_ptr + pid, num_speculate_tokens)
        tl.store(output_emitted_draft_token_num_ptr + pid, num_speculate_tokens)

def speculative_sampling(draft_probs: torch.Tensor, draft_token_ids: torch.Tensor, target_probs: torch.Tensor):
    batch_size, num_speculate_tokens, vocab_size = draft_probs.shape
    uniform_samples = torch.empty(batch_size, num_speculate_tokens + 1, device=draft_probs.device)
    uniform_samples.uniform_()
    BLOCK_SIZE = triton.next_power_of_2(vocab_size)
    output = torch.full((batch_size, num_speculate_tokens + 1), -1, dtype=torch.int64, device=draft_probs.device)
    output_accepted_token_num = torch.zeros(batch_size, dtype=torch.int64, device=draft_probs.device)
    output_emitted_draft_token_num = torch.zeros(batch_size, dtype=torch.int64, device=draft_probs.device)
    #debug = torch.full((128,), 0, dtype=torch.float32).to(0)
    grid = (batch_size, )
    binary = speculative_sampling_kernel[grid](draft_probs, draft_token_ids, uniform_samples, target_probs, output, output_accepted_token_num, output_emitted_draft_token_num,
                    num_speculate_tokens, vocab_size,
                    BLOCK_SIZE=BLOCK_SIZE, num_warps=32, num_stages=1)
    
    return output, output_accepted_token_num, output_emitted_draft_token_num
