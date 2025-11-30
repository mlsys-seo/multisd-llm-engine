import torch

import numpy as np
from numba import njit
import torch.nn.functional as F
# from .common import profile_nvtx



# @profile_nvtx("calculate_beta")
@torch.jit.script
def calculate_beta_torch(
        draft_probs: torch.Tensor, 
        verify_probs: torch.Tensor
    ) -> torch.Tensor:
    if verify_probs.shape[1] != draft_probs.shape[1]:
        verify_probs = verify_probs[:, :-1, :]

    draft_token_indices = draft_probs.nonzero()[:,-1]
    verify_token_indices = verify_probs.nonzero()[:,-1]
    all_token_indices = torch.cat([draft_token_indices, verify_token_indices], dim=0)
    all_token_indices = torch.unique(all_token_indices)
    draft_probs = draft_probs.index_select(2, all_token_indices)
    verify_probs = verify_probs.index_select(2, all_token_indices)

    beta_list = torch.minimum(draft_probs, verify_probs).sum(dim=-1)
    return beta_list


# @profile_nvtx("calculate_accept_prob")
@torch.jit.script
def calculate_accept_prob_torch(
    selected_draft_probs: torch.Tensor, 
    selected_verify_probs: torch.Tensor
) -> torch.Tensor:
    if selected_verify_probs.size(1) != selected_draft_probs.size(1):
        selected_verify_probs = selected_verify_probs[:, :-1, :]

    accept_prob = selected_verify_probs / selected_draft_probs
    accept_prob.nan_to_num_(0.0).clamp_(min=0.0, max=1.0)
    return accept_prob



@torch.jit.script
def substitute_tensor(
    tensor: torch.Tensor, 
    candidate_keys: torch.Tensor, 
    candidate_values: torch.Tensor
) -> torch.Tensor:
    # tensor: (N, M), candidate_keys: (K,), candidate_values: (K,)
    # tensor.unsqueeze(-1): (N, M, 1) 와 candidate_keys (K,)가 브로드캐스팅되어 (N, M, K) 차원으로 연산됨
    diff = torch.abs(tensor.unsqueeze(-1) - candidate_keys)
    min_idx = torch.argmin(diff, dim=-1)  # (N, M) 각 원소마다 가장 작은 차이의 인덱스
    result_tensor = candidate_values[min_idx]  # 해당 인덱스에 대응하는 candidate_values 선택
    return result_tensor


def stack_sparse_vectorized(sparse_tensors, dim=0):
    base_shape = sparse_tensors[0].shape
    
    N = len(sparse_tensors)
    d = len(base_shape)
    
    indices_list = [t._indices() for t in sparse_tensors]  # 각 요소의 shape: (d, nnz)
    values_list = [t._values() for t in sparse_tensors]      # 각 요소의 shape: (nnz,)
    
    nnz0 = indices_list[0].shape[1]
    for idx in indices_list:
        if idx.shape[1] != nnz0:
            raise ValueError("All sparse tensors must have the same number of non-zero elements (nnz) for vectorized stacking.")
    
    indices_tensor = torch.stack(indices_list, dim=0)  # shape: (N, d, nnz)
    values_tensor = torch.stack(values_list, dim=0)      # shape: (N, nnz)
    
    new_dim_indices = torch.arange(N, device=indices_tensor.device).view(N, 1, 1).expand(N, 1, nnz0)
    
    if dim == 0:
        new_indices_tensor = torch.cat([new_dim_indices, indices_tensor], dim=1)
    elif dim == d:
        new_indices_tensor = torch.cat([indices_tensor, new_dim_indices], dim=1)
    else:
        new_indices_tensor = torch.cat([indices_tensor[:, :dim, :], new_dim_indices, indices_tensor[:, dim:, :]], dim=1)
    # new_indices_tensor의 shape: (N, d+1, nnz)
    new_indices = new_indices_tensor.reshape(d+1, -1)  # shape: (d+1, N * nnz)
    new_values = values_tensor.reshape(-1)             # shape: (N * nnz,)
    
    new_shape_list = list(base_shape)
    new_shape_list.insert(dim, N)
    new_shape = tuple(new_shape_list)
    
    return torch.sparse_coo_tensor(new_indices, new_values, size=new_shape)


# numpy가 젤 빠름
def compute_exp_len_np(prob: np.ndarray) -> np.ndarray:
    batch_size, k = prob.shape
    prefix_prod = np.cumprod(prob, axis=1)
    idx = np.arange(1, k, dtype=prob.dtype)
    T = prefix_prod[:, :-1] * (1.0 - prob[:, 1:]) * idx
    cT = np.cumsum(T, axis=1)
    partial1_shifted = np.empty((batch_size, k), dtype=prob.dtype)
    partial1_shifted[:, 0] = 0
    partial1_shifted[:, 1:] = cT
    n_arange = np.arange(1, k+1, dtype=prob.dtype)
    exp_len = partial1_shifted + prefix_prod * n_arange
    return exp_len


@torch.jit.script
def compute_exp_len(prob: torch.Tensor) -> torch.Tensor:
    batch_size, k = prob.shape
    prefix_prod = torch.cumprod(prob, dim=1)  # shape: (batch_size, k)
    idx = torch.arange(1, k, device=prob.device, dtype=prob.dtype)  # [1..k-1]
    T = idx * prefix_prod[:, :-1] * (1.0 - prob[:, 1:])  # shape: (batch_size, k-1)
    cT = torch.cumsum(T, dim=1)  # shape: (batch_size, k-1)
    partial1_shifted = torch.cat(
        [torch.zeros(batch_size, 1, device=prob.device, dtype=prob.dtype),
         cT],
        dim=1
    )  # shape: (batch_size, k)
    n_arange = torch.arange(1, k+1, device=prob.device, dtype=prob.dtype).unsqueeze(0)
    exp_len = partial1_shifted + n_arange * prefix_prod
    return exp_len

import heapq
# @profile_nvtx("incremental_select")
@njit
def incremental_select(expected_tokens_diff: np.ndarray):
    n_queries, n_tokens = expected_tokens_diff.shape
    counts = np.zeros(n_queries, dtype=np.int32)

    heap = []
    for query in range(n_queries):
        heap.append((expected_tokens_diff[query, counts[query]], query))
    heapq.heapify(heap)

    results = [counts.copy()]
    while heap:
        value, query = heapq.heappop(heap)
        counts[query] += 1
        results.append(counts.copy())
        if counts[query] < n_tokens:
            heapq.heappush(heap, (expected_tokens_diff[query, counts[query]], query))
    return results
    # TODO:
    # results 만들어 놓고 update
    # tuple도 mem allocation 하고 있음
    # heap 대신 bucket 만들어놓고 query 넣었다 빼기

@njit
def incremental_select_new(expected_tokens_diff: np.ndarray):
    n_queries, n_tokens = expected_tokens_diff.shape
    counts = np.zeros(n_queries, dtype=np.int32)

    heap = []
    for query in range(n_queries):
        heap.append((expected_tokens_diff[query, counts[query]], query))
    heapq.heapify(heap)

    results = [counts.copy()]
    while heap:
        value, query = heapq.heappop(heap)
        counts[query] += 1
        results.append(counts.copy())
        if counts[query] < n_tokens:
            heapq.heappush(heap, (expected_tokens_diff[query, counts[query]], query))
    return results

@njit
def incremental_select_numba(expected_tokens_diff):
    """
    순서 제약을 유지하면서 각 쿼리의 토큰 선택 상태를 시뮬레이션.
    expected_tokens_diff는 음수 변환되어, 각 쿼리에서 최대 기여도를 최소값 찾기 방식을 사용.
    """
    n_queries, n_tokens = expected_tokens_diff.shape
    counts = np.zeros(n_queries, dtype=np.int32)
    max_steps = n_queries * n_tokens + 1  # 최대 선택 횟수
    results = np.empty((max_steps, n_queries), dtype=np.int32)
    results[0, :] = counts
    step = 1
    while True:
        best_val = 1e20  # 매우 큰 값
        best_query = -1
        for i in range(n_queries):
            if counts[i] < n_tokens:
                candidate = expected_tokens_diff[i, counts[i]]
                if candidate < best_val:
                    best_val = candidate
                    best_query = i
        if best_query == -1:
            break
        counts[best_query] += 1
        results[step, :] = counts
        step += 1
        if step >= max_steps:
            break
    return results[:step]


def generate_array_constructive_np(
    num_all_tokens: int,
    num_draft_steps: int,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
        
    out = np.zeros(batch_size, dtype=np.int32)

    num_full = min(num_all_tokens // num_draft_steps, batch_size)
    if num_full:
        idx = (np.arange(num_full) * batch_size) // num_full
        out[idx] = num_draft_steps
        num_all_tokens -= num_full * num_draft_steps

    if num_all_tokens:
        zero_pos = np.flatnonzero(out == 0)
        k = zero_pos.size
        counts = rng.multinomial(num_all_tokens,
                                 np.full(k, 1.0 / k, dtype=np.float64))
        out[zero_pos] = counts.astype(np.int32)
    return out
        
# def generate_array_constructive_np(num_all_tokens, num_draft_steps, batch_size):
#     num_draft_tokens_to_execute_per_batch = np.zeros(batch_size, dtype=np.int32)

#     num_full_queries = num_all_tokens // num_draft_steps
#     if num_full_queries:
#         num_draft_tokens_to_execute_per_batch[:num_full_queries] = num_draft_steps
#         num_all_tokens -= num_full_queries * num_draft_steps

#     remaining_queries = batch_size - num_full_queries
#     if remaining_queries:
#         arr = np.random.randint(0, remaining_queries, size=num_all_tokens)
#         query_indices, counts = np.unique(arr, return_counts=True)
#         query_indices += num_full_queries
#         num_draft_tokens_to_execute_per_batch[query_indices] = counts

#     # random
#     # arr = np.random.randint(1, batch_size, size=num_all_tokens)
#     # unique_values, counts = np.unique(arr, return_counts=True)
#     # num_draft_tokens_to_execute_per_batch[unique_values] = counts

    # return num_draft_tokens_to_execute_per_batch


if __name__ == "__main__":
    import time

    batch_size = 64
    num_tokens = 5
    vocab_size = 360000

    draft_token_ids_tensor = torch.randint(0, vocab_size, (batch_size, num_tokens), device='cuda')
    draft_probs = torch.rand(batch_size, num_tokens, vocab_size, device='cuda')
    verify_probs = torch.rand(batch_size, num_tokens+1, vocab_size, device='cuda')

    final_prob = torch.rand(batch_size, num_tokens, device='cuda')
    draft_probs = np.random.rand(batch_size, num_tokens)


    # baseline_func = incremental_select
    # new_func = incremental_select_numba
    # expected_tokens = compute_exp_len_np(draft_probs)
    # args = (expected_tokens,)


    profiled_batch_sizes = np.array([1] + list(range(4, batch_size+1, 4)))
    profiled_num_tokens = np.array([1] + list(range(4, batch_size*num_tokens+1, 4)))

    latency_profiles = {
        batch_size: {
            num_tokens: 1 for num_tokens in profiled_num_tokens
        } for batch_size in profiled_batch_sizes
    }

    closest_batch_sizes = profiled_batch_sizes[np.abs(profiled_batch_sizes - batch_size).argmin()].item()
    latency_profile = latency_profiles[closest_batch_sizes]


    keys, values = zip(*latency_profile.items())
    latency_profile_np = np.zeros(max(keys) + 1, dtype=float)
    latency_profile_np[list(keys)] = values
    print(latency_profile_np)

    baseline_func = get_num_tokens_to_verify
    new_func = get_num_tokens_to_verify_new
    args = (draft_probs, profiled_batch_sizes, latency_profile_np)




    for i in range(3):
        accept_prob1 = baseline_func(*args)

    start = time.time()
    for i in range(100):
        accept_prob1 = baseline_func(*args)
    torch.cuda.synchronize()
    end = time.time()
    print(f"{baseline_func.__name__} Time taken: {end - start} seconds")

    baseline_time = end - start

    for i in range(3):
        accept_prob1 = new_func(*args)
    
    start = time.time()
    for i in range(100):
        accept_prob1 = new_func(*args)
    torch.cuda.synchronize()
    end = time.time()
    print(f"{new_func.__name__} Time taken: {end - start} seconds")

    new_time = end - start

    print(f"{new_func.__name__} is {round(baseline_time / new_time, 2) - 1} times faster than {baseline_func.__name__}")





""" Legacy Functions



def find_best_draft_steps_per_batch(expected_effective_step, N):
    batch_size, num_draft_steps = expected_effective_step.shape
    best_steps = np.zeros(batch_size, dtype=np.int32)
    column_count = np.zeros(num_draft_steps, dtype=np.int32)

    delta = np.zeros((batch_size, num_draft_steps))
    delta[:, 0] = expected_effective_step[:, 0]
    for i in range(1, num_draft_steps):
        delta[:, i] = expected_effective_step[:, i] - expected_effective_step[:, i-1]
    
    candidates = []
    for b in range(batch_size):
        candidates.append((delta[b, 0].item(), b, 1))
        
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    allocated_draft_steps = 0
    
    while allocated_draft_steps < N and candidates:
        best_gain, b, step = candidates.pop(0)
        
        if step > num_draft_steps or allocated_draft_steps >= N:
            continue
        
        if column_count[step - 1] < step:
            best_steps[b] = step
            column_count[step - 1] += 1
            allocated_draft_steps += 1
            
            if step < num_draft_steps:
                candidates.append((delta[b, step].item(), b, step + 1))
                candidates.sort(key=lambda x: x[0], reverse=True)
    
    return best_steps

import heapq
def find_best_draft_steps_per_batch_new(expected_effective_step, N):
    batch_size, num_draft_steps = expected_effective_step.shape
    best_steps = np.zeros(batch_size, dtype=np.int32)
    column_count = np.zeros(num_draft_steps, dtype=np.int32)
    
    # 델타 계산을 벡터화: 첫번째 열은 그대로, 나머지는 차분 계산
    delta = np.empty_like(expected_effective_step)
    delta[:, 0] = expected_effective_step[:, 0]
    if num_draft_steps > 1:
        delta[:, 1:] = expected_effective_step[:, 1:] - expected_effective_step[:, :-1]
    
    # 최대 힙 사용: heapq는 기본적으로 최소 힙이므로 음수 값을 사용하여 최대 힙 효과를 냅니다.
    candidates = []
    for b in range(batch_size):
        # (이득, 배치 번호, 다음 step 번호) 형태에서 이득에 음수를 씌워 저장합니다.
        heapq.heappush(candidates, (-delta[b, 0].item(), b, 1))
    
    allocated_draft_steps = 0
    
    while allocated_draft_steps < N and candidates:
        neg_gain, b, step = heapq.heappop(candidates)
        
        if step > num_draft_steps:
            continue
        
        if column_count[step - 1] < step:
            best_steps[b] = step
            column_count[step - 1] += 1
            allocated_draft_steps += 1
            
            if step < num_draft_steps:
                heapq.heappush(candidates, (-delta[b, step].item(), b, step + 1))
    
    return best_steps

def filter_num_tokens_to_verify_np(num_expected_tokens: np.ndarray, threshold: float):
    B, T = num_expected_tokens.shape
    mask = num_expected_tokens >= threshold  # shape: (B, T)
    valid = mask.any(axis=1)  # shape: (B,)
    
    rev_argmax = np.argmax(mask, axis=1)
    last_index = T - rev_argmax - 1  # shape: (B,)
    last_index = np.where(valid, last_index, -1)
    
    num_tokens_to_verify = last_index + 1  # shape: (B,)
    clamped_indices = np.clip(last_index, 0, T-1)
    gathered = num_expected_tokens[np.arange(B), clamped_indices]
    last_element_values = np.where(valid, gathered, 0)
    
    return num_tokens_to_verify, last_element_values


@torch.jit.script
def filter_num_tokens_to_verify(num_expected_tokens: torch.Tensor, threshold: float):
    B, T = num_expected_tokens.shape
    mask = num_expected_tokens >= threshold
    valid = mask.any(dim=1)
    rev_argmax = mask.flip(dims=[1]).float().argmax(dim=1)
    last_index = T - rev_argmax - 1  # 뒤집었으므로 마지막 index 계산
    last_index = torch.where(valid, last_index, torch.full_like(last_index, -1))
    num_tokens_to_verify = last_index + 1
    gathered = num_expected_tokens.gather(1, last_index.clamp(min=0).unsqueeze(1)).squeeze(1)
    last_element_values = torch.where(valid, gathered, num_expected_tokens.new_zeros(()))
    return num_tokens_to_verify, last_element_values

    
"""
