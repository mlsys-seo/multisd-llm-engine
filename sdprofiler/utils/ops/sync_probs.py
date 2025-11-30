import torch
import triton
import triton.language as tl

@triton.jit
def sync_probs_kernel(
    draft_indptr_ptr, verify_indptr_ptr,
    draft_probs_ptr, verify_probs_ptr, sync_indptr_ptr,
    out_draft_probs_ptr, out_verify_probs_ptr,
    batch_size, num_draft_pos, draft_max_idx,
    num_verify_pos, verify_max_idx,
    BLOCK_SIZE: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)  # batch index
    d = tl.program_id(1)  # draft position index
    
    off_b_draft = b * num_draft_pos * draft_max_idx + d * draft_max_idx
    off_b_verify = b * num_verify_pos * verify_max_idx + d * verify_max_idx
    off_sync_indptr = b * num_verify_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)
    off_out_draft = b * num_draft_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)
    off_out_verify = b * num_verify_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)
    
    # set offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_d = offsets < draft_max_idx
    mask_v = offsets < verify_max_idx
    mask_ptr = offsets < (draft_max_idx + verify_max_idx)
    
    # load indices
    padding = tl.full((BLOCK_SIZE, ), -1, dtype=tl.int32)
    d_ptr = tl.load(draft_indptr_ptr + off_b_draft + offsets, mask=mask_d, other=padding)
    d_probs = tl.load(draft_probs_ptr + off_b_draft + offsets, mask=mask_d)
    v_ptr = tl.load(verify_indptr_ptr + off_b_verify + offsets, mask=mask_v, other=padding)
    v_probs = tl.load(verify_probs_ptr + off_b_verify + offsets, mask=mask_v)
    sync_idx = tl.load(sync_indptr_ptr + off_sync_indptr + offsets, mask=mask_ptr, other=padding)
    
    
    md_sync = sync_idx != -1
    len_sync = tl.sum(md_sync, 0)
    
    i = 0
    idx = tl.broadcast_to(i, (BLOCK_SIZE,))  
    while i < len_sync:
        idx_s = tl.gather(sync_idx, idx, axis=-1)
        v_one = tl.where(v_ptr == idx_s, v_probs, 0.0)
        v_one = tl.sort(v_one)
        v_one = tl.flip(v_one)
        
        d_one = tl.where(d_ptr == idx_s, d_probs, 0.0)
        d_one = tl.sort(d_one)
        d_one = tl.flip(d_one)
        
        # 하나만 복사
        mask = (offsets < 1)
        tl.store(out_verify_probs_ptr + off_out_verify + offsets + i, v_one, mask=mask)
        tl.store(out_draft_probs_ptr + off_out_draft + offsets + i, d_one, mask=mask)
        
        i += 1
        idx = idx + 1
    

@triton.jit
def sync_indptr_kernel(
    draft_indptr_ptr, verify_indptr_ptr, out_indptr_ptr,
    batch_size, num_draft_pos, draft_max_idx,
    num_verify_pos, verify_max_idx,
    BLOCK_SIZE: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)  # batch index
    d = tl.program_id(1)  # draft position index

    # pointers offsets
    # shapes: draft_indptr[b, d, i]
    off_b_draft = b * num_draft_pos * draft_max_idx + d * draft_max_idx
    off_b_verify = b * num_verify_pos * verify_max_idx + d * verify_max_idx
    off_out_indptr = b * num_verify_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)
    off_out_draft = b * num_draft_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)
    off_out_verify = b * num_verify_pos * (draft_max_idx + verify_max_idx) + d * (draft_max_idx + verify_max_idx)

    # set offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_d = offsets < draft_max_idx
    mask_v = offsets < verify_max_idx
    
    # load indices
    padding = tl.full((BLOCK_SIZE, ), -1, dtype=tl.int32)
    idx_d = tl.load(draft_indptr_ptr + off_b_draft + offsets, mask=mask_d, other=padding)
    idx_v = tl.load(verify_indptr_ptr + off_b_verify + offsets, mask=mask_v, other=padding)
    
    # masks
    m_d = idx_d != -1

    # merge unique sorted
    # allocate temp arrays in register
    len_d = tl.sum(m_d, 0)
    
    sorted_idx_d = tl.sort(idx_d)
    sorted_idx_d = tl.flip(sorted_idx_d)
    
    if len_d > 0:
        mask = offsets < len_d
        tl.store(out_indptr_ptr + off_out_indptr + offsets, sorted_idx_d, mask=mask)
    
    i = 0
    idx = tl.broadcast_to(i, (BLOCK_SIZE,))  
    while i < len_d:
        d_value = tl.gather(sorted_idx_d, idx, axis=-1)
        idx_v = tl.where(d_value == idx_v, -1, idx_v)
        i += 1
        idx = idx + 1
        
    sorted_idx_v = tl.sort(idx_v)
    sorted_idx_v = tl.flip(sorted_idx_v)
    
    m_v = sorted_idx_v != -1
    len_v = tl.sum(m_v, 0)
    if len_v > 0:
        mask = (offsets + len_d < len_d + len_v)
        tl.store(out_indptr_ptr + off_out_indptr + offsets + len_d, sorted_idx_v, mask=mask)
   

# Python wrapper
def sync_probs_df(
    draft_probs, verify_probs,
    draft_probs_indptr, verify_probs_indptr
):
    B, D, C = draft_probs_indptr.shape
    _, E, V = verify_probs_indptr.shape
    M = C + V
    
    # allocate outputs
    out_indptr = torch.full((B, E, M), -1, dtype=draft_probs_indptr.dtype, device=draft_probs_indptr.device)
    out_draft = torch.zeros((B, D, M), dtype=draft_probs.dtype, device=draft_probs.device)
    out_verify = torch.zeros((B, E, M), dtype=verify_probs.dtype, device=verify_probs.device)
    
    BLOCK_SIZE = triton.next_power_of_2(M)
    m = triton.next_power_of_2(M)

    grid = (B, D)
    sync_indptr_kernel[
        grid
    ](
        draft_probs_indptr, verify_probs_indptr, out_indptr,
        B, D, C, E, V,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=32, num_stages=1
    )
    
    sync_probs_kernel[
        grid
    ](
        draft_probs_indptr, verify_probs_indptr,
        draft_probs, verify_probs, out_indptr,
        out_draft, out_verify,
        B, D, C, E, V,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=32, num_stages=1
    )

    out_indptr[:, D:, :V] = verify_probs_indptr[:, D:, :].clone()
    out_verify[:, D:, :V] = verify_probs[:, D:, :].clone()
    out_draft_indptr = out_indptr.clone()
    return out_draft, out_draft_indptr, out_verify, out_indptr

def exist_method(
    draft_probs_origin, verify_probs_origin,
    draft_probs_indptr_origin, verify_probs_indptr_origin
):
    A, B, C = draft_probs_indptr_origin.shape
    _, E, D = verify_probs_indptr_origin.shape
    
    draft_probs_indptr = draft_probs_indptr_origin
    verify_probs_indptr = verify_probs_indptr_origin[:, :B, :]   # in fact, I should cut each row, since token lengths are different per request in oracle step
    draft_probs = draft_probs_origin
    verify_probs = verify_probs_origin[:, :B, :]
    
    mask1 = draft_probs_indptr != -1
    mask2 = verify_probs_indptr != -1
        
    new_probs_indptr = torch.full((A, B, C+D), -1, dtype=draft_probs_indptr.dtype, device=draft_probs_indptr.device)
    max_len = 0
    for a in range(A):  # TODO Maybe can remove loop? this is because of torch.unique function doesn't support individual multi-dim operation
        for b in range(B):
            indices1 = draft_probs_indptr[a, b, mask1[a, b]]
            indices2 = verify_probs_indptr[a, b, mask2[a, b]]
            if indices1.numel() == 0 and indices2.numel() == 0:
                continue
            unique_indices = torch.unique(torch.cat([indices1, indices2], dim=-1), sorted=True)
                    
            new_probs_indptr[a, b, :len(unique_indices)] = unique_indices
            max_len = max(max_len, len(unique_indices))
    new_probs_indptr = new_probs_indptr[:, :, :max_len].contiguous()

    new_probs1_value = torch.zeros((A, B, max_len), dtype=draft_probs.dtype, device=draft_probs.device)
    new_probs2_value = torch.zeros((A, E, max_len), dtype=verify_probs.dtype, device=verify_probs.device)
        
    map1 = torch.full_like(draft_probs_indptr, -1, dtype=torch.int64, device=draft_probs_indptr.device)
    map2 = torch.full_like(verify_probs_indptr, -1, dtype=torch.int64, device=verify_probs_indptr.device)
        
    matches1 = (new_probs_indptr.unsqueeze(-1) == draft_probs_indptr.unsqueeze(-2)) & (draft_probs_indptr.unsqueeze(-2) != -1)
    matches2 = (new_probs_indptr.unsqueeze(-1) == verify_probs_indptr.unsqueeze(-2)) & (verify_probs_indptr.unsqueeze(-2) != -1)

    matches1_index = matches1.nonzero()
    matches2_index = matches2.nonzero()
        
    map1[matches1_index[:, 0], matches1_index[:, 1], matches1_index[:, 3]] = matches1_index[:, 2]
    map2[matches2_index[:, 0], matches2_index[:, 1], matches2_index[:, 3]] = matches2_index[:, 2]
        
    mask1_index = mask1.nonzero()
    mask2_index = mask2.nonzero()
        
    new_probs1_value[mask1_index[:, 0], mask1_index[:, 1], map1[mask1]] = draft_probs[mask1]
    new_probs2_value[mask2_index[:, 0], mask2_index[:, 1], map2[mask2]] = verify_probs[mask2]

    new_probs_indptr = torch.nn.functional.pad(new_probs_indptr, (0, 0, 0, E-B, 0, 0), mode='constant', value=-1)
        
    if verify_probs_indptr_origin.shape[-1] > max_len:
        new_probs_indptr[:, B:, :] = verify_probs_indptr_origin[:, B:, :max_len]
        new_probs2_value[:, B:, :] = verify_probs_origin[:, B:, :max_len]
    else:
        new_probs_indptr[:, B:, :D] = verify_probs_indptr_origin[:, B:, :]
        new_probs2_value[:, B:, :D] = verify_probs_origin[:, B:, :]

    draft_probs_indptr_origin = new_probs_indptr.clone()
    verify_probs_indptr_origin = new_probs_indptr.clone()
    draft_probs_origin = new_probs1_value
    verify_probs_origin = new_probs2_value
    
    return draft_probs_origin, draft_probs_indptr_origin, verify_probs_origin, verify_probs_indptr_origin


# Test code
def test_sync_probs():
    import random
    B, D, C, E, V = 2, 4, 5, 5, 6
    
    dpi = torch.full((B, D, C), -1, dtype=torch.int32, device="cuda:0")  # 먼저 -1로 채움
    dp = torch.zeros_like(dpi, dtype=torch.float32, device="cuda:0")

    for i in range(B):
        for j in range(D):
            num_valid = random.randint(1, C)  # 0~5개의 유효한 값
            valid_values = random.sample(range(C), num_valid)  # 중복 없는 0~4에서 선택
            dpi[i, j, :num_valid] = torch.tensor(valid_values, dtype=torch.int32, device="cuda:0")
            
    # 마스킹 및 정규화된 난수 생성
    for i in range(B):
        for j in range(D):
            mask = dpi[i, j] != -1
            num_valid = mask.sum().item()
            if num_valid > 0:
                random_vals = torch.rand(num_valid, device="cuda:0")
                normalized_vals = random_vals / random_vals.sum()  # 합이 1이 되도록 정규화
                dp[i, j, mask] = normalized_vals
                
    vpi = torch.full((B, E, V), -1, dtype=torch.int32, device="cuda:0")  # 먼저 -1로 채움
    vp = torch.zeros_like(vpi, dtype=torch.float32, device="cuda:0")
    
    for i in range(B):
        for j in range(E):
            num_valid = random.randint(1, V)  # 0~5개의 유효한 값
            valid_values = random.sample(range(V), num_valid)  # 중복 없는 0~4에서 선택
            vpi[i, j, :num_valid] = torch.tensor(valid_values, dtype=torch.int32, device="cuda:0")
            
    # 마스킹 및 정규화된 난수 생성
    for i in range(B):
        for j in range(E):
            mask = vpi[i, j] != -1
            num_valid = mask.sum().item()
            if num_valid > 0:
                random_vals = torch.rand(num_valid, device="cuda:0")
                normalized_vals = random_vals / random_vals.sum()  # 합이 1이 되도록 정규화
                vp[i, j, mask] = normalized_vals
    
    print(dpi)
    print(vpi)
    print(dp)
    print(vp)

    out_dp, out_dpi, out_vp, out_vpi = sync_probs_df(dp, vp, dpi, vpi)
    torch.cuda.synchronize()
    out_dp, out_dpi, out_vp, out_vpi = exist_method(dp, vp, dpi, vpi)
    
    print(out_dp)
    print(out_vp)
    print(out_dpi)
    print(out_vpi)
    print("sync_probs_df test passed!")

if __name__ == '__main__':
    test_sync_probs()
