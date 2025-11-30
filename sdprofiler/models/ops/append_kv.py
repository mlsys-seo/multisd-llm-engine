import torch
import triton
import triton.language as tl



@triton.jit
def _append_paged_kv_cache_kernel(
    k_data_ptr, v_data_ptr,

    # int32 tensors on page-level
    kv_indices_ptr,
    kv_indptr_ptr,
    kv_last_page_len_ptr,

    # int32 tensors on token-level
    batch_indices_ptr,
    positions_ptr,

    # float tensors (K, V, [nnz, num_heads, head_dim])
    append_k_ptr,
    append_v_ptr,

    # scalar values
    nnz,         # total number of tokens
    page_size,   # size of each page
    num_heads,   # number of heads
    head_dim,    # dimension of each head
    num_pages,   # total number of pages

    # K/V 캐시 strides (layout='NHD' => shape=[N,P,H,D])
    stride_np,   # stride on dim=0
    stride_ph,   # stride on dim=1
    stride_hd,   # stride on dim=2

    # append_k,append_v strides ([nnz, H, D])
    append_k_stride_n,
    append_k_stride_h,
    append_v_stride_n,
    append_v_stride_h,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
      - x방향(grid_m): ceil_div(nnz, BLOCK_M)
      - y방향(grid_n): ceil_div(num_heads * head_dim, BLOCK_N)

    한 블록이 [BLOCK_M, BLOCK_N] 크기의 (i, dim) 매트릭스를 처리하게끔.
    """
    # 1) token range for current block
    pid_m = tl.program_id(axis=0)
    token_start = pid_m * BLOCK_M
    token_offsets = token_start + tl.arange(0, BLOCK_M)  # shape=[BLOCK_M]
    
    # 유효 범위 계산
    nnz = tl.load(nnz)
    mask_i = token_offsets < nnz
    # token_offsets = tl.where(mask_i, token_offsets, 0) # make offsets safe

    batch_idx = tl.load(batch_indices_ptr + token_offsets, mask=mask_i, other=0)
    pos       = tl.load(positions_ptr     + token_offsets, mask=mask_i, other=0)

    start_p  = tl.load(kv_indptr_ptr        + batch_idx,       mask=mask_i, other=0)
    end_p    = tl.load(kv_indptr_ptr        + batch_idx + 1,   mask=mask_i, other=0)
    # last_len = tl.load(kv_last_page_len_ptr + batch_idx,       mask=mask_i, other=0)

    length = (end_p - start_p) * page_size
    valid  = (pos < length) & mask_i  # shape=[BLOCK_M], boolean

    # page_iter, entry_idx
    offset    = start_p*page_size + pos  # shape=[BLOCK_M]
    page_iter = offset // page_size      # shape=[BLOCK_M]
    entry_idx = offset % page_size       # shape=[BLOCK_M]

    #  - page_iter[i] -> kv_indices_ptr + page_iter[i]
    real_page = tl.load(kv_indices_ptr + page_iter, mask=valid, other=0)

    # 2) 이 블록이 처리할 범위 (가로방향)
    pid_n = tl.program_id(axis=1)
    dim_start = pid_n * BLOCK_N
    dim_offsets = dim_start + tl.arange(0, BLOCK_N)  # shape=[BLOCK_N]
    mask_dim = dim_offsets < (num_heads * head_dim)

    # head_idx, feat_idx
    head_idx = dim_offsets // head_dim
    feat_idx = dim_offsets % head_dim
    # shape=[BLOCK_N] 각각

    # 3)
    # - mask=[BLOCK_M], dim_mask=[BLOCK_N]
    # - 브로드캐스팅 -> mask_2d=[BLOCK_M, BLOCK_N]
    mask_2d = valid[:, None] & mask_dim[None, :]

    # 4) K/V 캐시 offset 계산
    # (각각 shape=[BLOCK_M] or [BLOCK_N]) -> broadcast -> [BLOCK_M, BLOCK_N]

    # pointer offset = real_page*stride_np + entry_idx*stride_ph + head_idx*stride_hd + feat_idx
    real_page_2d  = real_page[:, None]
    entry_idx_2d  = entry_idx[:, None]
    head_idx_2d   = head_idx[None, :]
    feat_idx_2d   = feat_idx[None, :]

    kv_offset_2d = (real_page_2d * stride_np) \
                 + (entry_idx_2d * stride_ph) \
                 + (head_idx_2d  * stride_hd) \
                 + feat_idx_2d
    # shape=[BLOCK_M, BLOCK_N]

    # 5) append_k, append_v에서 읽어올 오프셋 계산

    token_offsets_2d = token_offsets[:, None]
    # shape=[BLOCK_M, 1]

    k_offset_2d = (token_offsets_2d * append_k_stride_n) \
                + (head_idx_2d  * append_k_stride_h) \
                + feat_idx_2d
    # shape=[BLOCK_M, BLOCK_N]

    v_offset_2d = (token_offsets_2d * append_v_stride_n) \
                + (head_idx_2d  * append_v_stride_h) \
                + feat_idx_2d

    # 6) load + store
    k_in_2d = tl.load(
        append_k_ptr + k_offset_2d,
        mask=mask_2d,
        other=0.0
    )
    tl.store(
        k_data_ptr + kv_offset_2d,
        k_in_2d,
        mask=mask_2d
    )

    v_in_2d = tl.load(
        append_v_ptr + v_offset_2d,
        mask=mask_2d,
        other=0.0
    )
    tl.store(
        v_data_ptr + kv_offset_2d,
        v_in_2d,
        mask=mask_2d
    )

def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: torch.Tensor | tuple,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_all_tokens: torch.Tensor
):
    
    # examples
    # key_states: torch.Size([2, 2, 64])
    # value_states: torch.Size([2, 2, 64])
    # batch_indices: torch.Size([2])
    # positions: torch.Size([2])
    # paged_kv_indices: torch.Size([42])
    # paged_kv_indptr: torch.Size([2])
    # paged_kv_last_page_len: torch.Size([1])
    # past_key_value: torch.Size([160, 2, 16, 2, 64])

    """ Shape 정리
    - append_key, append_value: [nnz, num_heads, head_dim]
    - batch_indices, positions: [nnz]
    - paged_kv_cache:
        * Tensor 하나인 경우(K,V 합쳐짐) → shape=[num_pages, 2, page_size, num_heads, head_dim] (NHD 가정)
        * (k_data, v_data) 튜플 → k_data.shape = v_data.shape = [num_pages, page_size, num_heads, head_dim]
    - kv_indices: [num_pages]
    - kv_indptr:  [batch_size+1]
    - kv_last_page_len: [batch_size]
    - kv_layout: default 'NHD', 여기서는 [num_pages, page_size, num_heads, head_dim]로 가정
    """

    num_heads, head_dim = append_key.shape[-2], append_key.shape[-1]

    k_data = paged_kv_cache[:, 0, :, :, :]  # [num_pages, page_size, num_heads, head_dim]
    v_data = paged_kv_cache[:, 1, :, :, :]

    # layout='NHD' --> (k_data.shape) = [N, P, H, D]  
    num_pages = k_data.shape[0]
    page_size = k_data.shape[1]
    assert k_data.shape[2] == num_heads, f"k_data.shape[2] should be num_heads but got {k_data.shape[2]}"
    assert k_data.shape[3] == head_dim,  f"k_data.shape[3] should be head_dim but got {k_data.shape[3]}"

    #   e.g. for shape=[N,P,H,D], typical strides might be:
    #       k_data.stride(0) = P*H*D
    #       k_data.stride(1) = H*D
    #       k_data.stride(2) = D
    stride_np = k_data.stride(0)
    stride_ph = k_data.stride(1)
    stride_hd = k_data.stride(2)

    # 4) append_k / append_v strides [nnz, num_heads, head_dim]
    append_k_stride_n = num_heads * head_dim
    append_k_stride_h = head_dim
    append_v_stride_n = num_heads * head_dim
    append_v_stride_h = head_dim
    
    grid = lambda meta: (
        triton.cdiv(positions.shape[0], meta['BLOCK_M']),
        triton.cdiv(num_heads * head_dim, meta['BLOCK_N'])
    )

    _append_paged_kv_cache_kernel[grid](
        k_data, v_data,

        kv_indices, kv_indptr, kv_last_page_len,

        batch_indices, positions,

        append_key, append_value,

        num_all_tokens,
        page_size,
        num_heads,
        head_dim,
        num_pages,

        stride_np,
        stride_ph,
        stride_hd,

        append_k_stride_n,
        append_k_stride_h,
        append_v_stride_n,
        append_v_stride_h,

        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=4,
        num_stages=2
    )


if __name__ == "__main__":
    print("===== Test append_paged_kv_cache =====")

    batch_size  = 2
    num_pages   = 12 
    page_size   = 4
    num_heads   = 2
    head_dim    = 3

    # paged_kv_cache: shape=[num_pages, 2, page_size, num_heads, head_dim]
    kv_cache_shape = (num_pages, 2, page_size, num_heads, head_dim)
    paged_kv_cache = torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda')

    kv_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device='cuda')
    kv_indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device='cuda')
    kv_last_page_len = torch.tensor([0, 1], dtype=torch.int32, device='cuda')


    num_tokens = 5
    padding_tokens = 3
    batch_indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device='cuda')
    positions     = torch.tensor([1, 2, 6, 7, 8], dtype=torch.int32, device='cuda')

    nnz = num_tokens + padding_tokens
    append_key   = torch.randn((nnz, num_heads, head_dim), dtype=torch.bfloat16, device='cuda')
    append_value = torch.randn_like(append_key)

    print("=== Before append ===")
    print("paged_kv_cache[ :, 0, :, 0, 0 ] (Key part)\n", paged_kv_cache[:, 0, :, 0, 0])
    print("paged_kv_cache[ :, 1, :, 0, 0 ] (Value part)\n", paged_kv_cache[:, 1, :, 0, 0])

    append_paged_kv_cache(
        append_key=append_key,
        append_value=append_value,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_cache=paged_kv_cache,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        kv_layout='NHD',
        num_all_tokens=torch.tensor(num_tokens, dtype=torch.int32, device='cuda')
    )

    print("\n=== After append ===")
    for batch_idx in range(kv_indptr.shape[0]-1):
        print(f"batch_idx: {batch_idx}")
        print(f"kv indices per batch: {kv_indices[kv_indptr[batch_idx]:kv_indptr[batch_idx+1]]}")
        print(f"kv last page len per batch: {kv_last_page_len[batch_idx]}")
    print("Key part => paged_kv_cache[:, 0, :, 0, 0]")
    print(paged_kv_cache[:, 0, :, 0, 0])
    print("\nValue part => paged_kv_cache[:, 1, :, 0, 0]")
    print(paged_kv_cache[:, 1, :, 0, 0])


