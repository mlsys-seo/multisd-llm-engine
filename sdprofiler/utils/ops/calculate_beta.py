import torch
import triton
import triton.language as tl

@triton.jit
def beta_overlap_kernel(
    draft_ptr,                 # ptr to draft_probs (B×S×N)
    verify_ptr,                # ptr to verify_probs (B×S×N)
    token_ids_ptr,             # ptr to filtered token indices [N]
    N,                         # number of non-zero tokens
    out_ptr,                   # ptr to output β (B×S)
    B, S,
    stride_b_d, stride_bs_d, stride_bsn_d,
    stride_b_v, stride_bs_v, stride_bsn_v,
    stride_b_out, stride_bs_out,
    BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_N: tl.constexpr,
    num_warps: tl.constexpr
):
    # block indices
    b_idx = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    s_idx = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    b_mask = b_idx < B
    s_mask = s_idx < S

    # meshgrid for offsets
    b_mesh = b_idx[:, None]
    s_mesh = s_idx[None, :]
    base_d = b_mesh * stride_b_d + s_mesh * stride_bs_d
    base_v = b_mesh * stride_b_v + s_mesh * stride_bs_v

    acc = tl.zeros((BLOCK_B, BLOCK_S), dtype=tl.float32)

    # iterate over filtered token indices in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        # load token ids chunk
        nid = n_start + tl.arange(0, BLOCK_N)
        mask_n = nid < N
        token_id_chunk = tl.load(token_ids_ptr + nid, mask=mask_n, other=0)
        # compute offsets
        offs_d = base_d[:, :, None] + token_id_chunk[None, None, :] * stride_bsn_d
        offs_v = base_v[:, :, None] + token_id_chunk[None, None, :] * stride_bsn_v

        # mask for valid elements
        mask = b_mask[:, None, None] & s_mask[None, :, None] & mask_n[None, None, :]
        pd = tl.load(draft_ptr + offs_d, mask=mask, other=0.0)
        pv = tl.load(verify_ptr + offs_v, mask=mask, other=0.0)
        acc += tl.sum(tl.minimum(pd, pv), axis=2)

    # store result
    out_off = b_mesh * stride_b_out + s_mesh * stride_bs_out
    write_mask = b_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=write_mask)


def calculate_beta_triton(
    draft_probs: torch.Tensor,
    verify_probs: torch.Tensor,
    BLOCK_B=16, BLOCK_S=4, BLOCK_N=256, num_warps=4
) -> torch.Tensor:
    """
    Optimized Triton β 계산 래퍼.
    1) non-zero 토큰만 필터링해 커널에 전달
    2) 더 작은 블록/그리드로 높은 병렬성 확보
    3) num_warps constexpr로 런치 오버헤드 감소
    """
    # slice verify length
    if verify_probs.shape[1] != draft_probs.shape[1]:
        verify_probs = verify_probs[:, :-1, :]

    # filter non-zero token indices
    draft_nz = draft_probs.nonzero()[:, -1]
    verify_nz = verify_probs.nonzero()[:, -1]
    all_nz = torch.unique(torch.cat([draft_nz, verify_nz], dim=0))
    N = all_nz.numel()

    # select filtered vocab dims
    draft_f = draft_probs.index_select(2, all_nz)
    verify_f = verify_probs.index_select(2, all_nz)

    B, S, _ = draft_f.shape
    output = torch.empty((B, S), device=draft_probs.device, dtype=draft_probs.dtype)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(S, BLOCK_S))
    beta_overlap_kernel[grid](
        draft_f, verify_f,
        all_nz, N,
        output,
        B, S,
        draft_f.stride(0), draft_f.stride(1), draft_f.stride(2),
        verify_f.stride(0), verify_f.stride(1), verify_f.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_S=BLOCK_S, BLOCK_N=BLOCK_N,
        num_warps=num_warps
    )
    return output
