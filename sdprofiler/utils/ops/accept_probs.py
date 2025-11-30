import torch
import triton
import triton.language as tl

@triton.jit
def accept_prob_kernel(
    draft_ptr, verify_ptr, token_ptr, out_ptr,
    B, S, V,
    stride_b_draft, stride_bs_draft, stride_bsv_draft,
    stride_b_verify, stride_bs_verify, stride_bsv_verify,
    stride_b_token, stride_bs_token,
    stride_b_out, stride_bs_out,
    BLOCK_B: tl.constexpr, BLOCK_S: tl.constexpr
):
    # block offsets
    b_idx = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    s_idx = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)

    # meshgrid
    b_mesh = b_idx[:, None]
    s_mesh = s_idx[None, :]

    # token indices
    token_offset = b_mesh * stride_b_token + s_mesh * stride_bs_token
    token = tl.load(token_ptr + token_offset)

    # valid mask
    mask = (b_mesh < B) & (s_mesh < S) & (token >= 0) & (token < V)

    # compute offsets
    draft_offset = b_mesh * stride_b_draft + s_mesh * stride_bs_draft + token
    verify_offset = b_mesh * stride_b_verify + s_mesh * stride_bs_verify + token

    # load
    draft = tl.load(draft_ptr + draft_offset, mask=mask, other=0.0)
    verify = tl.load(verify_ptr + verify_offset, mask=mask, other=0.0)

    # compute accept prob
    ratio = tl.where(draft == 0.0, 0.0, verify / draft)
    accept = tl.minimum(tl.maximum(ratio, 0.0), 1.0) * mask

    # store
    out_offset = b_mesh * stride_b_out + s_mesh * stride_bs_out
    tl.store(out_ptr + out_offset, accept)


def calculate_accept_prob(draft_probs, verify_probs, token_idx, BLOCK_B=64, BLOCK_S=64):
    """
    Compute per-position accept probabilities.
    """
    # align lengths
    if verify_probs.size(1) != draft_probs.size(1):
        verify_probs = verify_probs[:, : draft_probs.size(1), :]

    B, S, V = draft_probs.shape
    out = torch.empty((B, S), dtype=torch.float32, device=draft_probs.device)

    # launch Triton kernel
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(S, BLOCK_S))
    accept_prob_kernel[grid](
        # positional args
        draft_probs, verify_probs, token_idx, out,
        B, S, V,
        draft_probs.stride(0), draft_probs.stride(1), draft_probs.stride(2),
        verify_probs.stride(0), verify_probs.stride(1), verify_probs.stride(2),
        token_idx.stride(0), token_idx.stride(1),
        out.stride(0), out.stride(1),
        # **keyword-only** constexpr args
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S
    )
    return out
