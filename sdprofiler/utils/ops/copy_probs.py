import torch
import triton
import triton.language as tl

# Triton kernel: copy values from a ragged [num_all_tokens, vocab_size] tensor into
# a dense [batch_size, num_max_tokens, vocab_size] tensor according to indptr.
# ragged_probs: float32, shape [num_all_tokens, vocab_size]
# probs_indptr: int32, shape [batch_size + 1]
# probs: float32, shape [batch_size, num_max_tokens, vocab_size], pre-zeroed

@triton.jit
def _copy_ragged(
    ragged_ptr,               # pointer to ragged_probs
    probs_ptr,                # pointer to output probs
    indptr_ptr,               # pointer to probs_indptr
    num_all_tokens,           # total rows in ragged_probs
    batch_size,               # number of requests (indptr.numel()-1)
    num_max_tokens,           # max tokens per request
    vocab_size,               # vocab dimension
    BLOCK_SIZE: tl.constexpr  # size of vocab-block per program
):
    # program ids
    bid = tl.program_id(axis=0)  # batch index [0..batch_size)
    tid = tl.program_id(axis=1)  # token position [0..num_max_tokens)
    bid_v = tl.program_id(axis=2)  # vocab-block index

    # Only process valid batch entries
    # Load start and end pointers
    start = tl.load(indptr_ptr + bid)
    end = tl.load(indptr_ptr + bid + 1)
    length = end - start

    # Check if this token position exists
    in_range = tid < length

    # Compute actual row, clamp out-of-range to 0 to avoid illegal pointers
    row = start + tid
    row_clamped = tl.where(in_range, row, 0)

    # Compute vocab offsets within block
    offs = bid_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_vocab = offs < vocab_size
    mask = in_range & valid_vocab

    # Compute base pointers
    ragged_row_ptr = ragged_ptr + row_clamped * vocab_size
    out_row_ptr = probs_ptr + (bid * num_max_tokens + tid) * vocab_size

    # Masked load and store
    vals = tl.load(ragged_row_ptr + offs, mask=mask)
    tl.store(out_row_ptr + offs, vals, mask=mask)


def copy_ragged_probs(
    ragged_probs: torch.Tensor,
    probs_indptr: torch.Tensor,
    probs: torch.Tensor,
    block_size: int = 128,
    num_warps: int = 4
):
    """
    Copy ragged_probs into the dense probs tensor safely.

    Args:
        ragged_probs (Tensor): [num_all_tokens, vocab_size]
        probs_indptr (Tensor): [batch_size + 1]
        probs (Tensor): [max_batch, num_max_tokens, vocab_size]
    """
    assert ragged_probs.is_contiguous()
    assert probs.is_contiguous()
    # assert probs_indptr.dtype == torch.int32
    probs_indptr = probs_indptr.to(probs.device)

    # Derive dimensions
    num_all_tokens, vocab_size = ragged_probs.shape
    batch_size = probs_indptr.numel() - 1
    num_max_tokens = probs.shape[1]

    # Sanity check
    assert batch_size <= probs.shape[0], (
        f"probs tensor batch dim {probs.shape[0]} < required {batch_size}")

    num_vocab_blocks = (vocab_size + block_size - 1) // block_size
    grid = (batch_size, num_max_tokens, num_vocab_blocks)

    _copy_ragged[grid](
        ragged_probs,
        probs,
        probs_indptr,
        num_all_tokens,
        batch_size,
        num_max_tokens,
        vocab_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
