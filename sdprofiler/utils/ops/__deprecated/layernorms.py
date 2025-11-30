import torch
import triton
import triton.language as tl

from .utils import get_block_size

@triton.jit
def rmsnorm_forward_kernel(
    hidden_states_ptr,  # (batch_size, hidden_size)
    weight_ptr,         # (hidden_size,)
    output_ptr,         # (batch_size, hidden_size)
    eps,                # float
    B,                  # batch_size
    H,                  # hidden_size
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: 
      - mean of squares = sum(x^2)/H
      - normalization = x * 1/sqrt(mean_of_squares + eps)
      - output = weight * normalized_x
    """
    b_idx = tl.program_id(0) # program_id(0) : row index (batch idx)
    row_start = b_idx * H

    offs = tl.arange(0, BLOCK_SIZE)
    col_idx = offs

    # 모든 col_idx가 hidden_size 범위 밖일 수 있으므로 마스킹
    mask = col_idx < H

    # ----------------------------
    # 1) sum of x^2 (for variance)
    # ----------------------------
    sq_sum = tl.zeros([1], dtype=tl.float32)

    for i in range(0, (H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        current_col_idx = col_idx + i * BLOCK_SIZE
        valid = current_col_idx < H
        x = tl.where(
            valid,
            tl.load(hidden_states_ptr + row_start + current_col_idx, mask=valid),
            0.0
        )
        x_fp32 = x.to(tl.float32)
        sq_sum += tl.sum(x_fp32 * x_fp32, axis=0)

    mean_of_squares = sq_sum / H
    inv_rms = 1.0 / tl.sqrt(mean_of_squares + eps)

    # ----------------------------
    # 2) normalization & multiply
    # ----------------------------
    # 다시 한 번 (H + BLOCK_SIZE - 1) // BLOCK_SIZE 만큼 순회하며 결과를 저장
    for i in range(0, (H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        current_col_idx = col_idx + i * BLOCK_SIZE
        valid = current_col_idx < H
        x = tl.where(
            valid,
            tl.load(hidden_states_ptr + row_start + current_col_idx, mask=valid),
            0.0
        )
        # weight
        w = tl.where(
            valid,
            tl.load(weight_ptr + current_col_idx, mask=valid),
            0.0
        )
        # normalization
        x_fp32 = x.to(tl.float32)
        normed = x_fp32 * inv_rms
        # output = weight * normed
        out = w * normed
        # 저장 (dtype은 원본에 맞춰서 조정 가능)
        tl.store(output_ptr + row_start + current_col_idx, out, mask=valid)


def rmsnorm_forward(hidden_states: torch.Tensor,
                    weight: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Python wrapper for rmsnorm_forward_kernel.
    hidden_states: (batch_size, hidden_size)
    weight: (hidden_size,)
    """
    assert hidden_states.dim() == 2, "hidden_states should be [batch_size, hidden_size]"
    B, H = hidden_states.shape
    
    output = torch.empty_like(hidden_states)
    grid = (B,)

    block_size = get_block_size(H)

    rmsnorm_forward_kernel[grid](
        hidden_states, 
        weight, 
        output, 
        eps,
        B, 
        H, 
        BLOCK_SIZE=block_size
    )
    return output


# ======================
# Add + RMSNorm (Fused)
# ======================

@triton.jit
def add_fused_rmsnorm_forward_kernel(
    hidden_states_ptr,  # half  (B, H)
    residual_ptr,       # half  (B, H)
    weight_ptr,         # half  (H,)
    output_ptr,         # half  (B, H)
    eps,                # float
    B,                  # batch_size
    H,                  # hidden_size
    BLOCK_SIZE: tl.constexpr
):
    """
    Add + RMSNorm (baseline ver.):
      1) (x + residual)을 half로 더하고, half로 저장
      2) sum of squares는 float32로 누적
      3) inv_rms = 1 / sqrt(mean_of_squares + eps)
      4) output = weight * (x * inv_rms)  (x는 half에서 float32로 캐스팅)
    """
    b_idx = tl.program_id(0)
    row_start = b_idx * H

    offs = tl.arange(0, BLOCK_SIZE)
    col_idx = offs
    mask = col_idx < H

    sq_sum = tl.zeros([1], dtype=tl.float32)

    # hidden_size가 BLOCK_SIZE보다 클 수 있으니 여러 번 반복
    for i in range(0, (H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        current_col_idx = col_idx + i * BLOCK_SIZE
        valid = current_col_idx < H

        x_half = tl.load(hidden_states_ptr + row_start + current_col_idx, mask=valid)
        r_half = tl.load(residual_ptr + row_start + current_col_idx, mask=valid)

        x_add_half = x_half + r_half  # half

        tl.store(residual_ptr + row_start + current_col_idx, x_add_half, mask=valid)

        # sum of squares 계산은 float32
        x_fp32 = x_add_half.to(tl.float32)
        sq_sum += tl.sum(x_fp32 * x_fp32, axis=0)

    mean_of_squares = sq_sum / H
    inv_rms = 1.0 / tl.sqrt(mean_of_squares + eps)

    # 2) (x) half -> float32 로드
    for i in range(0, (H + BLOCK_SIZE - 1) // BLOCK_SIZE):
        current_col_idx = col_idx + i * BLOCK_SIZE
        valid = current_col_idx < H

        # half로 로드
        x_half = tl.load(residual_ptr + row_start + current_col_idx, mask=valid)
        w_half = tl.load(weight_ptr + current_col_idx, mask=valid)

        # float32로 변환 후 RMSNorm
        x_fp32 = x_half.to(tl.float32)
        w_fp32 = w_half.to(tl.float32)
        normed_fp32 = x_fp32 * inv_rms

        out_fp32 = normed_fp32 * w_fp32

        out_half = out_fp32.to(tl.float16)
        tl.store(output_ptr + row_start + current_col_idx, out_half, mask=valid)


def add_fused_rmsnorm_forward(x: torch.Tensor,
                              residual: torch.Tensor,
                              weight: torch.Tensor,
                              eps: float = 1e-6,
                              block_size: int = 128) -> torch.Tensor:
    """
    AddFusedRMSNorm:  out = RMSNorm(x + residual) (baseline 버전)
    (여기서는 '덧셈'을 half 정밀도로 수행)

    x       : (B, H) half
    residual: (B, H) half
    weight  : (H)    half
    eps     : float
    """
    assert x.ndim == 2 and x.shape == residual.shape
    B, H = x.shape

    block_size = get_block_size(H)
    out = torch.empty_like(x)  # half output
    grid = (B,)

    add_fused_rmsnorm_forward_kernel[grid](
        x,         # half
        residual,  # half
        weight,    # half
        out,       # half
        eps,
        B,
        H,
        BLOCK_SIZE=block_size
    )
    return out

if __name__ == "__main__":

    import torch.nn as nn
    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, dtype=torch.float16):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
        
    dtype = torch.float16
    batch_size, hidden_size, eps = 4, 16, 1e-6

    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
    rmsnorm = Qwen2RMSNorm(hidden_size, eps=eps).to("cuda")
    weight = rmsnorm.weight.data

    out_ref = rmsnorm(x)
    out_triton = rmsnorm_forward(x, weight, eps)
    print("RMSNorm (Triton) vs PyTorch:", (out_triton - out_ref).abs().max().item())
    print("RMSNorm (Triton) vs PyTorch Allclose:", torch.allclose(out_triton, out_ref))


    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
    residual = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
    residual_ref = x + residual
    out_ref = rmsnorm(x + residual)
    out_fused = add_fused_rmsnorm_forward(x, residual, weight, eps)    
    print("AddFusedRMSNorm (Triton) vs PyTorch:", (out_fused - out_ref).abs().max().item())
    print("AddFusedRMSNorm (Triton) vs PyTorch Allclose:", torch.allclose(out_fused, out_ref))
    print("residual (Triton) vs residual (PyTorch):", torch.allclose(residual, residual_ref))
