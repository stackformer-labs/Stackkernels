"""RMSNorm forward and backward kernels in Triton."""

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    Rstd_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Forward kernel: computes RMSNorm per row and saves reciprocal standard deviation (rstd)."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load input row
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute variance = mean(x^2) in float32
    variance = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Save rstd for backward pass
    tl.store(Rstd_ptr + row_idx, rstd)

    # Normalize
    normed = x * rstd
    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w

    # Store normalized output
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(y_ptrs, normed, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    dY_ptr,
    X_ptr,
    W_ptr,
    Rstd_ptr,
    dX_ptr,
    dW_ptr,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    N,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Single-pass backward kernel: reads x and dy once, computes dx and accumulates dw in one pass."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 1. Load inputs exactly once
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dY_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row_idx).to(tl.float32)

    x_hat = x * rstd

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        dy_w = dy * w
        # Accumulate dweight directly in this pass without re-reading from global memory
        tl.atomic_add(dW_ptr + col_offsets, dy * x_hat, mask=mask)
    else:
        dy_w = dy

    # 2. Compute dx in the same pass
    sum_dy_x_hat = tl.sum(dy_w * x_hat, axis=0)
    dx = rstd * (dy_w - (x_hat / N) * sum_dy_x_hat)

    # 3. Store dx
    tl.store(dX_ptr + row_idx * stride_dx_row + col_offsets, dx, mask=mask)


def rms_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Triton forward kernel for RMSNorm."""
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)

    grid = (M,)
    _rms_norm_fwd_kernel[grid](
        x,
        y,
        weight if weight is not None else x,
        rstd,
        x.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
        num_warps=num_warps,
    )
    return y, rstd


def rms_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch single-pass Triton backward kernel for RMSNorm."""
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight, dtype=torch.float32) if weight is not None else None

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)

    grid = (M,)
    _rms_norm_bwd_kernel[grid](
        dy,
        x,
        weight if weight is not None else x,
        rstd,
        dx,
        dw if dw is not None else x,
        dy.stride(0),
        x.stride(0),
        dx.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
        num_warps=num_warps,
    )

    if dw is not None:
        dw = dw.to(weight.dtype)

    return dx, dw
