"""LayerNorm forward and single-pass backward kernels in Triton."""

KERNEL_REV = 1

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Forward kernel: computes LayerNorm per row and saves mean and rstd."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load input row in float32
    x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 1. Compute mean
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean_ptr + row_idx, mean)

    # 2. Compute variance and rstd
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd_ptr + row_idx, rstd)

    # 3. Normalize
    normed = x_centered * rstd
    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w
    if HAS_BIAS:
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b

    # Store output
    y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(y_ptrs, normed, mask=mask)


@triton.jit
def _layer_norm_bwd_kernel(
    dY_ptr,
    X_ptr,
    W_ptr,
    Mean_ptr,
    Rstd_ptr,
    dX_ptr,
    dW_ptr,
    dB_ptr,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    N,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Single-pass backward kernel: reads x and dy once, computes dx and accumulates dw/db in one pass."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 1. Load inputs exactly once from global memory
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dY_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row_idx).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row_idx).to(tl.float32)

    x_hat = (x - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        dy_w = dy * w
        # Accumulate dweight directly into buffer without second pass
        tl.atomic_add(dW_ptr + col_offsets, dy * x_hat, mask=mask)
    else:
        dy_w = dy

    if HAS_BIAS:
        # Accumulate dbias directly into buffer without second pass
        tl.atomic_add(dB_ptr + col_offsets, dy, mask=mask)

    # 2. Compute dx in the same pass
    # S1 = sum(dy_w), S2 = sum(dy_w * x_hat)
    s1 = tl.sum(dy_w, axis=0)
    s2 = tl.sum(dy_w * x_hat, axis=0)

    dx = rstd * (dy_w - (s1 / N) - (x_hat / N) * s2)

    # 3. Store dx
    tl.store(dX_ptr + row_idx * stride_dx_row + col_offsets, dx, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton forward kernel for LayerNorm."""
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)

    grid = (M,)
    _layer_norm_fwd_kernel[grid](
        x,
        y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        mean,
        rstd,
        x.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
        HAS_BIAS=(bias is not None),
        num_warps=num_warps,
    )
    return y, mean, rstd


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch single-pass Triton backward kernel for LayerNorm."""
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight, dtype=torch.float32) if weight is not None else None
    db = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)

    grid = (M,)
    _layer_norm_bwd_kernel[grid](
        dy,
        x,
        weight if weight is not None else x,
        mean,
        rstd,
        dx,
        dw if dw is not None else x,
        db if db is not None else x,
        dy.stride(0),
        x.stride(0),
        dx.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
        HAS_BIAS=(bias is not None),
        num_warps=num_warps,
    )

    if dw is not None:
        dw = dw.to(weight.dtype)
    if db is not None:
        db = db.to(bias.dtype)

    return dx, dw, db
