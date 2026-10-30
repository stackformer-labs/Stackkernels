"""Fused residual add + normalization (RMSNorm / LayerNorm) forward and single-pass backward kernels."""

from typing import Literal

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


# --- Fused Add + RMSNorm ---

@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    X_ptr,
    Res_ptr,
    Y_ptr,
    Res_out_ptr,
    W_ptr,
    Rstd_ptr,
    stride_x_row,
    stride_res_row,
    stride_y_row,
    stride_res_out_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """Fused add + RMSNorm forward kernel."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 1. Load X and residual in float32
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res_ptr + row_idx * stride_res_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 2. Add residual
    accum = x + res
    tl.store(Res_out_ptr + row_idx * stride_res_out_row + col_offsets, accum, mask=mask)

    # 3. Compute variance = mean(accum^2)
    var = tl.sum(accum * accum, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd_ptr + row_idx, rstd)

    # 4. Normalize
    normed = accum * rstd
    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w

    # Store normalized output
    tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, normed, mask=mask)


@triton.jit
def _fused_add_rms_norm_bwd_kernel(
    dY_ptr,
    dRes_out_ptr,
    Res_out_ptr,
    W_ptr,
    Rstd_ptr,
    dX_ptr,
    dRes_ptr,
    dW_ptr,
    stride_dy_row,
    stride_dres_out_row,
    stride_res_out_row,
    stride_dx_row,
    stride_dres_row,
    N,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
):
    """Single-pass backward for fused add + RMSNorm."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load inputs once
    dy = tl.load(dY_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    res_out = tl.load(Res_out_ptr + row_idx * stride_res_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row_idx).to(tl.float32)

    x_hat = res_out * rstd

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        dy_w = dy * w
        tl.atomic_add(dW_ptr + col_offsets, dy * x_hat, mask=mask)
    else:
        dy_w = dy

    sum_dy_x_hat = tl.sum(dy_w * x_hat, axis=0)
    dres_from_norm = rstd * (dy_w - (x_hat / N) * sum_dy_x_hat)

    # Combine with incoming residual gradient if present
    if HAS_DRES_OUT:
        dres_incoming = tl.load(
            dRes_out_ptr + row_idx * stride_dres_out_row + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        d_total = dres_from_norm + dres_incoming
    else:
        d_total = dres_from_norm

    # Since res_out = x + residual: dX = d_total, dRes = d_total
    tl.store(dX_ptr + row_idx * stride_dx_row + col_offsets, d_total, mask=mask)
    tl.store(dRes_ptr + row_idx * stride_dres_row + col_offsets, d_total, mask=mask)


# --- Fused Add + LayerNorm ---

@triton.jit
def _fused_add_layer_norm_fwd_kernel(
    X_ptr,
    Res_ptr,
    Y_ptr,
    Res_out_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride_x_row,
    stride_res_row,
    stride_y_row,
    stride_res_out_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused add + LayerNorm forward kernel."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load X and residual in float32
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res_ptr + row_idx * stride_res_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Residual addition
    accum = x + res
    tl.store(Res_out_ptr + row_idx * stride_res_out_row + col_offsets, accum, mask=mask)

    # Mean and variance
    mean = tl.sum(accum, axis=0) / N
    tl.store(Mean_ptr + row_idx, mean)

    accum_centered = tl.where(mask, accum - mean, 0.0)
    var = tl.sum(accum_centered * accum_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd_ptr + row_idx, rstd)

    # Normalize
    normed = accum_centered * rstd
    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        normed = normed * w
    if HAS_BIAS:
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b

    tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, normed, mask=mask)


@triton.jit
def _fused_add_layer_norm_bwd_kernel(
    dY_ptr,
    dRes_out_ptr,
    Res_out_ptr,
    W_ptr,
    Mean_ptr,
    Rstd_ptr,
    dX_ptr,
    dRes_ptr,
    dW_ptr,
    dB_ptr,
    stride_dy_row,
    stride_dres_out_row,
    stride_res_out_row,
    stride_dx_row,
    stride_dres_row,
    N,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
):
    """Single-pass backward for fused add + LayerNorm."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    dy = tl.load(dY_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    res_out = tl.load(Res_out_ptr + row_idx * stride_res_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean_ptr + row_idx).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row_idx).to(tl.float32)

    x_hat = (res_out - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        dy_w = dy * w
        tl.atomic_add(dW_ptr + col_offsets, dy * x_hat, mask=mask)
    else:
        dy_w = dy

    if HAS_BIAS:
        tl.atomic_add(dB_ptr + col_offsets, dy, mask=mask)

    s1 = tl.sum(dy_w, axis=0)
    s2 = tl.sum(dy_w * x_hat, axis=0)
    dres_from_norm = rstd * (dy_w - (s1 / N) - (x_hat / N) * s2)

    if HAS_DRES_OUT:
        dres_incoming = tl.load(
            dRes_out_ptr + row_idx * stride_dres_out_row + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        d_total = dres_from_norm + dres_incoming
    else:
        d_total = dres_from_norm

    tl.store(dX_ptr + row_idx * stride_dx_row + col_offsets, d_total, mask=mask)
    tl.store(dRes_ptr + row_idx * stride_dres_row + col_offsets, d_total, mask=mask)


# --- Python Launchers ---

def fused_add_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Launch Triton forward kernel for fused add + norm.

    Returns:
        tuple (y, res_out, rstd, mean_or_None)
    """
    M, N = x.shape
    y = torch.empty_like(x)
    res_out = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (M,)

    if norm_type == "rms_norm":
        _fused_add_rms_norm_fwd_kernel[grid](
            x,
            residual,
            y,
            res_out,
            weight if weight is not None else x,
            rstd,
            x.stride(0),
            residual.stride(0),
            y.stride(0),
            res_out.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=(weight is not None),
            num_warps=num_warps,
        )
        return y, res_out, rstd, None
    elif norm_type == "layer_norm":
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        _fused_add_layer_norm_fwd_kernel[grid](
            x,
            residual,
            y,
            res_out,
            weight if weight is not None else x,
            bias if bias is not None else x,
            mean,
            rstd,
            x.stride(0),
            residual.stride(0),
            y.stride(0),
            res_out.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
            num_warps=num_warps,
        )
        return y, res_out, rstd, mean
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


def fused_add_norm_bwd(
    dy: torch.Tensor,
    dres_out: torch.Tensor | None,
    res_out: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    rstd: torch.Tensor,
    mean: torch.Tensor | None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch single-pass Triton backward kernel for fused add + norm.

    Returns:
        tuple (dx, dresidual, dw, db)
    """
    M, N = res_out.shape
    dx = torch.empty_like(res_out)
    dresidual = torch.empty_like(res_out)
    dw = torch.zeros_like(weight, dtype=torch.float32) if weight is not None else None
    db = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (M,)

    if norm_type == "rms_norm":
        _fused_add_rms_norm_bwd_kernel[grid](
            dy,
            dres_out if dres_out is not None else dy,
            res_out,
            weight if weight is not None else res_out,
            rstd,
            dx,
            dresidual,
            dw if dw is not None else res_out,
            dy.stride(0),
            dres_out.stride(0) if dres_out is not None else 0,
            res_out.stride(0),
            dx.stride(0),
            dresidual.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=(weight is not None),
            HAS_DRES_OUT=(dres_out is not None),
            num_warps=num_warps,
        )
    elif norm_type == "layer_norm":
        _fused_add_layer_norm_bwd_kernel[grid](
            dy,
            dres_out if dres_out is not None else dy,
            res_out,
            weight if weight is not None else res_out,
            mean if mean is not None else res_out,
            rstd,
            dx,
            dresidual,
            dw if dw is not None else res_out,
            db if db is not None else res_out,
            dy.stride(0),
            dres_out.stride(0) if dres_out is not None else 0,
            res_out.stride(0),
            dx.stride(0),
            dresidual.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
            HAS_DRES_OUT=(dres_out is not None),
            num_warps=num_warps,
        )
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    if dw is not None:
        dw = dw.to(weight.dtype)
    if db is not None:
        db = db.to(bias.dtype)

    return dx, dresidual, dw, db
