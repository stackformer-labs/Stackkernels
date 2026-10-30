"""Rotary Position Embedding (RoPE) forward and backward kernels in Triton."""

KERNEL_REV = 1

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


@triton.jit
def _rope_fwd_kernel(
    X_ptr,
    Y_ptr,
    Cos_ptr,
    Sin_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_ys,
    stride_yh,
    stride_yd,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    B,
    S,
    H,
    D,
    start_pos,
    BLOCK_D_HALF: tl.constexpr,
):
    pid = tl.program_id(0)

    # Decode 1D pid into (b, s, h)
    h_idx = pid % H
    rem = pid // H
    s_idx = rem % S
    b_idx = rem // S

    pos = s_idx + start_pos
    half_D = D // 2

    d_offsets = tl.arange(0, BLOCK_D_HALF)
    mask = d_offsets < half_D

    # Base pointers for this (b, s, h) vector
    base_x = X_ptr + b_idx * stride_xb + s_idx * stride_xs + h_idx * stride_xh
    base_y = Y_ptr + b_idx * stride_yb + s_idx * stride_ys + h_idx * stride_yh

    base_cos = Cos_ptr + pos * stride_cos_s
    base_sin = Sin_ptr + pos * stride_sin_s

    # Load first half and second half
    x1 = tl.load(base_x + d_offsets * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base_x + (d_offsets + half_D) * stride_xd, mask=mask, other=0.0).to(tl.float32)

    cos = tl.load(base_cos + d_offsets * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(base_sin + d_offsets * stride_sin_d, mask=mask, other=0.0).to(tl.float32)

    # RoPE rotation:
    # y1 = x1 * cos - x2 * sin
    # y2 = x1 * sin + x2 * cos
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # Store rotated outputs
    tl.store(base_y + d_offsets * stride_yd, y1, mask=mask)
    tl.store(base_y + (d_offsets + half_D) * stride_yd, y2, mask=mask)


@triton.jit
def _rope_bwd_kernel(
    dY_ptr,
    dX_ptr,
    Cos_ptr,
    Sin_ptr,
    stride_dyb,
    stride_dys,
    stride_dyh,
    stride_dyd,
    stride_dxb,
    stride_dxs,
    stride_dxh,
    stride_dxd,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    B,
    S,
    H,
    D,
    start_pos,
    BLOCK_D_HALF: tl.constexpr,
):
    pid = tl.program_id(0)

    h_idx = pid % H
    rem = pid // H
    s_idx = rem % S
    b_idx = rem // S

    pos = s_idx + start_pos
    half_D = D // 2

    d_offsets = tl.arange(0, BLOCK_D_HALF)
    mask = d_offsets < half_D

    base_dy = dY_ptr + b_idx * stride_dyb + s_idx * stride_dys + h_idx * stride_dyh
    base_dx = dX_ptr + b_idx * stride_dxb + s_idx * stride_dxs + h_idx * stride_dxh

    base_cos = Cos_ptr + pos * stride_cos_s
    base_sin = Sin_ptr + pos * stride_sin_s

    dy1 = tl.load(base_dy + d_offsets * stride_dyd, mask=mask, other=0.0).to(tl.float32)
    dy2 = tl.load(base_dy + (d_offsets + half_D) * stride_dyd, mask=mask, other=0.0).to(tl.float32)

    cos = tl.load(base_cos + d_offsets * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(base_sin + d_offsets * stride_sin_d, mask=mask, other=0.0).to(tl.float32)

    # Inverse rotation R(-theta)
    # dx1 = dy1 * cos + dy2 * sin
    # dx2 = dy2 * cos - dy1 * sin
    dx1 = dy1 * cos + dy2 * sin
    dx2 = dy2 * cos - dy1 * sin

    tl.store(base_dx + d_offsets * stride_dxd, dx1, mask=mask)
    tl.store(base_dx + (d_offsets + half_D) * stride_dxd, dx2, mask=mask)


def rope_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Launch Triton forward RoPE kernel for tensor of shape (B, S, H, D)."""
    B, S, H, D = x.shape
    y = torch.empty_like(x)
    half_D = D // 2

    BLOCK_D_HALF = triton.next_power_of_2(half_D)
    num_warps = determine_num_warps(BLOCK_D_HALF)
    grid = (B * S * H,)

    _rope_fwd_kernel[grid](
        x,
        y,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        B,
        S,
        H,
        D,
        start_pos,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
    )
    return y


def rope_bwd(
    dy: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Launch Triton backward RoPE kernel."""
    B, S, H, D = dy.shape
    dx = torch.empty_like(dy)
    half_D = D // 2

    BLOCK_D_HALF = triton.next_power_of_2(half_D)
    num_warps = determine_num_warps(BLOCK_D_HALF)
    grid = (B * S * H,)

    _rope_bwd_kernel[grid](
        dy,
        dx,
        cos,
        sin,
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dy.stride(3),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        dx.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        B,
        S,
        H,
        D,
        start_pos,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
    )
    return dx


__all__ = ["rope_fwd", "rope_bwd"]
