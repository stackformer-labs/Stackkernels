"""Dropout forward and backward kernels in Triton."""

KERNEL_REV = 1

from typing import Any

import torch
import triton
import triton.language as tl

from stackkernels._core._common import cdiv


@triton.jit
def _dropout_fwd_kernel(
    X_ptr,
    Y_ptr,
    n_elements,
    p,
    seed,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random > p
    y = tl.where(keep, (x.to(tl.float32) * scale).to(x.dtype), tl.zeros_like(x))
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _dropout_bwd_kernel(
    dY_ptr,
    dX_ptr,
    n_elements,
    p,
    seed,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dY_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random > p
    dx = tl.where(keep, (dy.to(tl.float32) * scale).to(dy.dtype), tl.zeros_like(dy))
    tl.store(dX_ptr + offsets, dx, mask=mask)


def dropout_fwd(
    x: torch.Tensor,
    p: float = 0.5,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Execute Triton dropout forward kernel."""
    if p <= 0.0:
        return x
    if p >= 1.0:
        return torch.zeros_like(x)

    n_elements = x.numel()
    y = torch.empty_like(x)
    scale = 1.0 / (1.0 - p)

    cfg = config or {}
    BLOCK_SIZE = int(cfg.get("BLOCK_SIZE", 1024))
    num_warps = int(cfg.get("num_warps", 4))
    num_stages = int(cfg.get("num_stages", 1))

    grid = (cdiv(n_elements, BLOCK_SIZE),)

    _dropout_fwd_kernel[grid](
        x,
        y,
        n_elements,
        p,
        seed,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def dropout_bwd(
    dy: torch.Tensor,
    p: float = 0.5,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Execute Triton dropout backward kernel."""
    if p <= 0.0:
        return dy
    if p >= 1.0:
        return torch.zeros_like(dy)

    n_elements = dy.numel()
    dx = torch.empty_like(dy)
    scale = 1.0 / (1.0 - p)

    cfg = config or {}
    BLOCK_SIZE = int(cfg.get("BLOCK_SIZE", 1024))
    num_warps = int(cfg.get("num_warps", 4))
    num_stages = int(cfg.get("num_stages", 1))

    grid = (cdiv(n_elements, BLOCK_SIZE),)

    _dropout_bwd_kernel[grid](
        dy,
        dx,
        n_elements,
        p,
        seed,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dx


__all__ = ["dropout_fwd", "dropout_bwd", "KERNEL_REV"]
