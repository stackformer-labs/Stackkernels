"""Fused Dropout and Residual Addition kernel in Triton."""

KERNEL_REV = 1

from typing import Any

import torch
import triton
import triton.language as tl

from stackkernels._core._common import cdiv
from stackkernels._core.dropout.dropout import dropout_bwd


@triton.jit
def _fused_dropout_add_fwd_kernel(
    X_ptr,
    Residual_ptr,
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
    res = tl.load(Residual_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random > p
    dropped = tl.where(keep, (x.to(tl.float32) * scale).to(x.dtype), tl.zeros_like(x))
    y = dropped + res
    tl.store(Y_ptr + offsets, y, mask=mask)


def fused_dropout_add_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.5,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Execute Triton fused dropout + residual addition forward kernel."""
    if p <= 0.0:
        return x + residual
    if p >= 1.0:
        return residual.clone()

    n_elements = x.numel()
    y = torch.empty_like(x)
    scale = 1.0 / (1.0 - p)

    cfg = config or {}
    BLOCK_SIZE = int(cfg.get("BLOCK_SIZE", 1024))
    num_warps = int(cfg.get("num_warps", 4))
    num_stages = int(cfg.get("num_stages", 1))

    grid = (cdiv(n_elements, BLOCK_SIZE),)

    _fused_dropout_add_fwd_kernel[grid](
        x,
        residual,
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


def fused_dropout_add_bwd(
    dy: torch.Tensor,
    p: float = 0.5,
    seed: int = 0,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute backward pass for fused dropout + residual addition.

    Returns:
        (dx, dresidual) where dresidual is dy.
    """
    dx = dropout_bwd(dy, p=p, seed=seed, config=config)
    return dx, dy


__all__ = ["fused_dropout_add_fwd", "fused_dropout_add_bwd", "KERNEL_REV"]
