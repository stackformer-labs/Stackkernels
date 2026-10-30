"""Elementwise activation kernels (GELU, SiLU, ReLU, LeakyReLU, Sigmoid) in Triton."""

from typing import Literal

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps

# Activation codes:
# 0: gelu_exact
# 1: gelu_tanh
# 2: silu
# 3: relu
# 4: leaky_relu
# 5: sigmoid


@triton.jit
def _elementwise_fwd_kernel(
    X_ptr,
    Y_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if ACT_TYPE == 0:
        # gelu exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        INV_SQRT2: tl.constexpr = 0.7071067811865475
        erf_val = tl.math.erf(x * INV_SQRT2)
        y = 0.5 * x * (1.0 + erf_val)
    elif ACT_TYPE == 1:
        # gelu tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k = 0.79788456 * (x + 0.044715 * x * x * x)
        exp_2k = tl.exp(2.0 * k)
        t = (exp_2k - 1.0) / (exp_2k + 1.0)
        y = 0.5 * x * (1.0 + t)
    elif ACT_TYPE == 2:
        # silu: x * sigmoid(x)
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
    elif ACT_TYPE == 3:
        # relu
        y = tl.maximum(x, 0.0)
    elif ACT_TYPE == 4:
        # leaky_relu
        y = tl.where(x > 0.0, x, x * alpha)
    else:
        # sigmoid
        y = 1.0 / (1.0 + tl.exp(-x))

    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _elementwise_bwd_kernel(
    dY_ptr,
    X_ptr,
    dX_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dY_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if ACT_TYPE == 0:
        # d/dx[gelu_exact] = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*pi)) * exp(-x^2 / 2)
        INV_SQRT2: tl.constexpr = 0.7071067811865475
        INV_SQRT_2PI: tl.constexpr = 0.3989422804014327
        erf_val = tl.math.erf(x * INV_SQRT2)
        cdf = 0.5 * (1.0 + erf_val)
        pdf = INV_SQRT_2PI * tl.exp(-0.5 * x * x)
        grad = cdf + x * pdf
    elif ACT_TYPE == 1:
        # d/dx[gelu_tanh]
        k = 0.79788456 * (x + 0.044715 * x * x * x)
        exp_2k = tl.exp(2.0 * k)
        t = (exp_2k - 1.0) / (exp_2k + 1.0)
        dk = 0.79788456 * (1.0 + 3.0 * 0.044715 * x * x)
        grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dk
    elif ACT_TYPE == 2:
        # d/dx[silu] = s * (1 + x * (1 - s))
        s = 1.0 / (1.0 + tl.exp(-x))
        grad = s * (1.0 + x * (1.0 - s))
    elif ACT_TYPE == 3:
        # d/dx[relu]
        grad = tl.where(x > 0.0, 1.0, 0.0)
    elif ACT_TYPE == 4:
        # d/dx[leaky_relu]
        grad = tl.where(x > 0.0, 1.0, alpha)
    else:
        # d/dx[sigmoid] = s * (1 - s)
        s = 1.0 / (1.0 + tl.exp(-x))
        grad = s * (1.0 - s)

    dx = dy * grad
    tl.store(dX_ptr + offsets, dx, mask=mask)


def _get_act_code(act: str, approximate: str = "none") -> int:
    a = act.lower().strip()
    if a == "gelu":
        return 1 if approximate.lower().strip() == "tanh" else 0
    elif a == "silu":
        return 2
    elif a == "relu":
        return 3
    elif a == "leaky_relu":
        return 4
    elif a == "sigmoid":
        return 5
    raise ValueError(f"Unknown elementwise activation: {act}")


def elementwise_fwd(
    x: torch.Tensor,
    act: str = "silu",
    approximate: str = "none",
    alpha: float = 0.01,
) -> torch.Tensor:
    """Launch Triton forward kernel for elementwise activation."""
    n_elements = x.numel()
    y = torch.empty_like(x)

    BLOCK_SIZE = 1024
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    act_code = _get_act_code(act, approximate)

    _elementwise_fwd_kernel[grid](
        x,
        y,
        n_elements,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
        ACT_TYPE=act_code,
        num_warps=num_warps,
    )
    return y


def elementwise_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    act: str = "silu",
    approximate: str = "none",
    alpha: float = 0.01,
) -> torch.Tensor:
    """Launch single-pass Triton backward kernel for elementwise activation."""
    n_elements = x.numel()
    dx = torch.empty_like(x)

    BLOCK_SIZE = 1024
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    act_code = _get_act_code(act, approximate)

    _elementwise_bwd_kernel[grid](
        dy,
        x,
        dx,
        n_elements,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
        ACT_TYPE=act_code,
        num_warps=num_warps,
    )
    return dx


__all__ = ["elementwise_fwd", "elementwise_bwd"]
