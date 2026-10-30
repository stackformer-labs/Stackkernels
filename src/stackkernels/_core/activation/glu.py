"""SwiGLU, GeGLU, ReGLU fused forward and backward kernels in Triton."""

KERNEL_REV = 1

from typing import Literal

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


# ACT_TYPE constants: 0 = silu (SwiGLU), 1 = gelu (GeGLU), 2 = relu (ReGLU)

@triton.jit
def _glu_fwd_kernel(
    Gate_ptr,
    Up_ptr,
    Y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    """Fused GLU forward kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # 1. Compute activation on gate
    if ACT_TYPE == 0:
        # silu(g) = g * sigmoid(g)
        sig = 1.0 / (1.0 + tl.exp(-g))
        act_g = g * sig
    elif ACT_TYPE == 1:
        # gelu tanh approximation: 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
        k = 0.79788456 * (g + 0.044715 * g * g * g)
        t = (tl.exp(2.0 * k) - 1.0) / (tl.exp(2.0 * k) + 1.0)
        act_g = 0.5 * g * (1.0 + t)
    else:
        # relu
        act_g = tl.maximum(g, 0.0)

    # 2. Gated multiplication
    y = act_g * u
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _glu_bwd_kernel(
    dY_ptr,
    Gate_ptr,
    Up_ptr,
    dGate_ptr,
    dUp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    """Fused GLU backward kernel recomputing activations on-chip."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dY_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Recompute activation and its first derivative on-chip
    if ACT_TYPE == 0:
        # silu(g) = g * s
        s = 1.0 / (1.0 + tl.exp(-g))
        act_g = g * s
        # d/dg[g * s] = s * (1 + g * (1 - s))
        d_act_g = s * (1.0 + g * (1.0 - s))
    elif ACT_TYPE == 1:
        # gelu tanh approximation
        k = 0.79788456 * (g + 0.044715 * g * g * g)
        exp_2k = tl.exp(2.0 * k)
        t = (exp_2k - 1.0) / (exp_2k + 1.0)
        act_g = 0.5 * g * (1.0 + t)
        dk = 0.79788456 * (1.0 + 3.0 * 0.044715 * g * g)
        d_act_g = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * dk
    else:
        # relu
        act_g = tl.maximum(g, 0.0)
        d_act_g = tl.where(g > 0.0, 1.0, 0.0)

    # Gradients:
    # dUp = dy * act(gate)
    # dGate = dy * up * act'(gate)
    dup = dy * act_g
    dgate = dy * u * d_act_g

    tl.store(dUp_ptr + offsets, dup, mask=mask)
    tl.store(dGate_ptr + offsets, dgate, mask=mask)


def _get_act_code(act: str) -> int:
    a = act.lower().strip()
    if a in ("silu", "swiglu"):
        return 0
    elif a in ("gelu", "geglu"):
        return 1
    elif a in ("relu", "reglu"):
        return 2
    raise ValueError(f"Unsupported GLU activation type: {act}")


def glu_fwd(
    gate: torch.Tensor,
    up: torch.Tensor,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> torch.Tensor:
    """Launch Triton forward kernel for GLU."""
    n_elements = gate.numel()
    y = torch.empty_like(gate)

    BLOCK_SIZE = 1024
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    act_code = _get_act_code(act)

    _glu_fwd_kernel[grid](
        gate,
        up,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ACT_TYPE=act_code,
        num_warps=num_warps,
    )
    return y


def glu_bwd(
    dy: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch single-pass Triton backward kernel for GLU."""
    n_elements = gate.numel()
    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)

    BLOCK_SIZE = 1024
    num_warps = determine_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    act_code = _get_act_code(act)

    _glu_bwd_kernel[grid](
        dy,
        gate,
        up,
        dgate,
        dup,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ACT_TYPE=act_code,
        num_warps=num_warps,
    )
    return dgate, dup


__all__ = ["glu_fwd", "glu_bwd"]
