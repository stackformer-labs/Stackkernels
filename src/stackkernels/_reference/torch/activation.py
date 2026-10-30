"""Activation functions in plain PyTorch with float32 accumulation."""

from typing import Literal

import torch
import torch.nn.functional as F


# --- Elementwise Activations ---

def gelu(x: torch.Tensor, approximate: Literal["none", "tanh"] = "none") -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU)."""
    return F.gelu(x, approximate=approximate)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish)."""
    return F.silu(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit (ReLU)."""
    return F.relu(x)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Leaky ReLU."""
    return F.leaky_relu(x, negative_slope=negative_slope)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation."""
    return torch.sigmoid(x)


# --- Gated Linear Units (GLU) ---

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up."""
    return F.silu(gate.float()).to(gate.dtype) * up


def geglu(gate: torch.Tensor, up: torch.Tensor, approximate: Literal["none", "tanh"] = "tanh") -> torch.Tensor:
    """GeGLU activation: gelu(gate) * up."""
    return F.gelu(gate.float(), approximate=approximate).to(gate.dtype) * up


def reglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """ReGLU activation: relu(gate) * up."""
    return F.relu(gate.float()).to(gate.dtype) * up


def glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> torch.Tensor:
    """General Gated Linear Unit dispatcher."""
    if act == "silu":
        return swiglu(gate, up)
    elif act == "gelu":
        return geglu(gate, up)
    elif act == "relu":
        return reglu(gate, up)
    else:
        raise ValueError(f"Unsupported GLU activation type: {act}")


__all__ = [
    "gelu",
    "silu",
    "relu",
    "leaky_relu",
    "sigmoid",
    "swiglu",
    "geglu",
    "reglu",
    "glu",
]
