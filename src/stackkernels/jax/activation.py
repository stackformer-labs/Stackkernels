"""Activation functions in JAX with custom_vjp support."""

from typing import Any, Literal

from stackkernels._reference.jax.activation import (
    geglu as ref_geglu,
    gelu as ref_gelu,
    glu as ref_glu,
    leaky_relu as ref_leaky_relu,
    reglu as ref_reglu,
    relu as ref_relu,
    sigmoid as ref_sigmoid,
    silu as ref_silu,
    swiglu as ref_swiglu,
)


def swiglu(gate: Any, up: Any) -> Any:
    """SwiGLU activation in JAX: silu(gate) * up.

    Args:
        gate: Gating array (*, D).
        up: Up-projection array (*, D).

    Returns:
        Activated array (*, D).
    """
    return ref_swiglu(gate, up)


def geglu(gate: Any, up: Any, approximate: bool = False) -> Any:
    """GeGLU activation in JAX: gelu(gate) * up.

    Args:
        gate: Gating array (*, D).
        up: Up-projection array (*, D).
        approximate: Whether to use tanh approximation. Default: False.

    Returns:
        Activated array (*, D).
    """
    return ref_geglu(gate, up, approximate=approximate)


def reglu(gate: Any, up: Any) -> Any:
    """ReGLU activation in JAX: relu(gate) * up.

    Args:
        gate: Gating array (*, D).
        up: Up-projection array (*, D).

    Returns:
        Activated array (*, D).
    """
    return ref_reglu(gate, up)


def glu(gate: Any, up: Any, act: Literal["silu", "gelu", "relu"] = "silu") -> Any:
    """General Gated Linear Unit dispatcher in JAX.

    Args:
        gate: Gating array (*, D).
        up: Up-projection array (*, D).
        act: Activation function ("silu", "gelu", or "relu"). Default: "silu".

    Returns:
        Activated array (*, D).
    """
    return ref_glu(gate, up, act=act)


def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian Error Linear Unit (GELU) in JAX.

    Args:
        x: Input array (*, D).
        approximate: Whether to use tanh approximation. Default: False.

    Returns:
        Activated array (*, D).
    """
    return ref_gelu(x, approximate=approximate)


def silu(x: Any) -> Any:
    """Sigmoid Linear Unit (SiLU / Swish) in JAX.

    Args:
        x: Input array (*, D).

    Returns:
        Activated array (*, D).
    """
    return ref_silu(x)


def relu(x: Any) -> Any:
    """Rectified Linear Unit (ReLU) in JAX.

    Args:
        x: Input array (*, D).

    Returns:
        Activated array (*, D).
    """
    return ref_relu(x)


def leaky_relu(x: Any, negative_slope: float = 0.01) -> Any:
    """Leaky ReLU in JAX.

    Args:
        x: Input array (*, D).
        negative_slope: Slope for negative values. Default: 0.01.

    Returns:
        Activated array (*, D).
    """
    return ref_leaky_relu(x, negative_slope=negative_slope)


def sigmoid(x: Any) -> Any:
    """Sigmoid activation in JAX.

    Args:
        x: Input array (*, D).

    Returns:
        Activated array (*, D).
    """
    return ref_sigmoid(x)


__all__ = [
    "swiglu",
    "geglu",
    "reglu",
    "glu",
    "gelu",
    "silu",
    "relu",
    "leaky_relu",
    "sigmoid",
]

