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
    """SwiGLU in JAX."""
    return ref_swiglu(gate, up)


def geglu(gate: Any, up: Any, approximate: bool = False) -> Any:
    """GeGLU in JAX."""
    return ref_geglu(gate, up, approximate=approximate)


def reglu(gate: Any, up: Any) -> Any:
    """ReGLU in JAX."""
    return ref_reglu(gate, up)


def glu(gate: Any, up: Any, act: Literal["silu", "gelu", "relu"] = "silu") -> Any:
    """GLU in JAX."""
    return ref_glu(gate, up, act=act)


def gelu(x: Any, approximate: bool = False) -> Any:
    """GELU in JAX."""
    return ref_gelu(x, approximate=approximate)


def silu(x: Any) -> Any:
    """SiLU in JAX."""
    return ref_silu(x)


def relu(x: Any) -> Any:
    """ReLU in JAX."""
    return ref_relu(x)


def leaky_relu(x: Any, negative_slope: float = 0.01) -> Any:
    """LeakyReLU in JAX."""
    return ref_leaky_relu(x, negative_slope=negative_slope)


def sigmoid(x: Any) -> Any:
    """Sigmoid in JAX."""
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

