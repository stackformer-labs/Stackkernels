"""Activation functions in JAX."""

from typing import Any, Literal


# --- Elementwise Activations ---

def gelu(x: Any, approximate: bool = False) -> Any:
    """Gaussian Error Linear Unit in JAX."""
    import jax.nn as jnn

    return jnn.gelu(x, approximate=approximate)


def silu(x: Any) -> Any:
    """Sigmoid Linear Unit in JAX."""
    import jax.nn as jnn

    return jnn.silu(x)


def relu(x: Any) -> Any:
    """Rectified Linear Unit in JAX."""
    import jax.nn as jnn

    return jnn.relu(x)


def leaky_relu(x: Any, negative_slope: float = 0.01) -> Any:
    """Leaky ReLU in JAX."""
    import jax.nn as jnn

    return jnn.leaky_relu(x, negative_slope=negative_slope)


def sigmoid(x: Any) -> Any:
    """Sigmoid activation in JAX."""
    import jax.nn as jnn

    return jnn.sigmoid(x)


# --- Gated Linear Units (GLU) ---

def swiglu(gate: Any, up: Any) -> Any:
    """SwiGLU activation in JAX: silu(gate) * up."""
    import jax.nn as jnn

    return jnn.silu(gate) * up


def geglu(gate: Any, up: Any, approximate: bool = False) -> Any:
    """GeGLU activation in JAX: gelu(gate) * up."""
    import jax.nn as jnn

    return jnn.gelu(gate, approximate=approximate) * up


def reglu(gate: Any, up: Any) -> Any:
    """ReGLU activation in JAX: relu(gate) * up."""
    import jax.nn as jnn

    return jnn.relu(gate) * up


def glu(
    gate: Any,
    up: Any,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> Any:
    """General GLU dispatcher in JAX."""
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

