"""Reference implementation of dropout and fused dropout + residual add in JAX."""

from typing import Any
import jax
import jax.numpy as jnp


def dropout(
    x: Any,
    p: float = 0.5,
    training: bool = True,
    key: Any | None = None,
) -> Any:
    """Reference implementation of dropout for JAX.

    Args:
        x: Input array.
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True. Default: True.
        key: JAX PRNGKey or integer seed.

    Returns:
        Array with dropout applied.
    """
    if not training or p == 0.0:
        return x

    if p >= 1.0:
        return jnp.zeros_like(x)

    if key is None:
        key = jax.random.PRNGKey(0)
    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    keep_prob = 1.0 - p
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    scale = 1.0 / keep_prob
    return (x * mask * scale).astype(x.dtype)


def fused_dropout_add(
    x: Any,
    residual: Any,
    p: float = 0.5,
    training: bool = True,
    key: Any | None = None,
) -> Any:
    """Reference implementation of fused dropout and residual addition for JAX.

    Args:
        x: Input array.
        residual: Residual array.
        p: Dropout probability. Default: 0.5.
        training: Apply dropout if True. Default: True.
        key: JAX PRNGKey or integer seed.

    Returns:
        dropout(x) + residual
    """
    return dropout(x, p=p, training=training, key=key) + residual


__all__ = ["dropout", "fused_dropout_add"]
