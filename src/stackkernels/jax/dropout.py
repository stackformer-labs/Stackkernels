"""JAX public API for dropout and fused dropout + residual add with custom_vjp."""

from typing import Any

from stackkernels._reference.jax.dropout import (
    dropout as ref_dropout,
    fused_dropout_add as ref_fused_dropout_add,
)


def dropout(
    x: Any,
    p: float = 0.5,
    training: bool = True,
    key: Any | None = None,
) -> Any:
    """Apply dropout to input array using low-memory PRNG reconstruction for JAX.

    Args:
        x: Input array of any shape.
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True; returns x directly if False. Default: True.
        key: JAX PRNGKey or integer seed.

    Returns:
        Array with dropout applied.
    """
    import jax
    import jax.numpy as jnp

    if not training or p <= 0.0:
        return x

    if p >= 1.0:
        return jnp.zeros_like(x)

    if key is None:
        sub_key = jax.random.PRNGKey(0)
    elif isinstance(key, int):
        sub_key = jax.random.PRNGKey(key)
    else:
        sub_key = key

    @jax.custom_vjp
    def _dropout_core(x_val):
        keep_prob = 1.0 - p
        mask = jax.random.bernoulli(sub_key, p=keep_prob, shape=x_val.shape)
        scale = 1.0 / keep_prob
        return (x_val * mask * scale).astype(x_val.dtype)

    def _dropout_fwd(x_val):
        out = _dropout_core(x_val)
        return out, None

    def _dropout_bwd(res, dy):
        keep_prob = 1.0 - p
        mask = jax.random.bernoulli(sub_key, p=keep_prob, shape=dy.shape)
        scale = 1.0 / keep_prob
        dx = (dy * mask * scale).astype(dy.dtype)
        return (dx,)

    _dropout_core.defvjp(_dropout_fwd, _dropout_bwd)
    return _dropout_core(x)


def fused_dropout_add(
    x: Any,
    residual: Any,
    p: float = 0.5,
    training: bool = True,
    key: Any | None = None,
) -> Any:
    """Fused Dropout and Residual Addition for JAX with custom_vjp.

    Computes:
        y = dropout(x, p) + residual

    Args:
        x: Input array of any shape.
        residual: Residual array matching shape of x.
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True; returns x + residual if False. Default: True.
        key: JAX PRNGKey or integer seed.

    Returns:
        dropout(x) + residual
    """
    import jax
    import jax.numpy as jnp

    if not training or p <= 0.0:
        return x + residual

    if p >= 1.0:
        return residual

    if key is None:
        sub_key = jax.random.PRNGKey(0)
    elif isinstance(key, int):
        sub_key = jax.random.PRNGKey(key)
    else:
        sub_key = key

    @jax.custom_vjp
    def _fused_core(x_val, res_val):
        keep_prob = 1.0 - p
        mask = jax.random.bernoulli(sub_key, p=keep_prob, shape=x_val.shape)
        scale = 1.0 / keep_prob
        dropped = (x_val * mask * scale).astype(x_val.dtype)
        return dropped + res_val

    def _fused_fwd(x_val, res_val):
        out = _fused_core(x_val, res_val)
        return out, None

    def _fused_bwd(res, dy):
        keep_prob = 1.0 - p
        mask = jax.random.bernoulli(sub_key, p=keep_prob, shape=dy.shape)
        scale = 1.0 / keep_prob
        dx = (dy * mask * scale).astype(dy.dtype)
        dres = dy
        return (dx, dres)

    _fused_core.defvjp(_fused_fwd, _fused_bwd)
    return _fused_core(x, residual)


__all__ = ["dropout", "fused_dropout_add"]
