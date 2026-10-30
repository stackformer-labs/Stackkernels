"""JAX public API for RMSNorm, LayerNorm, and Fused Add Norm with custom_vjp."""

from typing import Any, Literal

from stackkernels._reference.jax.norm import (
    fused_add_norm as ref_fused_add_norm,
    layer_norm as ref_layer_norm,
    rms_norm as ref_rms_norm,
)


def rms_norm(
    x: Any,
    weight: Any | None = None,
    eps: float = 1e-6,
) -> Any:
    """Root Mean Square Normalization for JAX with custom_vjp."""
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def _rms_norm_fn(x_val, w_val):
        return ref_rms_norm(x_val, w_val, eps=eps)

    def _rms_norm_fwd(x_val, w_val):
        orig_dtype = x_val.dtype
        x_f32 = x_val.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        rstd = jnp.reciprocal(jnp.sqrt(variance + eps))
        normed = (x_f32 * rstd).astype(orig_dtype)
        out = normed * w_val if w_val is not None else normed
        return out, (x_val, w_val, rstd)

    def _rms_norm_bwd(res, dy):
        x_val, w_val, rstd = res
        N = x_val.shape[-1]
        x_f32 = x_val.astype(jnp.float32)
        dy_f32 = dy.astype(jnp.float32)
        x_hat = x_f32 * rstd

        if w_val is not None:
            dy_w = dy_f32 * w_val
            axes = tuple(range(x_val.ndim - 1))
            dw = jnp.sum(dy_f32 * x_hat, axis=axes).astype(w_val.dtype)
        else:
            dy_w = dy_f32
            dw = None

        sum_dy_x_hat = jnp.sum(dy_w * x_hat, axis=-1, keepdims=True)
        dx = (rstd * (dy_w - (x_hat / N) * sum_dy_x_hat)).astype(x_val.dtype)
        return dx, dw

    _rms_norm_fn.defvjp(_rms_norm_fwd, _rms_norm_bwd)
    return _rms_norm_fn(x, weight)


def layer_norm(
    x: Any,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Any:
    """Layer Normalization for JAX with custom_vjp."""
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def _layer_norm_fn(x_val, w_val, b_val):
        return ref_layer_norm(x_val, w_val, b_val, eps=eps)

    def _layer_norm_fwd(x_val, w_val, b_val):
        orig_dtype = x_val.dtype
        x_f32 = x_val.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
        rstd = jnp.reciprocal(jnp.sqrt(var + eps))
        normed = ((x_f32 - mean) * rstd).astype(orig_dtype)
        out = normed
        if w_val is not None:
            out = out * w_val
        if b_val is not None:
            out = out + b_val
        return out, (x_val, w_val, b_val, mean, rstd)

    def _layer_norm_bwd(res, dy):
        x_val, w_val, b_val, mean, rstd = res
        N = x_val.shape[-1]
        x_f32 = x_val.astype(jnp.float32)
        dy_f32 = dy.astype(jnp.float32)
        x_hat = (x_f32 - mean) * rstd

        if w_val is not None:
            dy_w = dy_f32 * w_val
            axes = tuple(range(x_val.ndim - 1))
            dw = jnp.sum(dy_f32 * x_hat, axis=axes).astype(w_val.dtype)
        else:
            dy_w = dy_f32
            dw = None

        if b_val is not None:
            axes = tuple(range(x_val.ndim - 1))
            db = jnp.sum(dy_f32, axis=axes).astype(b_val.dtype)
        else:
            db = None

        s1 = jnp.sum(dy_w, axis=-1, keepdims=True)
        s2 = jnp.sum(dy_w * x_hat, axis=-1, keepdims=True)
        dx = (rstd * (dy_w - (s1 / N) - (x_hat / N) * s2)).astype(x_val.dtype)
        return dx, dw, db

    _layer_norm_fn.defvjp(_layer_norm_fwd, _layer_norm_bwd)
    return _layer_norm_fn(x, weight, bias)


def fused_add_norm(
    x: Any,
    residual: Any,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[Any, Any]:
    """Fused residual addition and normalization for JAX."""
    return ref_fused_add_norm(x, residual, weight=weight, bias=bias, eps=eps, norm_type=norm_type)


__all__ = ["rms_norm", "layer_norm", "fused_add_norm"]
