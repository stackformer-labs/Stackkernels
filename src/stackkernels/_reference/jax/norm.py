"""RMSNorm and LayerNorm reference implementations in JAX with fp32 accumulation."""

from typing import Any, Literal


def rms_norm(
    x: Any,
    weight: Any | None = None,
    eps: float = 1e-6,
) -> Any:
    """Compute Root Mean Square Normalization in JAX with float32 accumulation.

    Args:
        x: Input JAX array (*, D).
        weight: Optional scale array (D,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized array matching input shape and dtype.
    """
    import jax.numpy as jnp

    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    rstd = jnp.reciprocal(jnp.sqrt(variance + eps))
    out = (x_f32 * rstd).astype(orig_dtype)

    if weight is not None:
        out = out * weight

    return out


def layer_norm(
    x: Any,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Any:
    """Compute Layer Normalization in JAX with float32 accumulation.

    Args:
        x: Input JAX array (*, D).
        weight: Optional scale array (D,).
        bias: Optional bias array (D,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized array matching input shape and dtype.
    """
    import jax.numpy as jnp

    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
    rstd = jnp.reciprocal(jnp.sqrt(var + eps))
    out = ((x_f32 - mean) * rstd).astype(orig_dtype)

    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    return out


def fused_add_norm(
    x: Any,
    residual: Any,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[Any, Any]:
    """Fused residual addition and normalization in JAX."""
    res_out = x + residual
    if norm_type == "rms_norm":
        norm_out = rms_norm(res_out, weight=weight, eps=eps)
    elif norm_type == "layer_norm":
        norm_out = layer_norm(res_out, weight=weight, bias=bias, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return norm_out, res_out


__all__ = ["rms_norm", "layer_norm", "fused_add_norm"]
