"""Attention functions in JAX with MHA, GQA, and MQA support."""

from typing import Any

from stackkernels._reference.jax.attention import (
    attention_reference_jax,
    repeat_kv_jax,
)


def flash_attention(
    q: Any,
    k: Any,
    v: Any,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> Any:
    """Flash attention in JAX supporting MHA, GQA, and MQA.

    Args:
        q: Query array of shape (B, H_q, S_q, D).
        k: Key array of shape (B, H_kv, S_k, D).
        v: Value array of shape (B, H_kv, S_k, D).
        is_causal: Whether to apply causal masking. Default: False.
        sm_scale: Softmax scale factor. Defaults to 1 / sqrt(D).

    Returns:
        Attention output array of shape (B, H_q, S_q, D).
    """
    out, _ = attention_reference_jax(
        q,
        k,
        v,
        is_causal=is_causal,
        sm_scale=sm_scale,
    )
    return out


def flash_decode(
    q: Any,
    k: Any,
    v: Any,
    sm_scale: float | None = None,
) -> Any:
    """Flash decode for single-token generation (q_len = 1) in JAX.

    Args:
        q: Query array of shape (B, H_q, 1, D).
        k: Key array of shape (B, H_kv, S_k, D).
        v: Value array of shape (B, H_kv, S_k, D).
        sm_scale: Softmax scale factor. Defaults to 1 / sqrt(D).

    Returns:
        Attention output array of shape (B, H_q, 1, D).
    """
    out, _ = attention_reference_jax(
        q,
        k,
        v,
        is_causal=False,
        sm_scale=sm_scale,
    )
    return out


__all__ = ["flash_attention", "flash_decode"]
