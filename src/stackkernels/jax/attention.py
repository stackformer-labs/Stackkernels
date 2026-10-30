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
    """Flash attention in JAX."""
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
    """Flash decode for q_len = 1 in JAX."""
    out, _ = attention_reference_jax(
        q,
        k,
        v,
        is_causal=False,
        sm_scale=sm_scale,
    )
    return out


__all__ = ["flash_attention", "flash_decode"]
