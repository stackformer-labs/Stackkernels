"""Reference attention implementations in JAX (MHA, GQA, MQA, causal)."""

import math
from typing import Any


def repeat_kv_jax(x: Any, n_rep: int) -> Any:
    """Repeat key/value heads for GQA/MQA in JAX: (B, H_kv, S, D) -> (B, H_q, S, D)."""
    import jax.numpy as jnp

    if n_rep == 1:
        return x
    return jnp.repeat(x, repeats=n_rep, axis=1)


def attention_reference_jax(
    q: Any,
    k: Any,
    v: Any,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[Any, Any]:
    """Compute scaled dot-product attention in JAX with MHA, GQA, and MQA support.

    Accepts inputs in shape (B, H, S, D).
    """
    import jax.nn as jnn
    import jax.numpy as jnp

    B, H_q, S_q, D = q.shape
    _, H_kv, S_k, D_k = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    n_rep = H_q // H_kv
    k_expanded = repeat_kv_jax(k, n_rep)
    v_expanded = repeat_kv_jax(v, n_rep)

    q_f32 = q.astype(jnp.float32)
    k_f32 = k_expanded.astype(jnp.float32)
    v_f32 = v_expanded.astype(jnp.float32)

    scores = jnp.matmul(q_f32, jnp.swapaxes(k_f32, -2, -1)) * sm_scale

    if is_causal:
        q_pos = jnp.arange(S_q)[:, None]
        k_pos = jnp.arange(S_k)[None, :]
        mask = k_pos > (q_pos + (S_k - S_q))
        scores = jnp.where(mask[None, None, :, :], -1e9, scores)

    lse = jnn.logsumexp(scores, axis=-1)
    attn_weights = jnn.softmax(scores, axis=-1)
    out = jnp.matmul(attn_weights, v_f32).astype(q.dtype)

    return out, lse


__all__ = ["attention_reference_jax", "repeat_kv_jax"]
