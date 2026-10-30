"""Rotary Position Embedding reference implementation in JAX."""

from typing import Any


def precompute_freqs(
    dim: int,
    max_seq_len: int = 4096,
    base: float = 10000.0,
) -> tuple[Any, Any]:
    """Precompute cosine and sine frequency tables in JAX."""
    import jax.numpy as jnp

    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)

    cos_half = jnp.cos(freqs)
    sin_half = jnp.sin(freqs)

    cos = jnp.concatenate([cos_half, cos_half], axis=-1)
    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    return cos, sin


def rotate_half(x: Any) -> Any:
    """Rotate half of hidden dimension in JAX."""
    import jax.numpy as jnp

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(
    q: Any,
    k: Any | None = None,
    cos: Any | None = None,
    sin: Any | None = None,
    start_pos: int = 0,
) -> Any:
    """Apply RoPE in JAX supporting GQA and start_pos offset."""
    import jax.numpy as jnp

    dim = q.shape[-1]
    seq_len = q.shape[1]

    if cos is None or sin is None:
        cos, sin = precompute_freqs(dim, max_seq_len=start_pos + seq_len + 128)

    cos_slice = jnp.expand_dims(jnp.expand_dims(cos[start_pos : start_pos + seq_len, :], 0), 2)
    sin_slice = jnp.expand_dims(jnp.expand_dims(sin[start_pos : start_pos + seq_len, :], 0), 2)

    q_rot = (q * cos_slice) + (rotate_half(q) * sin_slice)

    if k is not None:
        k_rot = (k * cos_slice) + (rotate_half(k) * sin_slice)
        return q_rot, k_rot

    return q_rot


__all__ = ["precompute_freqs", "rotate_half", "apply_rope"]

