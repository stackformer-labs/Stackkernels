"""Rotary Position Embedding for JAX."""

from typing import Any

from stackkernels._reference.jax.rope import apply_rope as ref_apply_rope, precompute_freqs


def apply_rope(
    q: Any,
    k: Any | None = None,
    cos: Any | None = None,
    sin: Any | None = None,
    start_pos: int = 0,
) -> Any:
    """Apply RoPE in JAX supporting GQA and start_pos offset."""
    return ref_apply_rope(q, k, cos, sin, start_pos=start_pos)


__all__ = ["apply_rope", "precompute_freqs"]

