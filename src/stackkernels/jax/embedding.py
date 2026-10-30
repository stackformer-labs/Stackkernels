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
    """Apply Rotary Position Embedding (RoPE) in JAX.

    Supports Multi-Head (MHA), Grouped-Query (GQA), and Multi-Query Attention (MQA),
    as well as sequence start position offsets for generation.

    Args:
        q: Query array of shape (B, S, H_q, D).
        k: Optional Key array of shape (B, S, H_kv, D).
        cos: Precomputed cosine table (max_seq_len, D).
        sin: Precomputed sine table (max_seq_len, D).
        start_pos: Sequence position offset. Default: 0.

    Returns:
        Rotated query array, or tuple of (rotated query, rotated key) if k is provided.
    """
    return ref_apply_rope(q, k, cos, sin, start_pos=start_pos)


__all__ = ["apply_rope", "precompute_freqs"]

