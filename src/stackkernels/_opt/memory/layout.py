"""Contiguity, alignment, dtype and accumulation rules for attention kernels."""

from typing import Literal
import torch


def validate_attention_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """Validate shapes and head counts for MHA, GQA, and MQA attention.

    Expected layout: (B, H, S, D)

    Returns:
        tuple (B, H_q, H_kv, S_q, S_k, D)
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors (B, H, S, D), got q={q.ndim}D, k={k.ndim}D, v={v.ndim}D"
        )

    B, H_q, S_q, D = q.shape
    B_k, H_kv, S_k, D_k = k.shape
    B_v, H_v, S_v, D_v = v.shape

    if B != B_k or B != B_v:
        raise ValueError(f"Batch mismatch: q={B}, k={B_k}, v={B_v}")
    if H_kv != H_v:
        raise ValueError(f"K and V head count mismatch: {H_kv} vs {H_v}")
    if S_k != S_v:
        raise ValueError(f"K and V sequence length mismatch: {S_k} vs {S_v}")
    if D != D_k or D != D_v:
        raise ValueError(f"Head dimension mismatch: q={D}, k={D_k}, v={D_v}")
    if H_q % H_kv != 0:
        raise ValueError(
            f"Query heads ({H_q}) must be divisible by KV heads ({H_kv}) for GQA/MQA"
        )

    return B, H_q, H_kv, S_q, S_k, D


def get_padded_head_dim(D: int) -> int:
    """Return smallest power of 2 >= D for Triton tile padding (e.g., 80 -> 128, 96 -> 128)."""
    if D <= 16:
        return 16
    if D <= 32:
        return 32
    if D <= 64:
        return 64
    if D <= 128:
        return 128
    if D <= 256:
        return 256
    return 1 << (D - 1).bit_length()


__all__ = ["validate_attention_layout", "get_padded_head_dim"]
