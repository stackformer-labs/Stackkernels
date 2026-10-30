"""Reusable buffers for attention: LSE, backward delta, and Split-KV partials."""

import torch


def allocate_lse_buffer(
    batch_size: int,
    num_heads: int,
    seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate log-sum-exp buffer of shape (B, H, S) in float32."""
    return torch.empty((batch_size, num_heads, seqlen), dtype=torch.float32, device=device)


def allocate_delta_buffer(
    batch_size: int,
    num_heads: int,
    seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate backward delta buffer D = sum(dO * O) of shape (B, H, S) in float32."""
    return torch.empty((batch_size, num_heads, seqlen), dtype=torch.float32, device=device)


def allocate_split_kv_buffers(
    batch_size: int,
    num_heads: int,
    split_kv: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate intermediate buffers for FlashDecode split-KV reduction.

    Returns:
        tuple (partial_out: (B, H, split_kv, D), partial_lse: (B, H, split_kv))
    """
    partial_out = torch.empty(
        (batch_size, num_heads, split_kv, head_dim), dtype=dtype, device=device
    )
    partial_lse = torch.empty(
        (batch_size, num_heads, split_kv), dtype=torch.float32, device=device
    )
    return partial_out, partial_lse


__all__ = [
    "allocate_lse_buffer",
    "allocate_delta_buffer",
    "allocate_split_kv_buffers",
]
