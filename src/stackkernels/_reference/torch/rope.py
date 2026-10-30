"""Rotary Position Embedding (RoPE) reference implementation in PyTorch."""

import torch


def precompute_freqs(
    dim: int,
    max_seq_len: int = 4096,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency tables for RoPE.

    Returns:
        tuple (cos, sin) each of shape (max_seq_len, dim).
    """
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, half_dim)

    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)

    # Cat to full dim: (max_seq_len, dim)
    cos = torch.cat([cos_half, cos_half], dim=-1).to(dtype)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimension: (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply RoPE to a single tensor x of shape (B, S, H, D)."""
    seq_len = x.shape[1]
    # Slice freqs for current positions
    cos_slice = cos[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin_slice = sin[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)

    orig_dtype = x.dtype
    x_f32 = x.float()
    cos_f32 = cos_slice.float()
    sin_f32 = sin_slice.float()

    out = (x_f32 * cos_f32) + (rotate_half(x_f32) * sin_f32)
    return out.to(orig_dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    start_pos: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to Q (and optionally K for GQA/MQA).

    Args:
        q: Query tensor of shape (B, S, H_q, D).
        k: Optional Key tensor of shape (B, S, H_k, D) (H_k may != H_q for GQA).
        cos: Cosine table of shape (max_seq_len, D).
        sin: Sine table of shape (max_seq_len, D).
        start_pos: Starting position offset (e.g. for KV cache decoding).

    Returns:
        Rotated Q, or (rotated Q, rotated K) if K is provided.
    """
    dim = q.shape[-1]
    seq_len = q.shape[1]

    if cos is None or sin is None:
        cos, sin = precompute_freqs(
            dim=dim,
            max_seq_len=start_pos + seq_len + 128,
            device=q.device,
            dtype=q.dtype,
        )

    q_rot = apply_rope_single(q, cos, sin, start_pos=start_pos)

    if k is not None:
        k_rot = apply_rope_single(k, cos, sin, start_pos=start_pos)
        return q_rot, k_rot

    return q_rot


__all__ = ["precompute_freqs", "rotate_half", "apply_rope", "apply_rope_single"]

