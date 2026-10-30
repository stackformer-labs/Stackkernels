"""Rotary Position Embedding (RoPE) for PyTorch with GQA and start_pos support."""

from typing import Any

import torch
import torch.nn as nn

from stackkernels._core.positional.rope import rope_bwd, rope_fwd
from stackkernels._reference.torch.rope import (
    apply_rope_single as ref_apply_rope_single,
    precompute_freqs,
)
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import ensure_contiguous


class RoPEFunction(torch.autograd.Function):
    """Autograd function executing fused RoPE."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.start_pos = start_pos

        x_c = ensure_contiguous(x)
        cos_c = ensure_contiguous(cos)
        sin_c = ensure_contiguous(sin)

        if use_triton:
            y = rope_fwd(x_c, cos_c, sin_c, start_pos=start_pos)
            ctx.save_for_backward(cos_c, sin_c)
            return y
        else:
            y = ref_apply_rope_single(x_c, cos_c, sin_c, start_pos=start_pos)
            ctx.save_for_backward(cos_c, sin_c)
            return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        dy_c = ensure_contiguous(dy)
        cos_c, sin_c = ctx.saved_tensors

        if ctx.use_triton:
            dx = rope_bwd(dy_c, cos_c, sin_c, start_pos=ctx.start_pos)
            return dx, None, None, None
        else:
            # Rotating by -theta in backward
            seq_len = dy_c.shape[1]
            cos_slice = cos_c[ctx.start_pos : ctx.start_pos + seq_len, :].unsqueeze(0).unsqueeze(2).float()
            sin_slice = sin_c[ctx.start_pos : ctx.start_pos + seq_len, :].unsqueeze(0).unsqueeze(2).float()

            half = dy_c.shape[-1] // 2
            dy_f = dy_c.float()
            rot_dy = torch.cat((-dy_f[..., half:], dy_f[..., :half]), dim=-1)

            dx = (dy_f * cos_slice) - (rot_dy * sin_slice)
            return dx.to(dy_c.dtype), None, None, None


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
        k: Optional Key tensor of shape (B, S, H_k, D).
        cos: Precomputed cosine table (max_seq_len, D).
        sin: Precomputed sine table (max_seq_len, D).
        start_pos: Position offset for sequence decoding.

    Returns:
        Rotated Q, or tuple of (rotated Q, rotated K).
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

    q_rot = RoPEFunction.apply(q, cos, sin, start_pos)

    if k is not None:
        k_rot = RoPEFunction.apply(k, cos, sin, start_pos)
        return q_rot, k_rot

    return q_rot


class RoPE(nn.Module):
    """Rotary Position Embedding module caching frequency buffers.

    Precomputes and buffers sinusoidal rotation matrices, applying rotary
    embeddings to query and key tensors during forward execution.

    Args:
        dim: Head embedding dimension (must be even).
        max_seq_len: Maximum sequence length supported by the precomputed table. Default: 4096.
        base: Exponential base for frequency calculation. Default: 10000.0.
        device: Target device for precomputed frequency buffers. Default: None.

    Examples:
        >>> rope = RoPE(dim=64, max_seq_len=2048)
        >>> q = torch.randn(2, 512, 8, 64)
        >>> k = torch.randn(2, 512, 2, 64)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: Any = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        cos, sin = precompute_freqs(dim=dim, max_seq_len=max_seq_len, base=base, device=device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, k, self.cos, self.sin, start_pos=start_pos)


# Custom op for torch.compile
try:
    @torch.library.custom_op("stackkernels::rope_op", mutates_args=())
    def _rope_op(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        return RoPEFunction.apply(x, cos, sin, start_pos)

    @_rope_op.register_fake
    def _rope_op_fake(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        return torch.empty_like(x)
except Exception:
    pass


__all__ = ["apply_rope", "RoPE", "RoPEFunction", "precompute_freqs"]
