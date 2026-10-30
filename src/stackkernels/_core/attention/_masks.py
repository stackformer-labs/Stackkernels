"""Causal and sliding-window mask JIT helpers with bottom-right alignment."""

import torch
import triton
import triton.language as tl


@triton.jit
def causal_mask_check(
    m_idx,
    n_idx,
    seqlen_q,
    seqlen_k,
):
    """Bottom-right aligned causal mask: True if query can attend to key.

    Condition: n_idx <= m_idx + (seqlen_k - seqlen_q).
    """
    return n_idx <= (m_idx + (seqlen_k - seqlen_q))


def build_causal_mask(
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Construct bottom-right aligned causal boolean mask in PyTorch.

    Returns:
        mask of shape (seqlen_q, seqlen_k) where True indicates valid (unmasked) attention.
    """
    q_pos = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k_pos = torch.arange(seqlen_k, device=device).unsqueeze(0)
    return (k_pos <= (q_pos + (seqlen_k - seqlen_q))).to(dtype)


__all__ = ["causal_mask_check", "build_causal_mask"]
