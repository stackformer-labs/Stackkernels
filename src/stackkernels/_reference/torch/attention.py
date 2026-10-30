"""Reference attention implementations in PyTorch (MHA, GQA, MQA, causal)."""

import math
from typing import Any

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for GQA/MQA: (B, H_kv, S, D) -> (B, H_q, S, D)."""
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, repeats=n_rep, dim=1)


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention in PyTorch with MHA, GQA, and MQA support.

    Accepts inputs in shape (B, H, S, D).
    Supports:
        - MHA: H_q == H_kv
        - GQA: H_q > H_kv > 1
        - MQA: H_kv == 1
        - Unequal seqlen_q and seqlen_k with bottom-right causal alignment.

    Args:
        q: Query tensor of shape (B, H_q, S_q, D).
        k: Key tensor of shape (B, H_kv, S_k, D).
        v: Value tensor of shape (B, H_kv, S_k, D).
        is_causal: Whether to apply bottom-right aligned causal masking.
        sm_scale: Scale factor for dot products (defaults to 1 / sqrt(D)).

    Returns:
        tuple (output of shape (B, H_q, S_q, D), lse of shape (B, H_q, S_q))
    """
    B, H_q, S_q, D = q.shape
    _, H_kv, S_k, D_k = k.shape
    assert D == D_k, f"Head dimensions must match: {D} != {D_k}"
    assert H_q % H_kv == 0, f"Query heads {H_q} must be divisible by KV heads {H_kv}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Expand KV heads for GQA/MQA if necessary
    n_rep = H_q // H_kv
    k_expanded = repeat_kv(k, n_rep)
    v_expanded = repeat_kv(v, n_rep)

    # Compute dot products in float32 for numerical stability
    q_f32 = q.float()
    k_f32 = k_expanded.float()
    v_f32 = v_expanded.float()

    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * sm_scale  # (B, H_q, S_q, S_k)

    if is_causal:
        # Bottom-right aligned causal mask: k <= q + (S_k - S_q)
        q_pos = torch.arange(S_q, device=q.device).unsqueeze(1)  # (S_q, 1)
        k_pos = torch.arange(S_k, device=k.device).unsqueeze(0)  # (1, S_k)
        mask = k_pos > (q_pos + (S_k - S_q))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)  # (B, H_q, S_q)
    attn_weights = torch.softmax(scores, dim=-1)
    # Handle rows that were fully masked out
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    out = torch.matmul(attn_weights, v_f32).to(q.dtype)
    return out, lse


__all__ = ["attention_reference", "repeat_kv"]
