"""PyTorch wrappers for FlashAttention and FlashDecode."""

import math
from typing import Any

import torch
import torch.nn as nn

from stackkernels._core.attention.decode import flash_decode as core_flash_decode
from stackkernels._core.attention.flash_bwd import flash_attn_bwd
from stackkernels._core.attention.flash_fwd import flash_attn_fwd
from stackkernels._reference.torch.attention import attention_reference, repeat_kv
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import ensure_contiguous


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd function executing Triton FlashAttention with MHA, GQA, and MQA support."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        use_triton = should_use_triton(q)
        ctx.use_triton = use_triton
        ctx.is_causal = is_causal

        D = q.shape[-1]
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        ctx.sm_scale = sm_scale

        q_contig = ensure_contiguous(q)
        k_contig = ensure_contiguous(k)
        v_contig = ensure_contiguous(v)

        if use_triton:
            out, lse = flash_attn_fwd(
                q_contig,
                k_contig,
                v_contig,
                is_causal=is_causal,
                sm_scale=sm_scale,
            )
            ctx.save_for_backward(q_contig, k_contig, v_contig, out, lse)
            return out
        else:
            out, lse = attention_reference(
                q_contig,
                k_contig,
                v_contig,
                is_causal=is_causal,
                sm_scale=sm_scale,
            )
            ctx.save_for_backward(q_contig, k_contig, v_contig, out, lse)
            return out

    @staticmethod
    def backward(
        ctx: Any, do: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, out, lse = ctx.saved_tensors
        do_contig = ensure_contiguous(do)

        if ctx.use_triton:
            dq, dk, dv = flash_attn_bwd(
                do_contig,
                q,
                k,
                v,
                out,
                lse,
                is_causal=ctx.is_causal,
                sm_scale=ctx.sm_scale,
            )
            return dq, dk, dv, None, None
        else:
            # Reference backward computation
            B, H_q, S_q, D = q.shape
            _, H_kv, S_k, _ = k.shape
            group_size = H_q // H_kv

            q_req = q.detach().clone().requires_grad_(True)
            k_req = k.detach().clone().requires_grad_(True)
            v_req = v.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                out_ref, _ = attention_reference(
                    q_req,
                    k_req,
                    v_req,
                    is_causal=ctx.is_causal,
                    sm_scale=ctx.sm_scale,
                )
                out_ref.backward(do_contig)

            return q_req.grad, k_req.grad, v_req.grad, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute Flash Attention forward.

    Supports:
        - MHA: H_q == H_kv
        - GQA: H_q > H_kv > 1
        - MQA: H_kv == 1
        - Causal with bottom-right alignment: k <= q + (S_k - S_q)
        - Head dimensions: 64, 80, 96, 128, 256.

    Args:
        q: Query tensor (B, H_q, S_q, D).
        k: Key tensor (B, H_kv, S_k, D).
        v: Value tensor (B, H_kv, S_k, D).
        is_causal: Whether to apply bottom-right causal mask.
        sm_scale: Scale factor (default: 1 / sqrt(D)).

    Returns:
        Output tensor of shape (B, H_q, S_q, D).
    """
    return FlashAttentionFunction.apply(q, k, v, is_causal, sm_scale)


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute Split-KV FlashDecode for single-token generation (q_len = 1).

    Args:
        q: Query tensor of shape (B, H_q, 1, D).
        k: Key tensor of shape (B, H_kv, S_k, D).
        v: Value tensor of shape (B, H_kv, S_k, D).
        sm_scale: Scale factor (default: 1 / sqrt(D)).

    Returns:
        Output tensor of shape (B, H_q, 1, D).
    """
    if should_use_triton(q):
        return core_flash_decode(q, k, v, sm_scale=sm_scale)
    else:
        out, _ = attention_reference(q, k, v, is_causal=False, sm_scale=sm_scale)
        return out


class FlashAttention(nn.Module):
    """FlashAttention module with MHA, GQA, and MQA support.

    Computes IO-aware scaled dot-product attention tiling over SRAM to avoid
    materializing the full intermediate $S_q \\times S_k$ attention matrix.

    Args:
        is_causal: Whether to apply a bottom-right aligned causal mask. Default: False.
        sm_scale: Softmax scaling factor. Defaults to $1.0 / \\sqrt{D}$.

    Examples:
        >>> attn = FlashAttention(is_causal=True)
        >>> q = torch.randn(2, 8, 1024, 64)
        >>> k = torch.randn(2, 8, 1024, 64)
        >>> v = torch.randn(2, 8, 1024, 64)
        >>> out = attn(q, k, v)
    """

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.sm_scale = sm_scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return flash_attention(
            q,
            k,
            v,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
        )


# Register torch.library.custom_op
try:

    @torch.library.custom_op("stackkernels::flash_attn_op", mutates_args=())
    def _flash_attn_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        sm_scale: float | None,
    ) -> torch.Tensor:
        return flash_attention(q, k, v, is_causal=is_causal, sm_scale=sm_scale)

    @_flash_attn_op.register_fake
    def _flash_attn_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        sm_scale: float | None,
    ) -> torch.Tensor:
        return torch.empty_like(q)
except Exception:
    pass


__all__ = [
    "flash_attention",
    "flash_decode",
    "FlashAttention",
    "FlashAttentionFunction",
]
