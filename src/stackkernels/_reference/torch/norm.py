"""RMSNorm and LayerNorm reference implementations in plain PyTorch with fp32 accumulation."""

from typing import Literal

import torch


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Root Mean Square Normalization (RMSNorm) in plain PyTorch.

    Accumulates in float32 for numerical stability.

    Args:
        x: Input tensor of arbitrary shape (*, D).
        weight: Optional learnable scale parameter of shape (D,).
        eps: Small constant added to variance for numerical stability.

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    out = (x_f32 * rstd).to(orig_dtype)

    if weight is not None:
        out = out * weight

    return out


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute standard Layer Normalization in plain PyTorch with float32 accumulation.

    Args:
        x: Input tensor of shape (*, D).
        weight: Optional scale parameter of shape (D,).
        bias: Optional bias parameter of shape (D,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    out = ((x_f32 - mean) * rstd).to(orig_dtype)

    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    return out


def fused_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual addition and normalization in plain PyTorch.

    Computes:
        res_out = x + residual
        norm_out = Norm(res_out)

    Returns:
        tuple (norm_out, res_out)
    """
    res_out = x + residual
    if norm_type == "rms_norm":
        norm_out = rms_norm(res_out, weight=weight, eps=eps)
    elif norm_type == "layer_norm":
        norm_out = layer_norm(res_out, weight=weight, bias=bias, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return norm_out, res_out


__all__ = ["rms_norm", "layer_norm", "fused_add_norm"]
