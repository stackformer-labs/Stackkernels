"""rms_norm, RMSNorm, layer_norm, LayerNorm, fused_add_norm"""

from typing import Any, Literal

import torch
import torch.nn as nn

from stackkernels._core.norm.fused_add_norm import fused_add_norm_bwd, fused_add_norm_fwd
from stackkernels._core.norm.layer_norm import layer_norm_bwd, layer_norm_fwd
from stackkernels._core.norm.rms_norm import rms_norm_bwd, rms_norm_fwd
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import (
    ensure_contiguous,
    flatten_to_2d,
    unflatten_from_2d,
)


class RMSNormFunction(torch.autograd.Function):
    """Autograd function executing Triton or Reference RMSNorm."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.eps = eps
        ctx.has_weight = weight is not None

        if use_triton:
            x_2d, orig_shape = flatten_to_2d(x)
            w_contig = ensure_contiguous(weight) if weight is not None else None
            y_2d, rstd = rms_norm_fwd(x_2d, w_contig, eps=eps)
            ctx.orig_shape = orig_shape
            ctx.save_for_backward(x_2d, w_contig, rstd)
            return unflatten_from_2d(y_2d, orig_shape)
        else:
            orig_dtype = x.dtype
            x_f32 = x.float()
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            normed = (x_f32 * rstd).to(orig_dtype)
            out = normed * weight if weight is not None else normed
            ctx.save_for_backward(x, weight, rstd)
            return out

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        dy_contig = ensure_contiguous(dy)

        if ctx.use_triton:
            x_2d, w_contig, rstd = ctx.saved_tensors
            dy_2d, _ = flatten_to_2d(dy_contig)
            dx_2d, dw = rms_norm_bwd(dy_2d, x_2d, w_contig, rstd, eps=ctx.eps)
            dx = unflatten_from_2d(dx_2d, ctx.orig_shape)
            return dx, dw, None
        else:
            x, weight, rstd = ctx.saved_tensors
            N = x.shape[-1]
            x_f32 = x.float()
            dy_f32 = dy_contig.float()
            x_hat = x_f32 * rstd

            if weight is not None:
                dy_w = dy_f32 * weight
                dw = (dy_f32 * x_hat).sum(dim=list(range(x.ndim - 1))).to(weight.dtype)
            else:
                dy_w = dy_f32
                dw = None

            sum_dy_x_hat = (dy_w * x_hat).sum(dim=-1, keepdim=True)
            dx = (rstd * (dy_w - (x_hat / N) * sum_dy_x_hat)).to(x.dtype)
            return dx, dw, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root Mean Square Normalization (RMSNorm) functional interface.

    Args:
        x: Input tensor (*, D).
        weight: Optional learnable scale parameter (D,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor with same shape and dtype as x.
    """
    return RMSNormFunction.apply(x, weight, eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) module."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}"


class LayerNormFunction(torch.autograd.Function):
    """Autograd function executing Triton or Reference LayerNorm."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.eps = eps
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None

        if use_triton:
            x_2d, orig_shape = flatten_to_2d(x)
            w_contig = ensure_contiguous(weight) if weight is not None else None
            b_contig = ensure_contiguous(bias) if bias is not None else None
            y_2d, mean, rstd = layer_norm_fwd(x_2d, w_contig, b_contig, eps=eps)
            ctx.orig_shape = orig_shape
            ctx.save_for_backward(x_2d, w_contig, b_contig, mean, rstd)
            return unflatten_from_2d(y_2d, orig_shape)
        else:
            orig_dtype = x.dtype
            x_f32 = x.float()
            mean = x_f32.mean(dim=-1, keepdim=True)
            var = ((x_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            normed = ((x_f32 - mean) * rstd).to(orig_dtype)
            out = normed
            if weight is not None:
                out = out * weight
            if bias is not None:
                out = out + bias
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            return out

    @staticmethod
    def backward(
        ctx: Any, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None]:
        dy_contig = ensure_contiguous(dy)

        if ctx.use_triton:
            x_2d, w_contig, b_contig, mean, rstd = ctx.saved_tensors
            dy_2d, _ = flatten_to_2d(dy_contig)
            dx_2d, dw, db = layer_norm_bwd(
                dy_2d, x_2d, w_contig, b_contig, mean, rstd, eps=ctx.eps
            )
            dx = unflatten_from_2d(dx_2d, ctx.orig_shape)
            return dx, dw, db, None
        else:
            x, weight, bias, mean, rstd = ctx.saved_tensors
            N = x.shape[-1]
            x_f32 = x.float()
            dy_f32 = dy_contig.float()
            x_hat = (x_f32 - mean) * rstd

            if weight is not None:
                dy_w = dy_f32 * weight
                dw = (dy_f32 * x_hat).sum(dim=list(range(x.ndim - 1))).to(weight.dtype)
            else:
                dy_w = dy_f32
                dw = None

            if bias is not None:
                db = dy_f32.sum(dim=list(range(x.ndim - 1))).to(bias.dtype)
            else:
                db = None

            s1 = dy_w.sum(dim=-1, keepdim=True)
            s2 = (dy_w * x_hat).sum(dim=-1, keepdim=True)
            dx = (rstd * (dy_w - (s1 / N) - (x_hat / N) * s2)).to(x.dtype)
            return dx, dw, db, None


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer Normalization functional interface.

    Args:
        x: Input tensor (*, D).
        weight: Optional scale parameter (D,).
        bias: Optional bias parameter (D,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor with same shape and dtype as x.
    """
    return LayerNormFunction.apply(x, weight, bias, eps)


class LayerNorm(nn.Module):
    """Standard Layer Normalization module."""

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class FusedAddNormFunction(torch.autograd.Function):
    """Autograd function executing fused residual addition and normalization."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-6,
        norm_type: str = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.eps = eps
        ctx.norm_type = norm_type
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None

        if use_triton:
            x_2d, orig_shape = flatten_to_2d(x)
            res_2d, _ = flatten_to_2d(residual)
            w_contig = ensure_contiguous(weight) if weight is not None else None
            b_contig = ensure_contiguous(bias) if bias is not None else None

            y_2d, res_out_2d, rstd, mean = fused_add_norm_fwd(
                x_2d, res_2d, w_contig, b_contig, eps=eps, norm_type=norm_type
            )
            ctx.orig_shape = orig_shape
            ctx.save_for_backward(res_out_2d, w_contig, b_contig, rstd, mean)
            return unflatten_from_2d(y_2d, orig_shape), unflatten_from_2d(res_out_2d, orig_shape)
        else:
            res_out = x + residual
            orig_dtype = res_out.dtype
            res_f32 = res_out.float()

            if norm_type == "rms_norm":
                var = res_f32.pow(2).mean(dim=-1, keepdim=True)
                rstd = torch.rsqrt(var + eps)
                normed = (res_f32 * rstd).to(orig_dtype)
                out = normed * weight if weight is not None else normed
                ctx.save_for_backward(res_out, weight, bias, rstd, None)
            else:
                mean = res_f32.mean(dim=-1, keepdim=True)
                var = ((res_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
                rstd = torch.rsqrt(var + eps)
                normed = ((res_f32 - mean) * rstd).to(orig_dtype)
                out = normed
                if weight is not None:
                    out = out * weight
                if bias is not None:
                    out = out + bias
                ctx.save_for_backward(res_out, weight, bias, rstd, mean)

            return out, res_out

    @staticmethod
    def backward(
        ctx: Any, dy: torch.Tensor, dres_out: torch.Tensor | None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        dy_contig = ensure_contiguous(dy)
        dres_contig = ensure_contiguous(dres_out) if dres_out is not None else None

        if ctx.use_triton:
            res_out_2d, w_contig, b_contig, rstd, mean = ctx.saved_tensors
            dy_2d, _ = flatten_to_2d(dy_contig)
            dres_2d, _ = flatten_to_2d(dres_contig) if dres_contig is not None else (None, None)

            dx_2d, dres_2d, dw, db = fused_add_norm_bwd(
                dy=dy_2d,
                dres_out=dres_2d,
                res_out=res_out_2d,
                weight=w_contig,
                bias=b_contig,
                rstd=rstd,
                mean=mean,
                eps=ctx.eps,
                norm_type=ctx.norm_type,
            )
            dx = unflatten_from_2d(dx_2d, ctx.orig_shape)
            dres = unflatten_from_2d(dres_2d, ctx.orig_shape)
            return dx, dres, dw, db, None, None
        else:
            res_out, weight, bias, rstd, mean = ctx.saved_tensors
            N = res_out.shape[-1]
            res_f32 = res_out.float()
            dy_f32 = dy_contig.float()

            if ctx.norm_type == "rms_norm":
                x_hat = res_f32 * rstd
                if weight is not None:
                    dy_w = dy_f32 * weight
                    dw = (dy_f32 * x_hat).sum(dim=list(range(res_out.ndim - 1))).to(weight.dtype)
                else:
                    dy_w = dy_f32
                    dw = None
                db = None

                sum_dy_x_hat = (dy_w * x_hat).sum(dim=-1, keepdim=True)
                dres_norm = (rstd * (dy_w - (x_hat / N) * sum_dy_x_hat)).to(res_out.dtype)
            else:
                x_hat = (res_f32 - mean) * rstd
                if weight is not None:
                    dy_w = dy_f32 * weight
                    dw = (dy_f32 * x_hat).sum(dim=list(range(res_out.ndim - 1))).to(weight.dtype)
                else:
                    dy_w = dy_f32
                    dw = None

                if bias is not None:
                    db = dy_f32.sum(dim=list(range(res_out.ndim - 1))).to(bias.dtype)
                else:
                    db = None

                s1 = dy_w.sum(dim=-1, keepdim=True)
                s2 = (dy_w * x_hat).sum(dim=-1, keepdim=True)
                dres_norm = (rstd * (dy_w - (s1 / N) - (x_hat / N) * s2)).to(res_out.dtype)

            d_total = dres_norm if dres_contig is None else (dres_norm + dres_contig)
            return d_total, d_total, dw, db, None, None


def fused_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual addition and normalization.

    Computes:
        res_out = x + residual
        norm_out = Norm(res_out)

    Args:
        x: Input tensor (*, D).
        residual: Residual connection tensor (*, D).
        weight: Optional scale parameter (D,).
        bias: Optional bias parameter (D,).
        eps: Epsilon constant.
        norm_type: "rms_norm" or "layer_norm".

    Returns:
        tuple (norm_out, res_out)
    """
    return FusedAddNormFunction.apply(x, residual, weight, bias, eps, norm_type)


# Register custom_ops for torch.compile / dynamo support without graph breaks
try:
    @torch.library.custom_op("stackkernels::rms_norm_op", mutates_args=())
    def _rms_norm_op(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        return rms_norm(x, weight, eps)

    @_rms_norm_op.register_fake
    def _rms_norm_op_fake(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("stackkernels::layer_norm_op", mutates_args=())
    def _layer_norm_op(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        return layer_norm(x, weight, bias, eps)

    @_layer_norm_op.register_fake
    def _layer_norm_op_fake(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("stackkernels::fused_add_norm_op", mutates_args=())
    def _fused_add_norm_op(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        norm_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_add_norm(x, residual, weight, bias, eps, norm_type)

    @_fused_add_norm_op.register_fake
    def _fused_add_norm_op_fake(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        norm_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(x), torch.empty_like(x)
except Exception:
    pass


__all__ = [
    "rms_norm",
    "RMSNorm",
    "RMSNormFunction",
    "layer_norm",
    "LayerNorm",
    "LayerNormFunction",
    "fused_add_norm",
    "FusedAddNormFunction",
]
