"""PyTorch wrappers for Cross Entropy and Fused Linear Cross Entropy."""

from typing import Any, Literal

import torch
import torch.nn as nn

from stackkernels._core.loss.cross_entropy import cross_entropy_bwd, cross_entropy_fwd
from stackkernels._core.loss.fused_linear_ce import (
    fused_linear_cross_entropy_bwd,
    fused_linear_cross_entropy_fwd,
)
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import ensure_contiguous, flatten_to_2d, unflatten_from_2d


class CrossEntropyFunction(torch.autograd.Function):
    """Autograd function executing Triton or Reference Cross Entropy Loss."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        use_triton = should_use_triton(logits)
        ctx.use_triton = use_triton
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.orig_logits_shape = logits.shape
        ctx.orig_target_shape = target.shape
        ctx.orig_dtype = logits.dtype

        logits_contig = ensure_contiguous(logits)
        target_contig = ensure_contiguous(target)

        vocab_size = logits_contig.shape[-1]
        logits_2d = logits_contig.reshape(-1, vocab_size)
        target_1d = target_contig.reshape(-1)

        valid_mask = target_1d != ignore_index
        n_valid = int(valid_mask.sum().clamp(min=1).item())
        ctx.n_valid = n_valid

        if use_triton:
            loss_1d, lse_1d = cross_entropy_fwd(
                logits_2d,
                target_1d,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
            ctx.save_for_backward(logits_2d, target_1d, lse_1d)
        else:
            logits_f32 = logits_2d.float()
            lse_1d = torch.logsumexp(logits_f32, dim=-1)
            safe_targets = torch.where(valid_mask, target_1d, torch.zeros_like(target_1d))
            target_logits = logits_f32.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            nll = lse_1d - target_logits

            if label_smoothing > 0.0:
                smooth_loss = lse_1d - logits_f32.mean(dim=-1)
                loss_1d = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
            else:
                loss_1d = nll

            loss_1d = torch.where(valid_mask, loss_1d, torch.zeros_like(loss_1d))
            ctx.save_for_backward(logits_2d, target_1d, lse_1d)

        if reduction == "mean":
            return loss_1d.sum() / n_valid
        elif reduction == "sum":
            return loss_1d.sum()
        else:
            return loss_1d.reshape(target.shape)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        logits_2d, target_1d, lse_1d = ctx.saved_tensors
        grad_contig = ensure_contiguous(grad_output)
        scale_grad = (1.0 / ctx.n_valid) if ctx.reduction == "mean" else 1.0

        if ctx.use_triton:
            dlogits_2d = cross_entropy_bwd(
                grad_contig,
                logits_2d,
                target_1d,
                lse_1d,
                ignore_index=ctx.ignore_index,
                label_smoothing=ctx.label_smoothing,
                scale_grad=scale_grad,
            )
        else:
            V = logits_2d.shape[-1]
            p = torch.softmax(logits_2d.float(), dim=-1)
            is_target = target_1d.unsqueeze(-1) == torch.arange(V, device=logits_2d.device)
            valid = (target_1d != ctx.ignore_index).unsqueeze(-1)
            smooth_prob = ctx.label_smoothing / V
            if ctx.label_smoothing > 0.0:
                q = torch.where(is_target, 1.0 - ctx.label_smoothing + smooth_prob, smooth_prob)
            else:
                q = torch.where(is_target, 1.0, 0.0)

            dz = torch.where(valid, p - q, torch.zeros_like(p))
            if grad_contig.dim() > 0 and grad_contig.numel() > 1:
                dy = grad_contig.reshape(-1, 1).float() * scale_grad
            else:
                dy = grad_contig.float() * scale_grad
            dlogits_2d = (dz * dy).to(ctx.orig_dtype)

        return (
            dlogits_2d.reshape(ctx.orig_logits_shape),
            None,
            None,
            None,
            None,
        )


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute Cross Entropy loss using Triton kernels or PyTorch fallback.

    Args:
        logits: Input logits (*, V).
        target: Target class indices (*).
        ignore_index: Target index to ignore.
        label_smoothing: Label smoothing constant in [0.0, 1.0].
        reduction: "mean", "sum", or "none".

    Returns:
        Loss tensor.
    """
    return CrossEntropyFunction.apply(
        logits, target, ignore_index, label_smoothing, reduction
    )


class CrossEntropyLoss(nn.Module):
    """Cross Entropy loss module backed by Triton kernels.

    Computes cross entropy using an online softmax algorithm to maintain numerical
    stability while reducing intermediate memory traffic.

    Args:
        ignore_index: Target index to ignore in loss computation. Default: -100.
        label_smoothing: Smoothing factor in [0.0, 1.0]. Default: 0.0.
        reduction: Reduction to apply: "mean", "sum", or "none". Default: "mean".

    Examples:
        >>> criterion = CrossEntropyLoss()
        >>> logits = torch.randn(2, 512, 32000)
        >>> target = torch.randint(0, 32000, (2, 512))
        >>> loss = criterion(logits, target)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Autograd function executing Fused Linear + Cross Entropy without full logits."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.orig_x_shape = x.shape
        ctx.orig_target_shape = target.shape
        ctx.has_bias = bias is not None
        ctx.orig_dtype = x.dtype

        x_contig = ensure_contiguous(x)
        w_contig = ensure_contiguous(weight)
        t_contig = ensure_contiguous(target)
        b_contig = ensure_contiguous(bias) if bias is not None else None

        K = x_contig.shape[-1]
        x_2d = x_contig.reshape(-1, K)
        t_1d = t_contig.reshape(-1)

        valid_mask = t_1d != ignore_index
        n_valid = int(valid_mask.sum().clamp(min=1).item())
        ctx.n_valid = n_valid

        if use_triton:
            loss_1d, lse_1d = fused_linear_cross_entropy_fwd(
                x_2d,
                w_contig,
                t_1d,
                bias=b_contig,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )
            ctx.save_for_backward(x_2d, w_contig, t_1d, lse_1d, b_contig)
        else:
            logits_2d = torch.matmul(x_2d.float(), w_contig.t().float())
            if b_contig is not None:
                logits_2d += b_contig.float()

            lse_1d = torch.logsumexp(logits_2d, dim=-1)
            safe_targets = torch.where(valid_mask, t_1d, torch.zeros_like(t_1d))
            target_logits = logits_2d.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            nll = lse_1d - target_logits

            if label_smoothing > 0.0:
                smooth_loss = lse_1d - logits_2d.mean(dim=-1)
                loss_1d = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
            else:
                loss_1d = nll

            loss_1d = torch.where(valid_mask, loss_1d, torch.zeros_like(loss_1d))
            ctx.save_for_backward(x_2d, w_contig, t_1d, lse_1d, b_contig)

        if reduction == "mean":
            return loss_1d.sum() / n_valid
        elif reduction == "sum":
            return loss_1d.sum()
        else:
            return loss_1d.reshape(target.shape)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None, None, None, None]:
        x_2d, w_contig, t_1d, lse_1d, b_contig = ctx.saved_tensors
        grad_contig = ensure_contiguous(grad_output)
        scale_grad = (1.0 / ctx.n_valid) if ctx.reduction == "mean" else 1.0

        if ctx.use_triton:
            dx_2d, dw, db = fused_linear_cross_entropy_bwd(
                grad_contig,
                x_2d,
                w_contig,
                t_1d,
                lse_1d,
                bias=b_contig,
                ignore_index=ctx.ignore_index,
                label_smoothing=ctx.label_smoothing,
                scale_grad=scale_grad,
            )
        else:
            logits_2d = torch.matmul(x_2d.float(), w_contig.t().float())
            if b_contig is not None:
                logits_2d += b_contig.float()

            V = w_contig.shape[0]
            p = torch.softmax(logits_2d, dim=-1)
            is_target = t_1d.unsqueeze(-1) == torch.arange(V, device=x_2d.device)
            valid = (t_1d != ctx.ignore_index).unsqueeze(-1)
            smooth_prob = ctx.label_smoothing / V
            if ctx.label_smoothing > 0.0:
                q = torch.where(is_target, 1.0 - ctx.label_smoothing + smooth_prob, smooth_prob)
            else:
                q = torch.where(is_target, 1.0, 0.0)

            dz = torch.where(valid, p - q, torch.zeros_like(p))
            if grad_contig.dim() > 0 and grad_contig.numel() > 1:
                dy = grad_contig.reshape(-1, 1).float() * scale_grad
            else:
                dy = grad_contig.float() * scale_grad
            dz = dz * dy

            dx_2d = torch.matmul(dz.to(w_contig.dtype), w_contig)
            dw = torch.matmul(dz.t().to(w_contig.dtype), x_2d)
            db = dz.sum(dim=0).to(b_contig.dtype) if b_contig is not None else None

        dx = dx_2d.reshape(ctx.orig_x_shape).to(ctx.orig_dtype)
        return dx, dw, None, db, None, None, None


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute Fused Linear Projection + Cross Entropy loss without materializing full logits.

    Args:
        x: Input activations (*, K).
        weight: Embedding / LM-head weight (V, K).
        target: Target class indices (*).
        bias: Optional bias parameter (V,).
        ignore_index: Target index to ignore.
        label_smoothing: Label smoothing factor in [0.0, 1.0].
        reduction: "mean", "sum", or "none".

    Returns:
        Loss tensor.
    """
    return FusedLinearCrossEntropyFunction.apply(
        x, weight, target, bias, ignore_index, label_smoothing, reduction
    )


class FusedLinearCrossEntropyLoss(nn.Module):
    """Fused Linear Projection + Cross Entropy loss module.

    Fuses the final LM head projection ($x W^T$) directly into the online softmax
    cross-entropy loss loop, eliminating materialization of the full logits tensor
    and significantly cutting peak activation memory during training.

    Args:
        ignore_index: Target index to ignore in loss computation. Default: -100.
        label_smoothing: Smoothing factor in [0.0, 1.0]. Default: 0.0.
        reduction: Reduction to apply: "mean", "sum", or "none". Default: "mean".

    Examples:
        >>> criterion = FusedLinearCrossEntropyLoss()
        >>> hidden_states = torch.randn(2, 512, 4096)
        >>> lm_head_weight = torch.randn(32000, 4096)
        >>> target = torch.randint(0, 32000, (2, 512))
        >>> loss = criterion(hidden_states, lm_head_weight, target)
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return fused_linear_cross_entropy(
            x,
            weight,
            target,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


# Register torch.library.custom_op if available
try:

    @torch.library.custom_op("stackkernels::cross_entropy_op", mutates_args=())
    def _cross_entropy_op(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        label_smoothing: float,
        reduction: str,
    ) -> torch.Tensor:
        return cross_entropy(logits, target, ignore_index, label_smoothing, reduction)

    @_cross_entropy_op.register_fake
    def _cross_entropy_op_fake(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        label_smoothing: float,
        reduction: str,
    ) -> torch.Tensor:
        if reduction == "none":
            return torch.empty(target.shape, dtype=logits.dtype, device=logits.device)
        return torch.empty((), dtype=logits.dtype, device=logits.device)

    @torch.library.custom_op("stackkernels::fused_linear_ce_op", mutates_args=())
    def _fused_linear_ce_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        ignore_index: int,
        label_smoothing: float,
        reduction: str,
    ) -> torch.Tensor:
        return fused_linear_cross_entropy(
            x, weight, target, bias, ignore_index, label_smoothing, reduction
        )

    @_fused_linear_ce_op.register_fake
    def _fused_linear_ce_op_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None,
        ignore_index: int,
        label_smoothing: float,
        reduction: str,
    ) -> torch.Tensor:
        if reduction == "none":
            return torch.empty(target.shape, dtype=x.dtype, device=x.device)
        return torch.empty((), dtype=x.dtype, device=x.device)
except Exception:
    pass


__all__ = [
    "cross_entropy",
    "CrossEntropyLoss",
    "CrossEntropyFunction",
    "fused_linear_cross_entropy",
    "FusedLinearCrossEntropyLoss",
    "FusedLinearCrossEntropyFunction",
]
