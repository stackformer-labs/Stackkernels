"""Loss functions in JAX with custom_vjp support."""

from typing import Any, Literal

from stackkernels._reference.jax.loss import (
    cross_entropy as ref_cross_entropy,
    fused_linear_cross_entropy as ref_fused_linear_cross_entropy,
)


def cross_entropy(
    logits: Any,
    target: Any,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Any:
    """Cross entropy loss in JAX with ignore_index and label smoothing.

    Args:
        logits: Unnormalized log probabilities (*, V).
        target: Target class indices (*).
        ignore_index: Target value that is ignored and does not contribute to gradient. Default: -100.
        label_smoothing: Label smoothing factor in [0.0, 1.0]. Default: 0.0.
        reduction: Specifies reduction to apply: "mean", "sum", or "none". Default: "mean".

    Returns:
        Computed loss array.
    """
    return ref_cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def fused_linear_cross_entropy(
    x: Any,
    weight: Any,
    target: Any,
    bias: Any | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Any:
    """Fused linear projection + cross entropy in JAX.

    Args:
        x: Input activation array (*, K).
        weight: Embedding / LM-head weight array (V, K).
        target: Target class indices (*).
        bias: Optional bias array (V,). Default: None.
        ignore_index: Target index to ignore. Default: -100.
        label_smoothing: Smoothing factor in [0.0, 1.0]. Default: 0.0.
        reduction: "mean", "sum", or "none". Default: "mean".

    Returns:
        Computed loss array.
    """
    return ref_fused_linear_cross_entropy(
        x,
        weight,
        target,
        bias=bias,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


__all__ = ["cross_entropy", "fused_linear_cross_entropy"]

