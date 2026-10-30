"""Loss reference implementations in PyTorch."""

from typing import Literal

import torch
import torch.nn.functional as F


def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Cross entropy loss reference in PyTorch with ignore_index and label smoothing."""
    # Flatten to (N, V) and (N,)
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    target_flat = target.reshape(-1)

    loss = F.cross_entropy(
        logits_flat.float(),
        target_flat,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    if reduction == "none":
        return loss.reshape(target.shape)
    return loss


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Fused linear projection + cross entropy reference in PyTorch."""
    logits = F.linear(x, weight, bias)
    return cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


__all__ = ["cross_entropy", "fused_linear_cross_entropy"]

