"""Reference implementation of dropout and fused dropout + residual add in PyTorch."""

import torch
import torch.nn.functional as F


def dropout(
    x: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """Reference implementation of dropout.

    Args:
        x: Input tensor (*,).
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True. Default: True.
        seed: Optional RNG seed for deterministic execution.

    Returns:
        Tensor with dropout applied.
    """
    if not training or p == 0.0:
        return x

    if p >= 1.0:
        return torch.zeros_like(x)

    if seed is not None:
        # Deterministic generation using provided seed
        gen = torch.Generator(device=x.device).manual_seed(seed)
        rand = torch.rand(x.shape, generator=gen, device=x.device, dtype=torch.float32)
        mask = (rand > p).to(x.dtype)
        scale = 1.0 / (1.0 - p)
        return x * mask * scale

    return F.dropout(x, p=p, training=training)


def fused_dropout_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """Reference implementation of fused dropout and residual addition.

    Args:
        x: Input tensor (*,).
        residual: Residual tensor matching shape of x.
        p: Dropout probability. Default: 0.5.
        training: Apply dropout if True. Default: True.
        seed: Optional RNG seed.

    Returns:
        dropout(x) + residual
    """
    return dropout(x, p=p, training=training, seed=seed) + residual


__all__ = ["dropout", "fused_dropout_add"]

