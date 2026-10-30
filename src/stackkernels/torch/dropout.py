"""Dropout and fused dropout + residual add for PyTorch."""

from typing import Any

import torch
import torch.nn as nn

from stackkernels._core.dropout.dropout import dropout_bwd, dropout_fwd
from stackkernels._core.dropout.fused_dropout_add import (
    fused_dropout_add_bwd,
    fused_dropout_add_fwd,
)
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import ensure_contiguous


class DropoutFunction(torch.autograd.Function):
    """Autograd function executing fused low-memory dropout."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        p: float = 0.5,
        training: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        if not training or p <= 0.0:
            ctx.training = False
            ctx.p = 0.0
            ctx.seed = 0
            ctx.use_triton = False
            return x

        if p >= 1.0:
            ctx.training = True
            ctx.p = 1.0
            ctx.seed = 0
            ctx.use_triton = False
            return torch.zeros_like(x)

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        use_triton = should_use_triton(x)
        ctx.training = training
        ctx.p = p
        ctx.seed = seed
        ctx.use_triton = use_triton

        x_c = ensure_contiguous(x)
        if use_triton:
            return dropout_fwd(x_c, p=p, seed=seed)
        else:
            gen = torch.Generator(device=x.device).manual_seed(seed)
            rand = torch.rand(x.shape, generator=gen, device=x.device, dtype=torch.float32)
            mask = (rand > p).to(x.dtype)
            scale = 1.0 / (1.0 - p)
            return x_c * mask * scale

    @staticmethod
    def backward(
        ctx: Any, dy: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None]:
        if not ctx.training or ctx.p <= 0.0:
            return dy, None, None, None

        if ctx.p >= 1.0:
            return torch.zeros_like(dy), None, None, None

        dy_c = ensure_contiguous(dy)
        if ctx.use_triton:
            dx = dropout_bwd(dy_c, p=ctx.p, seed=ctx.seed)
            return dx, None, None, None
        else:
            gen = torch.Generator(device=dy.device).manual_seed(ctx.seed)
            rand = torch.rand(dy.shape, generator=gen, device=dy.device, dtype=torch.float32)
            mask = (rand > ctx.p).to(dy.dtype)
            scale = 1.0 / (1.0 - ctx.p)
            dx = dy_c * mask * scale
            return dx, None, None, None


class FusedDropoutAddFunction(torch.autograd.Function):
    """Autograd function executing fused low-memory dropout with residual addition."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor,
        p: float = 0.5,
        training: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        if not training or p <= 0.0:
            ctx.training = False
            ctx.p = 0.0
            ctx.seed = 0
            ctx.use_triton = False
            return x + residual

        if p >= 1.0:
            ctx.training = True
            ctx.p = 1.0
            ctx.seed = 0
            ctx.use_triton = False
            return residual.clone()

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        use_triton = should_use_triton(x)
        ctx.training = training
        ctx.p = p
        ctx.seed = seed
        ctx.use_triton = use_triton

        x_c = ensure_contiguous(x)
        res_c = ensure_contiguous(residual)

        if use_triton:
            return fused_dropout_add_fwd(x_c, res_c, p=p, seed=seed)
        else:
            gen = torch.Generator(device=x.device).manual_seed(seed)
            rand = torch.rand(x.shape, generator=gen, device=x.device, dtype=torch.float32)
            mask = (rand > p).to(x.dtype)
            scale = 1.0 / (1.0 - p)
            return (x_c * mask * scale) + res_c

    @staticmethod
    def backward(
        ctx: Any, dy: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        if not ctx.training or ctx.p <= 0.0:
            return dy, dy, None, None, None

        if ctx.p >= 1.0:
            return torch.zeros_like(dy), dy, None, None, None

        dy_c = ensure_contiguous(dy)
        if ctx.use_triton:
            dx, dresidual = fused_dropout_add_bwd(dy_c, p=ctx.p, seed=ctx.seed)
            return dx, dresidual, None, None, None
        else:
            gen = torch.Generator(device=dy.device).manual_seed(ctx.seed)
            rand = torch.rand(dy.shape, generator=gen, device=dy.device, dtype=torch.float32)
            mask = (rand > ctx.p).to(dy.dtype)
            scale = 1.0 / (1.0 - ctx.p)
            dx = dy_c * mask * scale
            return dx, dy, None, None, None


def dropout(
    x: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply dropout to input tensor using fused low-memory Philox PRNG.

    Args:
        x: Input tensor (*,).
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True; returns x directly if False. Default: True.
        seed: Optional RNG seed. If provided, ensures deterministic execution.

    Returns:
        Tensor with dropout applied.
    """
    return DropoutFunction.apply(x, p, training, seed)


def fused_dropout_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    """Fused Dropout and Residual Addition.

    Computes:
        y = dropout(x, p) + residual
    in a single pass without allocating intermediate dropped activation or mask.

    Args:
        x: Input tensor (*,).
        residual: Residual tensor matching shape of x.
        p: Probability of an element to be zeroed. Default: 0.5.
        training: Apply dropout if True; returns x + residual if False. Default: True.
        seed: Optional RNG seed.

    Returns:
        Tensor with dropout and residual addition applied.
    """
    return FusedDropoutAddFunction.apply(x, residual, p, training, seed)


class Dropout(nn.Module):
    """Dropout module using StackKernels low-memory kernel.

    Args:
        p: Probability of an element to be zeroed. Default: 0.5.
        inplace: Inplace dropout flag (maintained for API parity with nn.Dropout). Default: False.

    Examples:
        >>> drop = Dropout(p=0.2)
        >>> x = torch.randn(2, 64, 128)
        >>> out = drop(x)
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        return dropout(x, p=self.p, training=self.training, seed=seed)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class FusedDropoutAdd(nn.Module):
    """Fused Dropout and Residual Addition module.

    Computes:
        y = dropout(x, p) + residual

    Args:
        p: Probability of an element to be zeroed. Default: 0.5.

    Examples:
        >>> fused = FusedDropoutAdd(p=0.1)
        >>> x = torch.randn(2, 64, 128)
        >>> res = torch.randn(2, 64, 128)
        >>> out = fused(x, res)
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, seed: int | None = None
    ) -> torch.Tensor:
        return fused_dropout_add(
            x, residual, p=self.p, training=self.training, seed=seed
        )

    def extra_repr(self) -> str:
        return f"p={self.p}"


try:
    @torch.library.custom_op("stackkernels::dropout_op", mutates_args=())
    def _dropout_op(
        x: torch.Tensor, p: float, training: bool, seed: int
    ) -> torch.Tensor:
        return dropout(x, p=p, training=training, seed=seed)

    @_dropout_op.register_fake
    def _dropout_op_fake(
        x: torch.Tensor, p: float, training: bool, seed: int
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("stackkernels::fused_dropout_add_op", mutates_args=())
    def _fused_dropout_add_op(
        x: torch.Tensor, residual: torch.Tensor, p: float, training: bool, seed: int
    ) -> torch.Tensor:
        return fused_dropout_add(x, residual, p=p, training=training, seed=seed)

    @_fused_dropout_add_op.register_fake
    def _fused_dropout_add_op_fake(
        x: torch.Tensor, residual: torch.Tensor, p: float, training: bool, seed: int
    ) -> torch.Tensor:
        return torch.empty_like(x)
except Exception:
    pass


__all__ = [
    "dropout",
    "Dropout",
    "DropoutFunction",
    "fused_dropout_add",
    "FusedDropoutAdd",
    "FusedDropoutAddFunction",
]
