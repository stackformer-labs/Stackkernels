"""swiglu, geglu, gelu, silu, relu, leaky_relu, sigmoid for PyTorch."""

from typing import Any, Literal

import torch
import torch.nn.functional as F

from stackkernels._core.activation.elementwise import elementwise_bwd, elementwise_fwd
from stackkernels._core.activation.glu import glu_bwd, glu_fwd
from stackkernels.torch._dispatch import should_use_triton
from stackkernels.torch._utils import ensure_contiguous


class GLUFunction(torch.autograd.Function):
    """Autograd function executing fused GLU (SwiGLU, GeGLU, ReGLU)."""

    @staticmethod
    def forward(
        ctx: Any,
        gate: torch.Tensor,
        up: torch.Tensor,
        act: str = "silu",
    ) -> torch.Tensor:
        use_triton = should_use_triton(gate)
        ctx.use_triton = use_triton
        ctx.act = act

        gate_c = ensure_contiguous(gate)
        up_c = ensure_contiguous(up)

        if use_triton:
            y = glu_fwd(gate_c, up_c, act=act)
            ctx.save_for_backward(gate_c, up_c)
            return y
        else:
            g_f32 = gate_c.float()
            u_f32 = up_c.float()
            if act in ("silu", "swiglu"):
                act_g = F.silu(g_f32)
            elif act in ("gelu", "geglu"):
                act_g = F.gelu(g_f32, approximate="tanh")
            else:
                act_g = F.relu(g_f32)
            y = (act_g * u_f32).to(gate.dtype)
            ctx.save_for_backward(gate_c, up_c)
            return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        dy_c = ensure_contiguous(dy)
        gate_c, up_c = ctx.saved_tensors

        if ctx.use_triton:
            dgate, dup = glu_bwd(dy_c, gate_c, up_c, act=ctx.act)
            return dgate, dup, None
        else:
            # Recompute on-chip/in-memory
            g = gate_c.float()
            u = up_c.float()
            dy_f = dy_c.float()

            if ctx.act in ("silu", "swiglu"):
                s = torch.sigmoid(g)
                act_g = g * s
                d_act_g = s * (1.0 + g * (1.0 - s))
            elif ctx.act in ("gelu", "geglu"):
                # Approximate tanh derivative matching Triton
                k = 0.79788456 * (g + 0.044715 * g * g * g)
                t = torch.tanh(k)
                act_g = 0.5 * g * (1.0 + t)
                dk = 0.79788456 * (1.0 + 3.0 * 0.044715 * g * g)
                d_act_g = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * dk
            else:
                act_g = torch.relu(g)
                d_act_g = (g > 0.0).float()

            dup = (dy_f * act_g).to(up_c.dtype)
            dgate = (dy_f * u * d_act_g).to(gate_c.dtype)
            return dgate, dup, None


class ElementwiseFunction(torch.autograd.Function):
    """Autograd function executing fused elementwise activations."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        act: str = "silu",
        approximate: str = "none",
        alpha: float = 0.01,
    ) -> torch.Tensor:
        use_triton = should_use_triton(x)
        ctx.use_triton = use_triton
        ctx.act = act
        ctx.approximate = approximate
        ctx.alpha = alpha

        x_c = ensure_contiguous(x)
        if use_triton:
            y = elementwise_fwd(x_c, act=act, approximate=approximate, alpha=alpha)
            ctx.save_for_backward(x_c)
            return y
        else:
            if act == "gelu":
                y = F.gelu(x_c, approximate=approximate)
            elif act == "silu":
                y = F.silu(x_c)
            elif act == "relu":
                y = F.relu(x_c)
            elif act == "leaky_relu":
                y = F.leaky_relu(x_c, negative_slope=alpha)
            elif act == "sigmoid":
                y = torch.sigmoid(x_c)
            else:
                raise ValueError(f"Unknown activation: {act}")
            ctx.save_for_backward(x_c)
            return y

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        dy_c = ensure_contiguous(dy)
        (x_c,) = ctx.saved_tensors

        if ctx.use_triton:
            dx = elementwise_bwd(
                dy_c, x_c, act=ctx.act, approximate=ctx.approximate, alpha=ctx.alpha
            )
            return dx, None, None, None
        else:
            x_f = x_c.float()
            dy_f = dy_c.float()
            act = ctx.act

            if act == "gelu":
                if ctx.approximate == "tanh":
                    k = 0.79788456 * (x_f + 0.044715 * x_f * x_f * x_f)
                    t = torch.tanh(k)
                    dk = 0.79788456 * (1.0 + 3.0 * 0.044715 * x_f * x_f)
                    grad = 0.5 * (1.0 + t) + 0.5 * x_f * (1.0 - t * t) * dk
                else:
                    INV_SQRT2 = 0.7071067811865475
                    INV_SQRT_2PI = 0.3989422804014327
                    cdf = 0.5 * (1.0 + torch.erf(x_f * INV_SQRT2))
                    pdf = INV_SQRT_2PI * torch.exp(-0.5 * x_f * x_f)
                    grad = cdf + x_f * pdf
            elif act == "silu":
                s = torch.sigmoid(x_f)
                grad = s * (1.0 + x_f * (1.0 - s))
            elif act == "relu":
                grad = (x_f > 0.0).float()
            elif act == "leaky_relu":
                grad = torch.where(x_f > 0.0, 1.0, ctx.alpha)
            elif act == "sigmoid":
                s = torch.sigmoid(x_f)
                grad = s * (1.0 - s)
            else:
                raise ValueError(f"Unknown activation: {act}")

            dx = (dy_f * grad).to(x_c.dtype)
            return dx, None, None, None


def _split_if_needed(gate: torch.Tensor, up: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if up is None:
        return torch.chunk(gate, 2, dim=-1)
    return gate, up


# --- Public Functional API ---

def swiglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up.

    If `up` is None, `gate` is split into two halves along its last dimension.

    Args:
        gate: Gating tensor (*, D) or combined tensor (*, 2*D) if up is None.
        up: Optional up-projection tensor (*, D).

    Returns:
        Activated tensor (*, D).

    Examples:
        >>> gate = torch.randn(2, 512, 2048)
        >>> up = torch.randn(2, 512, 2048)
        >>> out = swiglu(gate, up)
    """
    g, u = _split_if_needed(gate, up)
    return GLUFunction.apply(g, u, "silu")


def geglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """GeGLU activation: gelu(gate) * up (using tanh approximation).

    If `up` is None, `gate` is split into two halves along its last dimension.

    Args:
        gate: Gating tensor (*, D) or combined tensor (*, 2*D) if up is None.
        up: Optional up-projection tensor (*, D).

    Returns:
        Activated tensor (*, D).

    Examples:
        >>> gate = torch.randn(2, 512, 2048)
        >>> up = torch.randn(2, 512, 2048)
        >>> out = geglu(gate, up)
    """
    g, u = _split_if_needed(gate, up)
    return GLUFunction.apply(g, u, "gelu")


def reglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """ReGLU activation: relu(gate) * up.

    If `up` is None, `gate` is split into two halves along its last dimension.

    Args:
        gate: Gating tensor (*, D) or combined tensor (*, 2*D) if up is None.
        up: Optional up-projection tensor (*, D).

    Returns:
        Activated tensor (*, D).

    Examples:
        >>> gate = torch.randn(2, 512, 2048)
        >>> up = torch.randn(2, 512, 2048)
        >>> out = reglu(gate, up)
    """
    g, u = _split_if_needed(gate, up)
    return GLUFunction.apply(g, u, "relu")


def glu(
    gate: torch.Tensor,
    up: torch.Tensor | None = None,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> torch.Tensor:
    """General Gated Linear Unit dispatcher.

    Args:
        gate: Gating tensor (*, D) or combined tensor (*, 2*D) if up is None.
        up: Optional up-projection tensor (*, D).
        act: Activation function to apply to gate: "silu", "gelu", or "relu".

    Returns:
        Activated tensor (*, D).
    """
    g, u = _split_if_needed(gate, up)
    return GLUFunction.apply(g, u, act)


def gelu(x: torch.Tensor, approximate: Literal["none", "tanh"] = "none") -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU).

    Args:
        x: Input tensor (*, D).
        approximate: Approximation algorithm, either "none" (exact erf) or "tanh".

    Returns:
        Activated tensor with identical shape and dtype.
    """
    return ElementwiseFunction.apply(x, "gelu", approximate, 0.0)


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).

    Args:
        x: Input tensor (*, D).

    Returns:
        Activated tensor with identical shape and dtype.
    """
    return ElementwiseFunction.apply(x, "silu", "none", 0.0)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit (ReLU): max(0, x).

    Args:
        x: Input tensor (*, D).

    Returns:
        Activated tensor with identical shape and dtype.
    """
    return ElementwiseFunction.apply(x, "relu", "none", 0.0)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Leaky Rectified Linear Unit: max(x, x * negative_slope).

    Args:
        x: Input tensor (*, D).
        negative_slope: Slope for negative values. Default: 0.01.

    Returns:
        Activated tensor with identical shape and dtype.
    """
    return ElementwiseFunction.apply(x, "leaky_relu", "none", negative_slope)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x)).

    Args:
        x: Input tensor (*, D).

    Returns:
        Activated tensor with identical shape and dtype.
    """
    return ElementwiseFunction.apply(x, "sigmoid", "none", 0.0)


# Custom ops registration for torch.compile
try:
    @torch.library.custom_op("stackkernels::glu_op", mutates_args=())
    def _glu_op(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
        return GLUFunction.apply(gate, up, act)

    @_glu_op.register_fake
    def _glu_op_fake(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
        return torch.empty_like(gate)

    @torch.library.custom_op("stackkernels::elementwise_op", mutates_args=())
    def _elementwise_op(x: torch.Tensor, act: str, approximate: str, alpha: float) -> torch.Tensor:
        return ElementwiseFunction.apply(x, act, approximate, alpha)

    @_elementwise_op.register_fake
    def _elementwise_op_fake(x: torch.Tensor, act: str, approximate: str, alpha: float) -> torch.Tensor:
        return torch.empty_like(x)
except Exception:
    pass


__all__ = [
    "swiglu",
    "geglu",
    "reglu",
    "glu",
    "gelu",
    "silu",
    "relu",
    "leaky_relu",
    "sigmoid",
    "GLUFunction",
    "ElementwiseFunction",
]
