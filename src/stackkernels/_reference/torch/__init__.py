"""PyTorch reference implementations."""

from stackkernels._reference.torch.activation import (
    geglu,
    gelu,
    glu,
    leaky_relu,
    reglu,
    relu,
    sigmoid,
    silu,
    swiglu,
)
from stackkernels._reference.torch.norm import fused_add_norm, layer_norm, rms_norm
from stackkernels._reference.torch.rope import apply_rope, precompute_freqs

__all__ = [
    "rms_norm",
    "layer_norm",
    "fused_add_norm",
    "gelu",
    "silu",
    "relu",
    "leaky_relu",
    "sigmoid",
    "swiglu",
    "geglu",
    "reglu",
    "glu",
    "apply_rope",
    "precompute_freqs",
]
