"""JAX public API for stackkernels."""

from stackkernels.jax.activation import (
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
from stackkernels.jax.embedding import apply_rope, precompute_freqs
from stackkernels.jax.loss import cross_entropy, fused_linear_cross_entropy
from stackkernels.jax.norm import fused_add_norm, layer_norm, rms_norm

__all__ = [
    "rms_norm",
    "layer_norm",
    "fused_add_norm",
    "swiglu",
    "geglu",
    "reglu",
    "glu",
    "gelu",
    "silu",
    "relu",
    "leaky_relu",
    "sigmoid",
    "apply_rope",
    "precompute_freqs",
    "cross_entropy",
    "fused_linear_cross_entropy",
]
