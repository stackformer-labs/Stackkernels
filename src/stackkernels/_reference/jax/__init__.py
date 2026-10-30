"""JAX reference implementations."""

from stackkernels._reference.jax.activation import (
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
from stackkernels._reference.jax.attention import attention_reference_jax, repeat_kv_jax
from stackkernels._reference.jax.loss import cross_entropy, fused_linear_cross_entropy
from stackkernels._reference.jax.norm import fused_add_norm, layer_norm, rms_norm
from stackkernels._reference.jax.rope import apply_rope, precompute_freqs

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
    "attention_reference_jax",
    "repeat_kv_jax",
]
