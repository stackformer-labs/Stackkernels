"""Basic JAX usage example for StackKernels.

Demonstrates:
1. Functional APIs with JAX arrays (rms_norm, swiglu, flash_attention, cross_entropy)
2. custom_vjp automatic differentiation
"""

import jax
import jax.numpy as jnp
import stackkernels
from stackkernels.jax import (
    apply_rope,
    cross_entropy,
    flash_attention,
    precompute_freqs,
    rms_norm,
    swiglu,
)


def main() -> None:
    print(f"StackKernels version: {stackkernels.__version__}")
    print(f"JAX backend:          {jax.default_backend()}")

    batch, seq_len, dim = 2, 64, 256

    # 1. Normalization with JIT and Autodiff
    x = jnp.ones((batch, seq_len, dim), dtype=jnp.float32)
    weight = jnp.ones((dim,), dtype=jnp.float32)

    def loss_fn(x_in, w_in):
        normed = rms_norm(x_in, w_in)
        return jnp.sum(normed)

    grad_x, grad_w = jax.grad(loss_fn, argnums=(0, 1))(x, weight)
    print(f"[1] JAX rms_norm grad_x shape: {grad_x.shape}, grad_w shape: {grad_w.shape}")

    # 2. Activation (SwiGLU)
    gate = jnp.ones((batch, seq_len, dim), dtype=jnp.float32)
    up = jnp.ones((batch, seq_len, dim), dtype=jnp.float32)
    act = swiglu(gate, up)
    print(f"[2] JAX swiglu output shape:   {act.shape}")

    # 3. RoPE
    num_heads, head_dim = 4, 64
    q = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
    k = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
    cos, sin = precompute_freqs(dim=head_dim, max_seq_len=seq_len + 64)
    q_rot, k_rot = apply_rope(q, k, cos=cos, sin=sin)
    print(f"[3] JAX RoPE output q_rot:     {q_rot.shape}")

    # 4. Attention
    v = jnp.ones((batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
    attn = flash_attention(q_rot, k_rot, v, is_causal=True)
    print(f"[4] JAX flash_attention shape: {attn.shape}")

    # 5. Loss
    vocab_size = 1024
    logits = jnp.ones((batch * seq_len, vocab_size), dtype=jnp.float32)
    targets = jnp.zeros((batch * seq_len,), dtype=jnp.int32)
    loss_val = cross_entropy(logits, targets)
    print(f"[5] JAX cross_entropy loss:    {loss_val.item():.4f}")


if __name__ == "__main__":
    main()

