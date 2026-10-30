# JAX User Guide and PyTorch Parity

This guide explains how to use StackKernels within JAX pipelines, detailing functional operator parity, automatic differentiation with `custom_vjp`, and differences from the PyTorch frontend.

---

## 1. Overview of the JAX Frontend

The `stackkernels.jax` package provides functionally pure implementations of StackKernels operations. Unlike PyTorch, which provides stateful `nn.Module` classes and autograd functions, the JAX frontend is strictly functional:

- Functions operate on immutable `jax.Array` objects.
- Gradients are defined through `jax.custom_vjp`, enabling seamless composition with `jax.grad`, `jax.value_and_grad`, `jax.jit`, and `jax.vmap`.
- Hardware GPU launches route via `jax.triton_call`, while CPU environments use pure JAX primitives.

---

## 2. API Parity with PyTorch

Every operation available in `stackkernels.torch` has a direct functional equivalent in `stackkernels.jax`:

| Operation Family | JAX Function | PyTorch Equivalent | Gradient Support |
|:---|:---|:---|:---:|
| **Normalization** | `rms_norm(x, weight=None, eps=1e-6)` | `rms_norm` | Yes (`custom_vjp`) |
| | `layer_norm(x, weight=None, bias=None, eps=1e-5)` | `layer_norm` | Yes (`custom_vjp`) |
| | `fused_add_norm(x, res, weight=None, bias=None, eps=1e-6, norm_type="rms_norm")` | `fused_add_norm` | Yes (`custom_vjp`) |
| **Gated Activations** | `swiglu(gate, up)` | `swiglu` | Yes (`custom_vjp`) |
| | `geglu(gate, up, approximate=False)` | `geglu` | Yes (`custom_vjp`) |
| | `reglu(gate, up)` | `reglu` | Yes (`custom_vjp`) |
| | `glu(gate, up, act="silu")` | `glu` | Yes (`custom_vjp`) |
| **Elementwise** | `gelu`, `silu`, `relu`, `leaky_relu`, `sigmoid` | Same | Yes (`custom_vjp`) |
| **Positional** | `apply_rope(q, k=None, cos=None, sin=None, start_pos=0)` | `apply_rope` | Yes (`custom_vjp`) |
| | `precompute_freqs(dim, max_seq_len=4096, base=10000.0)` | `precompute_freqs` | N/A |
| **Loss Functions** | `cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.0)` | `cross_entropy` | Yes (`custom_vjp`) |
| | `fused_linear_cross_entropy(x, weight, target, bias=None, ...)` | `fused_linear_cross_entropy` | Yes (`custom_vjp`) |
| **Attention** | `flash_attention(q, k, v, is_causal=False, sm_scale=None)` | `flash_attention` | Yes |
| | `flash_decode(q, k, v, sm_scale=None)` | `flash_decode` | No (inf. only) |

---

## 3. Key Differences from PyTorch

1. **Explicit Gate and Up Tensors**:
    In PyTorch, `swiglu(x)` can accept a single tensor of size `2*D` and automatically split it if `up` is None. In JAX, `swiglu(gate, up)` requires both `gate` and `up` to be explicitly passed as separate arrays to preserve compile-time shape guarantees.
2. **Stateless Modules**:
    JAX does not provide `nn.Module` classes (such as `RMSNorm` or `RoPE`). Instead, manage weights explicitly in parameter dictionaries (e.g. Flax Linen or Equinox) and pass them directly to functional calls.
3. **No Dynamic Autograd States**:
    PyTorch autograd functions store intermediate tensors on `ctx.save_for_backward`. In JAX, forward transforms return explicit residual tuples to backward functions according to standard `custom_vjp` conventions.

---

## 4. Minimal Runnable Example with JIT and Autodiff

```python
import jax
import jax.numpy as jnp
from stackkernels.jax import apply_rope, precompute_freqs, rms_norm, swiglu

# 1. Normalization with automatic differentiation
def loss_fn(x, weight):
    normed = rms_norm(x, weight, eps=1e-6)
    return jnp.sum(normed)

x = jnp.ones((2, 128, 512), dtype=jnp.float32)
weight = jnp.ones((512,), dtype=jnp.float32)

# Compute gradients with respect to x and weight
grad_x, grad_w = jax.grad(loss_fn, argnums=(0, 1))(x, weight)
print(f"grad_x shape: {grad_x.shape}, grad_w shape: {grad_w.shape}")

# 2. Rotary Position Embedding with JIT compilation
cos, sin = precompute_freqs(dim=64, max_seq_len=512)

@jax.jit
def rotate_and_gate(q, k, gate, up):
    q_rot, k_rot = apply_rope(q, k, cos=cos, sin=sin)
    act = swiglu(gate, up)
    return q_rot, k_rot, act

q = jnp.ones((2, 128, 8, 64), dtype=jnp.float32)
k = jnp.ones((2, 128, 2, 64), dtype=jnp.float32)
gate = jnp.ones((2, 128, 512), dtype=jnp.float32)
up = jnp.ones((2, 128, 512), dtype=jnp.float32)

q_rot, k_rot, act = rotate_and_gate(q, k, gate, up)
print(f"JIT rotated Q shape: {q_rot.shape}, Act shape: {act.shape}")
```

