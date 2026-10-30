# Dropout, KV-Cache Update, and Sampling

This page documents the low-memory Philox dropout operators in StackKernels, along with the development status and roadmap for planned kernel operations (in-place KV-cache updating and Top-K / Top-P sampling).

---

## 1. Fused Low-Memory Dropout

Standard dropout materializes a boolean or byte mask tensor of shape `x.shape` in global GPU memory (HBM) during the forward pass and reads it back during the backward pass. For long-context models, this induces significant memory footprint and bandwidth overhead.

StackKernels implements **stateless Philox PRNG dropout**:
- **Zero-allocation backward pass**: Only an 8-byte integer `seed` is saved in the autograd context instead of the $N$-element mask.
- In the backward pass, `tl.rand(seed, offsets)` regenerates the identical pseudo-random sequence directly in SRAM registers, entirely bypassing global memory reads.
- **`fused_dropout_add`** fuses dropout masking, scaling, and residual addition (`y = dropout(x, p) + residual`) into a single kernel launch.

### PyTorch Usage

```python
import torch
import stackkernels.torch as sk

x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16, requires_grad=True)
residual = torch.randn_like(x)

# 1. Functional dropout
y = sk.dropout(x, p=0.1, training=True)
y.backward(torch.ones_like(y))

# 2. Fused dropout + residual addition
out = sk.fused_dropout_add(x, residual, p=0.1, training=True)

# 3. Module interface
layer = sk.FusedDropoutAdd(p=0.1)
out = layer(x, residual)
```

### JAX Usage

```python
import jax
import jax.numpy as jnp
import stackkernels.jax as sk_jax

x = jnp.ones((2, 512, 4096), dtype=jnp.float16)
residual = jnp.ones_like(x)
key = jax.random.PRNGKey(42)

# Functional dropout with custom_vjp
out = sk_jax.dropout(x, p=0.1, training=True, key=key)

# Fused dropout + residual add
out_fused = sk_jax.fused_dropout_add(x, residual, p=0.1, training=True, key=key)
```

---

## 2. Operations in Development (Roadmap)

The following operations are defined as roadmap milestones in the StackKernels repository and currently exist as stub modules in `src/stackkernels/torch/`:

| Module Path | Planned Operation | Current State | Target Release |
|:---|:---|:---:|:---:|
| `src/stackkernels/torch/cache.py` | Paged and Contiguous In-Place KV-Cache Update | Stub / Not yet implemented | Future release |
| `src/stackkernels/torch/sampling.py` | Fused Top-K and Top-P (Nucleus) Generation Sampling | Stub / Not yet implemented | Future release |

### KV-Cache In-Place Update (`cache.py`)

- **Objective**: Insert newly generated key and value projections into contiguous or paged KV-cache buffers in-place during autoregressive decoding, eliminating memory reallocations and scatter operations.
- **Planned Interface**:
  ```python
  def kv_cache_update(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_indices: torch.Tensor) -> None:
      ...
  ```

### Fused Sampling (`sampling.py`)

- **Objective**: Combine temperature scaling, top-k filtering, top-p cumulative thresholding, and categorical sampling into a single GPU pass to accelerate autoregressive token generation.
- **Planned Interface**:
  ```python
  def sample_top_p_top_k(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
      ...
  ```

---

## Recommended Alternatives Today

For features currently in development:

- **For KV Caching**: Slice existing pre-allocated PyTorch tensors in-place:
  ```python
  k_cache[:, :, current_pos : current_pos + 1, :] = k_new
  v_cache[:, :, current_pos : current_pos + 1, :] = v_new
  ```
- **For Sampling**: Use `torch.topk` or standard Hugging Face / Stackformer generation utilities.
