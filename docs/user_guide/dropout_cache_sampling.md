# Planned Operations: Dropout, KV-Cache Update, and Sampling

This page documents the current development status of planned kernel operations—including fused dropout, in-place KV-cache updating, and Top-K / Top-P sampling.

---

## Current Status: Not Yet Implemented

The following operations are defined as roadmap milestones in the StackKernels repository and currently exist as stub modules in `src/stackkernels/torch/`:

| Module Path | Planned Operation | Current State | Target Release |
|:---|:---|:---:|:---:|
| `src/stackkernels/torch/dropout.py` | Fused Dropout and Fused Dropout + Add | Stub / Not yet implemented | Future release |
| `src/stackkernels/torch/cache.py` | Paged and Contiguous In-Place KV-Cache Update | Stub / Not yet implemented | Future release |
| `src/stackkernels/torch/sampling.py` | Fused Top-K and Top-P (Nucleus) Generation Sampling | Stub / Not yet implemented | Future release |

These operations are not included in the public `__all__` exports of `stackkernels.torch` or `stackkernels.jax` and are not registered in the optimization layer.

---

## Planned Architecture and Use Cases

### 1. Fused Dropout (`dropout.py`)

- **Objective**: Fuse random binomial masking and residual addition into preceding normalization or linear layers using Philox pseudo-random number generator (PRNG) states directly inside SRAM.
- **Planned Interface**:
  ```python
  # Planned signature (future release)
  def fused_dropout_add(x: torch.Tensor, residual: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
      ...
  ```

### 2. KV-Cache In-Place Update (`cache.py`)

- **Objective**: Insert newly generated key and value projections into contiguous or paged KV-cache buffers in-place during autoregressive decoding, eliminating memory reallocations and scatter operations.
- **Planned Interface**:
  ```python
  # Planned signature (future release)
  def kv_cache_update(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_indices: torch.Tensor) -> None:
      ...
  ```

### 3. Fused Sampling (`sampling.py`)

- **Objective**: Combine temperature scaling, top-k filtering, top-p cumulative thresholding, and categorical sampling into a single GPU pass to accelerate autoregressive token generation.
- **Planned Interface**:
  ```python
  # Planned signature (future release)
  def sample_top_p_top_k(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
      ...
  ```

---

## Recommended Alternatives Today

Until these kernels are introduced, use the standard framework equivalents:

- **For Dropout**: Use `torch.nn.functional.dropout(x, p=p, training=training)`.
- **For KV Caching**: Slice existing pre-allocated PyTorch tensors in-place:
  ```python
  k_cache[:, :, current_pos : current_pos + 1, :] = k_new
  v_cache[:, :, current_pos : current_pos + 1, :] = v_new
  ```
- **For Sampling**: Use `torch.topk` or standard Hugging Face / Stackformer generation utilities.

