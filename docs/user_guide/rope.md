# Rotary Position Embedding (RoPE)

This guide documents the Rotary Position Embedding (RoPE) implementation in StackKernels, including the functional `apply_rope` operator, the `RoPE` module, frequency precomputation, and support for Grouped-Query Attention (GQA) and autoregressive decoding offsets.

---

## 1. Overview and Architecture Fit

Rotary Position Embedding encodes token position information into query and key representations by multiplying pairs of features by rotational frequency matrices. `apply_rope` replaces standard RoPE layers (such as those in LLaMA, Mistral, and Stackformer). It applies rotation directly to query and key tensors in high-throughput Triton kernels, supporting unequal head counts (GQA/MQA) and arbitrary sequence offsets (`start_pos`) for autoregressive generation.

---

## 2. Signatures

Copied directly from `src/stackkernels/torch/embedding.py`:

```python
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    start_pos: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    ...

def precompute_freqs(
    dim: int,
    max_seq_len: int = 4096,
    base: float = 10000.0,
    device: Any = None,
    dtype: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...

class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: Any = None,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...
```

---

## 3. Supported Dtypes, Shapes, and Devices

- **Dtypes**: `torch.float16`, `torch.float32`, and `torch.bfloat16` (sm80+).
- **Shapes**:
    - `q`: `(B, S_q, H_q, D)`
    - `k`: `(B, S_k, H_kv, D)` where $H_q \ge H_{kv}$
    - Head dimension $D$ must be even (typically 64, 80, 96, 128).
- **Devices**: CUDA GPUs via Triton; CPU via pure PyTorch reference fallback.
- **Fallback Trigger**: Executing on CPU or setting `stackkernels.options(backend="reference")`.

---

## 4. Forward and Backward Status

- **Forward**: Fully implemented in Triton. Slices the frequency tables `cos` and `sin` from `start_pos` to `start_pos + seq_len`, performs interleaved or split rotation, and outputs rotated tensors.
- **Backward**: Custom autograd function (`RoPEFunction`). Backward rotates incoming adjoint gradients by $-\theta$, exactly inverting the forward transformation with full numerical stability.

---

## 5. Minimal Runnable Example

```python
import torch
from stackkernels.torch import RoPE, apply_rope

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, seq_len = 2, 256
num_q_heads, num_kv_heads, head_dim = 8, 2, 64  # GQA configuration

# Using the RoPE module
rope = RoPE(dim=head_dim, max_seq_len=2048, device=device)

q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device, requires_grad=True)
k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, requires_grad=True)

# Forward pass with sequence offset for decoding
q_rot, k_rot = rope(q, k, start_pos=16)

loss = q_rot.sum() + k_rot.sum()
loss.backward()

assert q.grad is not None and k.grad is not None
print(f"Rotated Q: {q_rot.shape}, Rotated K: {k_rot.shape}")
```

---

## 6. Equivalent Plain PyTorch

Below is the standard reference implementation of RoPE rotation for a single tensor:

```python
import torch

def ref_apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    seq_len = x.shape[1]
    cos_slice = cos[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(2)
    sin_slice = sin[start_pos : start_pos + seq_len, :].unsqueeze(0).unsqueeze(2)

    half = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos_slice) + (x_rot * sin_slice)
```

StackKernels avoids creating concatenated tensor views and evaluates trigonometric mixing in-place inside kernel registers.

---

## 7. Tuning Notes

- **Kernel Class**: `elementwise`.
- **Tuning Knobs**:
    - `BLOCK_SIZE`: Number of tokens processed per block along batch and sequence dimensions.
    - `num_warps`: 4 or 8 warps.
- **Heuristic Choice**: Tiles across the flattened batch and sequence dimensions to ensure parallel saturation across all GPU multiprocessors.

---

## 8. Validation Status on T4 and Limitations

- **Tesla T4 (sm75)**: Fully validated in fp16 and fp32. Verified with GQA ($H_q \neq H_{kv}$), MQA ($H_{kv} = 1$), and non-zero `start_pos`.
- **Known Limitations**:
    - BF16 execution on sm75 falls back to reference code.
    - The head dimension $D$ must be divisible by 2.

