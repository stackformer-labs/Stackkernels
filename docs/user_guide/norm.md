# Normalization: RMSNorm, LayerNorm, and Fused Add + Norm

This guide explains how to use StackKernels normalization operations—including Root Mean Square Normalization (`rms_norm`), standard Layer Normalization (`layer_norm`), and fused residual addition with normalization (`fused_add_norm`).

---

## 1. Overview and Architecture Fit

Normalization layers stabilize training dynamics across transformer architectures. `rms_norm` computes root mean square normalization over the final hidden dimension, replacing standard transformer pre- and post-attention normalization layers (e.g. `LlamaRMSNorm` in modern LLMs). `layer_norm` computes mean-centering and variance-scaling, replacing `torch.nn.LayerNorm`. `fused_add_norm` combines the residual connection addition ($x + \text{residual}$) and normalization into a single memory pass, eliminating an intermediate round-trip to global memory between transformer sub-layers.

---

## 2. Signatures

Copied directly from `src/stackkernels/torch/norm.py`:

```python
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    ...

def fused_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    norm_type: Literal["rms_norm", "layer_norm"] = "rms_norm",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Object-oriented `nn.Module` classes:

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        ...

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        ...
```

---

## 3. Supported Dtypes, Shapes, and Devices

- **Dtypes**: `torch.float16`, `torch.float32`, and `torch.bfloat16` (BF16 requires compute capability sm80+).
- **Shapes**: Arbitrary batch and sequence dimensions `(*, D)`. Hidden dimension $D$ supports single-pass reductions up to 8192.
- **Devices**: CUDA GPUs via Triton kernels; CPU via pure PyTorch reference fallback.
- **Fallback Trigger**: Executing on CPU or setting `stackkernels.options(backend="reference")` triggers the pure PyTorch reference path.

---

## 4. Forward and Backward Status

- **Forward**: Fully implemented in pure Triton. Computes normalized output and caches reciprocal standard deviation (`rstd`) in SRAM.
- **Backward**: Fully implemented with custom `torch.autograd.Function`. The backward pass computes gradients for input $x$ and optional weight/bias parameters in a single kernel pass. `fused_add_norm` returns gradients for both input $x$ and the residual tensor.

---

## 5. Minimal Runnable Example

```python
import torch
from stackkernels.torch import RMSNorm, fused_add_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, seq_len, dim = 2, 512, 1024

# 1. Module usage
norm_layer = RMSNorm(dim=dim, eps=1e-6).to(device)
x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
y = norm_layer(x)
y.sum().backward()
assert x.grad is not None

# 2. Fused residual add + norm
residual = torch.randn(batch_size, seq_len, dim, device=device)
weight = torch.ones(dim, device=device)
normed, res_out = fused_add_norm(x, residual, weight=weight, norm_type="rms_norm")
print(f"Normed shape: {normed.shape}, Residual shape: {res_out.shape}")
```

---

## 6. Equivalent Plain PyTorch

Below is the equivalent standard PyTorch implementation of `rms_norm`:

```python
import torch

def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normed = (x.float() * rstd).to(x.dtype)
    return normed * weight if weight is not None else normed
```

StackKernels eliminates intermediate tensor allocations and keeps reduction accumulators inside GPU register files.

---

## 7. Tuning Notes

- **Kernel Class**: `row` (row-major memory reduction).
- **Tuning Knobs**:
    - `BLOCK_SIZE`: Power-of-two tile along the hidden dimension (typically 512, 1024, 2048, or 4096).
    - `num_warps`: 4 or 8 warps per thread block.
    - `num_stages`: Default 1 for reduction kernels.
- **Heuristic Choice**: The class heuristic evaluates device SM count and active warp registers to balance thread occupancy without exceeding shared memory budgets.

---

## 8. Validation Status on T4 and Limitations

- **Tesla T4 (sm75)**: Fully validated in fp16 and fp32. Verified numerical consistency against PyTorch references across sequence lengths from 64 to 8192.
- **Known Limitations**:
    - BF16 cannot run natively on Tesla T4 due to hardware lack of bfloat16 tensor instructions; attempting BF16 on sm75 routes to reference fallback.
    - Hidden dimensions $D > 8192$ require multi-pass row tiling.

