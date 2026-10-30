# Activations: Gated Units (SwiGLU, GeGLU) and Elementwise Functions

This guide details the activation functions in StackKernels, including fused Gated Linear Units (`swiglu`, `geglu`, `reglu`, `glu`) and fused elementwise activations (`gelu`, `silu`, `relu`, `leaky_relu`, `sigmoid`).

---

## 1. Overview and Architecture Fit

Gated Linear Units (GLUs) project input tokens into two distinct representations—a gating pathway and an up-projection—and compute elementwise multiplication of the activated gate with the up-projection. `swiglu` replaces standard transformer MLP activations (as used in LLaMA, Mistral, and Stackformer). In standard PyTorch, computing SwiGLU requires multiple elementwise kernel launches and intermediate tensor allocations. StackKernels fuses gate activation and multiplication into a single Triton kernel pass.

---

## 2. Signatures

Copied directly from `src/stackkernels/torch/activation.py`:

```python
def swiglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    ...

def geglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    ...

def reglu(gate: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    ...

def glu(
    gate: torch.Tensor,
    up: torch.Tensor | None = None,
    act: Literal["silu", "gelu", "relu"] = "silu",
) -> torch.Tensor:
    ...

def gelu(x: torch.Tensor, approximate: Literal["none", "tanh"] = "none") -> torch.Tensor:
    ...

def silu(x: torch.Tensor) -> torch.Tensor:
    ...

def relu(x: torch.Tensor) -> torch.Tensor:
    ...

def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    ...

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    ...
```

If `up` is omitted, the function automatically chunks the input `gate` tensor into two halves along its final dimension (`torch.chunk(gate, 2, dim=-1)`).

---

## 3. Supported Dtypes, Shapes, and Devices

- **Dtypes**: `torch.float16`, `torch.float32`, and `torch.bfloat16` (sm80+).
- **Shapes**: Arbitrary tensor shapes `(*, D)`.
- **Devices**: CUDA GPUs via Triton kernels; CPU via PyTorch eager reference fallback.
- **Fallback Trigger**: Executing on CPU or setting `stackkernels.options(backend="reference")` routes execution to standard PyTorch operators (`torch.nn.functional`).

---

## 4. Forward and Backward Status

- **Forward**: Implemented in Triton. Gated kernels read gate and up tensors, evaluate the chosen non-linearity, and write out activated tensors in a single memory pass.
- **Backward**: Custom autograd function (`GLUFunction` and `ElementwiseFunction`). Gated backward calculates analytic gradients for both gate and up tensors simultaneously.

---

## 5. Minimal Runnable Example

```python
import torch
from stackkernels.torch import geglu, swiglu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, seq_len, dim = 2, 512, 2048

# Separate gate and up tensors
gate = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
up = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

out = swiglu(gate, up)
out.sum().backward()
assert gate.grad is not None and up.grad is not None

# Combined tensor split automatically
combined = torch.randn(batch_size, seq_len, 2 * dim, device=device)
out_combined = geglu(combined)
assert out_combined.shape == (batch_size, seq_len, dim)
```

---

## 6. Equivalent Plain PyTorch

Below is the standard PyTorch implementation of `swiglu`:

```python
import torch
import torch.nn.functional as F

def ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate.float()) * up.float()
```

In standard PyTorch, `F.silu` allocates an intermediate tensor before the multiplication kernel runs. StackKernels computes both directly in GPU registers.

---

## 7. Tuning Notes

- **Kernel Class**: `elementwise`.
- **Tuning Knobs**:
    - `BLOCK_SIZE`: 256, 512, 1024, or 2048 elements per thread block.
    - `num_warps`: 4 or 8 warps.
    - `num_stages`: 1 or 2 stages.
- **Heuristic Choice**: Maximizes streaming multiprocessor (SM) occupancy by calculating grid sizes that cover available SMs evenly.

---

## 8. Validation Status on T4 and Limitations

- **Tesla T4 (sm75)**: Fully validated in fp16 and fp32. Confirmed zero regression on forward and backward gradients across all GLU variants and elementwise activations.
- **Known Limitations**:
    - BF16 requires compute capability 8.0 or newer.

