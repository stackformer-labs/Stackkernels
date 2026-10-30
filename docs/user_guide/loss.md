# Loss Functions: Cross-Entropy and Fused Linear Cross-Entropy

This guide explains how to use StackKernels loss functions—including standalone online softmax `cross_entropy` and `fused_linear_cross_entropy`, which eliminates the materialization of large vocabulary logit matrices.

---

## 1. Overview and Architecture Fit

In autoregressive language modeling, calculating cross-entropy loss over large vocabularies ($V = 32{,}000$ to $128{,}000+$) is typically the most memory-intensive step of the backward pass. Standard PyTorch computes full logits through matrix multiplication ($Z = X W^T$), writes the entire $B \times S \times V$ tensor to global GPU memory, and reads it back during softmax and loss evaluation. 

`cross_entropy` implements an online softmax algorithm that accumulates log-sum-exp in a single streaming pass. `fused_linear_cross_entropy` fuses the linear projection GEMM and the online cross-entropy reduction directly into a tiled kernel, computing loss and gradients without ever allocating the full logit matrix in global memory. This replaces the LM head and loss calculation in Stackformer and other transformer training pipelines.

---

## 2. Signatures

Copied directly from `src/stackkernels/torch/loss.py`:

```python
def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    ...

def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    ...

class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        ...

class FusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        ...
```

---

## 3. Supported Dtypes, Shapes, and Devices

- **Dtypes**: `torch.float16`, `torch.float32`, and `torch.bfloat16` (sm80+).
- **Shapes**:
    - For `cross_entropy`: `logits` of shape `(*, V)`, `target` of shape `(*)`.
    - For `fused_linear_cross_entropy`: `x` of shape `(*, K)`, `weight` of shape `(V, K)`, optional `bias` of shape `(V,)`, `target` of shape `(*)`.
- **Devices**: CUDA GPUs via Triton; CPU via eager PyTorch reference fallback.
- **Fallback Trigger**: Executing on CPU or setting `stackkernels.options(backend="reference")`.

---

## 4. Forward and Backward Status

- **Forward**: Fully implemented in Triton. Uses the online softmax algorithm to compute the log-sum-exp (LSE) vector and target log-probabilities on the fly.
- **Backward**: Custom autograd function (`CrossEntropyFunction` and `FusedLinearCrossEntropyFunction`). In `fused_linear_cross_entropy`, the backward pass computes gradients for hidden states $x$, embedding weights $W$, and optional bias $b$ without allocating intermediate full logit tensors.

---

## 5. Minimal Runnable Example

```python
import torch
from stackkernels.torch import FusedLinearCrossEntropyLoss, cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, seq_len = 2, 256
hidden_dim = 1024
vocab_size = 8192

# 1. Standalone Cross-Entropy
logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
loss = cross_entropy(logits, target, ignore_index=-100, label_smoothing=0.1)
loss.backward()
assert logits.grad is not None

# 2. Fused Linear + Cross-Entropy (No full logits allocated)
loss_fn = FusedLinearCrossEntropyLoss(ignore_index=-100)
x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
lm_head = torch.randn(vocab_size, hidden_dim, device=device, requires_grad=True)

fused_loss = loss_fn(x, lm_head, target)
fused_loss.backward()
assert x.grad is not None and lm_head.grad is not None
print(f"Fused loss value: {fused_loss.item():.4f}")
```

---

## 6. Equivalent Plain PyTorch

Below is the standard unfused PyTorch workflow for linear projection followed by cross-entropy:

```python
import torch
import torch.nn.functional as F

def ref_linear_cross_entropy(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Allocates (B * S, V) tensor in global memory:
    logits = torch.matmul(x.view(-1, x.shape[-1]), weight.t())
    # Reads back (B * S, V) tensor from global memory:
    loss = F.cross_entropy(logits, target.view(-1), ignore_index=-100)
    return loss
```

For a sequence of 4096 tokens and a vocabulary of 32,000 in FP16, `logits` alone occupies 262 MB per batch element. `fused_linear_cross_entropy` tiles across vocabulary chunks, discarding intermediate projections once log-sum-exp and target statistics are accumulated.

---

## 7. Tuning Notes

- **Kernel Class**: `row` (for standalone CE) and `tiled` (for fused linear CE).
- **Tuning Knobs**:
    - `BLOCK_SIZE`: 512, 1024, or 2048 elements per vocabulary chunk.
    - `num_warps`: 4 or 8 warps per block.
    - `num_stages`: 1 or 2 pipeline stages.
- **Heuristic Choice**: Determines vocabulary tile sizing based on SM shared memory capacity and register pressure to avoid spilling accumulators.

---

## 8. Validation Status on T4 and Limitations

- **Tesla T4 (sm75)**: Fully validated in fp16 and fp32. Verified with `ignore_index = -100`, positive label smoothing, and reduction modes (`mean`, `sum`, `none`).
- **Known Limitations**:
    - BF16 on sm75 falls back to reference code.
    - Fused linear cross-entropy requires the hidden dimension $K$ and vocabulary size $V$ to be multiples of 16 for optimal memory alignment.

