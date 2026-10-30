# Stackformer Integration Guide

This guide demonstrates how to replace standard PyTorch layers in the Stackformer architecture with StackKernels fused operators, based on the working example in `examples/stackformer_integration.py`.

---

## 1. Integration Overview

Stackformer is a modern, modular transformer implementation designed for research and production scale. By swapping out standard PyTorch attention, normalization, and MLP layers for StackKernels, you reduce intermediate GPU memory allocations and eliminate kernel launch overhead.

The primary replacement targets are:
- Pre- and post-attention normalizations: Replace with `stackkernels.torch.RMSNorm`.
- Gated Feed-Forward Networks (MLP): Replace activation chains with `stackkernels.torch.swiglu`.
- Self- and Cross-Attention: Replace quadratic attention matrices with `stackkernels.torch.flash_attention`.
- Positional encodings: Replace manual tensor rotations with `stackkernels.torch.RoPE`.
- Language Model Head loss: Replace projection and loss with `stackkernels.torch.FusedLinearCrossEntropyLoss`.

---

## 2. Transformer Layer Replacement Code

Below is the complete implementation of a transformer block utilizing StackKernels, matching `examples/stackformer_integration.py`:

```python
import torch
import torch.nn as nn
from stackkernels.torch import RMSNorm, flash_attention, swiglu


class StackformerAttention(nn.Module):
    """Multi-Head and Grouped-Query Attention using StackKernels FlashAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads
        self.is_causal = is_causal

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        # Project inputs to (B, H, S, D) format expected by flash_attention
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        out = flash_attention(q, k, v, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(out)


class StackformerMLP(nn.Module):
    """Feed-Forward block using fused StackKernels SwiGLU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = swiglu(gate, up)
        return self.down_proj(act)


class StackformerBlock(nn.Module):
    """Complete Transformer layer using StackKernels."""

    def __init__(self, dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.attn = StackformerAttention(dim, num_heads)
        self.post_attention_norm = RMSNorm(dim)
        self.mlp = StackformerMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        h = x + self.attn(self.input_norm(x))
        out = h + self.mlp(self.post_attention_norm(h))
        return out
```

---

## 3. Running the Integration Example

You can execute the standalone verification script from the root of the repository:

```bash
python examples/stackformer_integration.py
```

Expected output:

```text
Input shape:  torch.Size([2, 64, 256])
Output shape: torch.Size([2, 64, 256])
Stackformer block ran successfully with StackKernels!
```

---

## 4. Training Loss Integration with Fused Linear CE

To integrate the fused loss into your training step, replace standard LM head projection and `nn.CrossEntropyLoss`:

```python
from stackkernels.torch import FusedLinearCrossEntropyLoss

# Initialize loss module
loss_fn = FusedLinearCrossEntropyLoss(ignore_index=-100)

# In your training loop:
# final_hidden_states: (B, S, D)
# lm_head_weight: (V, D)
# target_token_ids: (B, S)
loss = loss_fn(final_hidden_states, model.lm_head.weight, target_token_ids)
loss.backward()
```

This bypasses materializing the large $(B \times S, V)$ logits matrix in global memory, enabling larger batch sizes or longer sequence lengths without out-of-memory errors.

