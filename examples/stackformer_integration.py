"""Integration example: Drop-in replacement for Stackformer transformer blocks.

Demonstrates replacing standard PyTorch transformer layers in Stackformer
with StackKernels fused operators (RMSNorm, SwiGLU MLP, and FlashAttention).
"""

import torch
import torch.nn as nn
from stackkernels.torch import RMSNorm, flash_attention, swiglu


class StackformerAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention block using StackKernels."""

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
        # Project to (B, H, S, D) for flash_attention
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        out = flash_attention(q, k, v, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(out)


class StackformerMLP(nn.Module):
    """Feed-Forward block using StackKernels SwiGLU."""

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
        # Pre-norm architecture
        h = x + self.attn(self.input_norm(x))
        out = h + self.mlp(self.post_attention_norm(h))
        return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 256
    num_heads = 4
    hidden_dim = 512
    batch_size = 2
    seq_len = 64

    block = StackformerBlock(dim, num_heads, hidden_dim).to(device)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    out = block(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape
    print("Stackformer block ran successfully with StackKernels!")


if __name__ == "__main__":
    main()

