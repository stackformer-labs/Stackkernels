"""Basic PyTorch usage example for StackKernels.

Demonstrates:
1. Functional APIs (rms_norm, swiglu, apply_rope, cross_entropy, flash_attention)
2. Object-oriented PyTorch modules (RMSNorm, RoPE, CrossEntropyLoss)
3. Transparent fallback between Triton GPU kernels and pure PyTorch reference
"""

import torch
import stackkernels
from stackkernels.torch import (
    RMSNorm,
    RoPE,
    CrossEntropyLoss,
    apply_rope,
    cross_entropy,
    flash_attention,
    rms_norm,
    swiglu,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"StackKernels version: {stackkernels.__version__}")
    print(f"Running on device:    {device} ({dtype})")
    print(f"Active backend:       {stackkernels.get_backend()}")

    # 1. Normalization
    batch, seq_len, dim = 2, 128, 512
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.ones(dim, device=device, dtype=dtype)

    norm_out = rms_norm(x, weight)
    print(f"[1] rms_norm output shape: {norm_out.shape}")

    # Backward pass
    norm_out.sum().backward()
    print(f"    grad on x shape:        {x.grad.shape}")

    # 2. Activation (SwiGLU)
    x.grad = None
    gate = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    up = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    act_out = swiglu(gate, up)
    print(f"[2] swiglu output shape:   {act_out.shape}")

    # 3. Rotary Position Embedding (RoPE)
    num_heads, head_dim = 8, 64
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    rope = RoPE(dim=head_dim, max_seq_len=seq_len + 64, device=device)
    q_rot, k_rot = rope(q, k)
    print(f"[3] RoPE output q_rot:     {q_rot.shape}, k_rot: {k_rot.shape}")

    # 4. Attention
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    attn_out = flash_attention(q_rot, k_rot, v, is_causal=True)
    print(f"[4] flash_attention shape: {attn_out.shape}")

    # 5. Loss
    vocab_size = 4096
    logits = torch.randn(batch * seq_len, vocab_size, device=device, dtype=dtype)
    targets = torch.randint(0, vocab_size, (batch * seq_len,), device=device)
    loss = cross_entropy(logits, targets, ignore_index=-100)
    print(f"[5] cross_entropy loss:    {loss.item():.4f}")


if __name__ == "__main__":
    main()

