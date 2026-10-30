# Attention: FlashAttention-2 and Split-KV FlashDecode

This guide details the FlashAttention and FlashDecode implementations in StackKernels, including the variant matrix across Multi-Head, Grouped-Query, and Multi-Query Attention, causal bottom-right alignment, and single-token decode acceleration.

---

## 1. Overview and Architecture Fit

Standard attention computation materializes an intermediate $S_q \times S_k$ attention score matrix in global GPU memory, causing quadratic $O(S^2)$ memory bandwidth bottlenecks. `flash_attention` implements the FlashAttention-2 algorithm in OpenAI Triton, tiling query, key, and value blocks directly within fast GPU SRAM and computing softmax incrementally via online scaling. It replaces standard scaled dot-product attention in Stackformer and transformer architectures.

For inference decoding with a single query token ($S_q = 1$) and long cached context sequences ($S_k \gg 1$), `flash_decode` partitions the KV context across multiple thread blocks using Split-KV reduction, maintaining high GPU SM occupancy.

---

## 2. Signatures

Copied directly from `src/stackkernels/torch/attention.py`:

```python
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    ...

def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    ...

class FlashAttention(nn.Module):
    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: float | None = None,
    ) -> None:
        ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

---

## 3. Supported Dtypes, Shapes, and Devices

- **Dtypes**: `torch.float16`, `torch.float32`, and `torch.bfloat16` (sm80+).
- **Head Dimensions ($D$)**: 64, 80, 96, 128, and 256.
- **Shapes**:
    - `q`: `(B, H_q, S_q, D)`
    - `k`: `(B, H_kv, S_k, D)` where $H_q \ge H_{kv}$
    - `v`: `(B, H_kv, S_k, D)`
- **Devices**: CUDA GPUs via Triton; CPU via pure PyTorch reference fallback.
- **Fallback Trigger**: Executing on CPU or setting `stackkernels.options(backend="reference")`.

---

## 4. Forward and Backward Status

- **Forward**: Fully implemented in Triton. Writes out the final attention representation and saves log-sum-exp (`lse`) statistics for backward autograd.
- **Backward**: Custom autograd function (`FlashAttentionFunction`). The backward pass reads saved query, key, value, output, and LSE tensors, recomputes attention probabilities on the fly within SRAM tiles, and writes out `dq`, `dk`, and `dv` gradients without materializing the full attention matrix.
- **FlashDecode Backward**: `flash_decode` is an inference-only operator designed for single-token autoregressive generation ($S_q = 1$). It does not implement a backward pass.

---

## 5. Minimal Runnable Example

```python
import torch
from stackkernels.torch import flash_attention, flash_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_q_heads, num_kv_heads, head_dim = 8, 2, 64  # GQA
seq_len = 512

# 1. Prefill / Training FlashAttention
q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, requires_grad=True)
k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, requires_grad=True)
v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, requires_grad=True)

out = flash_attention(q, k, v, is_causal=True)
out.sum().backward()
assert q.grad is not None and k.grad is not None and v.grad is not None
print(f"FlashAttention output: {out.shape}")

# 2. Decode FlashAttention (single token)
q_dec = torch.randn(batch_size, num_q_heads, 1, head_dim, device=device)
k_cache = torch.randn(batch_size, num_kv_heads, 1024, head_dim, device=device)
v_cache = torch.randn(batch_size, num_kv_heads, 1024, head_dim, device=device)

out_dec = flash_decode(q_dec, k_cache, v_cache)
print(f"FlashDecode output:    {out_dec.shape}")
```

---

## 6. Equivalent Plain PyTorch

Below is the standard unfused reference implementation of multi-head attention:

```python
import math
import torch

def ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    # Repeat KV heads for GQA/MQA if necessary
    group_size = q.shape[1] // k.shape[1]
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if is_causal:
        s_q, s_k = q.shape[-2], k.shape[-2]
        mask = torch.ones((s_q, s_k), device=q.device, dtype=torch.bool).tril(diagonal=s_k - s_q)
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

In standard PyTorch, `scores` requires $O(B \times H \times S_q \times S_k)$ memory. StackKernels tiles computation so that scores exist only temporarily in SRAM.

---

## 7. Attention Variant Matrix (Stackformer Mapping)

| Architecture Variant | Q Heads ($H_q$) | KV Heads ($H_{kv}$) | Sequence Lengths | Triton Kernel Dispatched |
|:---|:---:|:---:|:---:|:---|
| Multi-Head Attention (MHA) | $H$ | $H$ | $S_q > 1$, $S_k \ge S_q$ | `flash_attn_fwd` (`stackkernels._core.attention.flash_fwd`) |
| Grouped-Query Attention (GQA) | $H_q$ | $1 < H_{kv} < H_q$ | $S_q > 1$, $S_k \ge S_q$ | `flash_attn_fwd` with internal modulo head mapping |
| Multi-Query Attention (MQA) | $H_q$ | $1$ | $S_q > 1$, $S_k \ge S_q$ | `flash_attn_fwd` with broadcast KV stride |
| Cross-Attention | $H_q$ | $H_{kv}$ | $S_q \neq S_k$, non-causal | `flash_attn_fwd` (`is_causal=False`) |
| Single-Token Decode | $H_q$ | $H_{kv}$ | $S_q = 1$, $S_k > 1$ | `flash_decode` (`stackkernels._core.attention.decode`) |
| Causal Masking Alignment | $H_q$ | $H_{kv}$ | Any | Bottom-right aligned: $k \le q + (S_k - S_q)$ |

---

## 8. Validation Status on T4 and Limitations

- **Tesla T4 (sm75)**: Fully validated in fp16 and fp32. Verified across MHA, GQA, MQA, causal masking, and head dimensions 64 and 128 in `benchmarks/results/validation_t4.json`.
- **Known Limitations**:
    - BF16 is not supported natively on sm75; falls back to reference code.
    - Head dimensions must be one of the explicitly tuned sizes: 64, 80, 96, 128, or 256.
    - `flash_decode` does not support backward differentiation (inference only).

