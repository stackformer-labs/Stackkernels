"""Tests for Flash Attention forward and backward (MHA, GQA, MQA, causal, multi-head-dim)."""

import pytest
import torch

from stackkernels._reference.torch.attention import attention_reference
from stackkernels.torch.attention import FlashAttention, flash_attention


@pytest.mark.parametrize("heads", [
    (8, 8),   # MHA
    (8, 2),   # GQA
    (8, 1),   # MQA
])
@pytest.mark.parametrize("seq_lens", [
    (32, 32),  # Equal seqlen
    (16, 64),  # S_q < S_k (prefix / cross-attention)
    (64, 32),  # S_q > S_k (chunked)
])
@pytest.mark.parametrize("head_dim", [64, 80, 96, 128, 256])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_forward(heads, seq_lens, head_dim, is_causal):
    torch.manual_seed(42)
    B = 2
    H_q, H_kv = heads
    S_q, S_k = seq_lens
    D = head_dim

    q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
    k = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)
    v = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)

    out = flash_attention(q, k, v, is_causal=is_causal)
    expected, _ = attention_reference(q, k, v, is_causal=is_causal)

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("heads", [
    (4, 4),  # MHA
    (4, 2),  # GQA
    (4, 1),  # MQA
])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attn_backward(heads, is_causal):
    torch.manual_seed(42)
    B = 2
    H_q, H_kv = heads
    S_q, S_k = 16, 16
    D = 64

    q1 = torch.randn(B, H_q, S_q, D, dtype=torch.float32, requires_grad=True)
    k1 = torch.randn(B, H_kv, S_k, D, dtype=torch.float32, requires_grad=True)
    v1 = torch.randn(B, H_kv, S_k, D, dtype=torch.float32, requires_grad=True)

    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)

    out1 = flash_attention(q1, k1, v1, is_causal=is_causal)
    out2, _ = attention_reference(q2, k2, v2, is_causal=is_causal)

    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v1.grad, v2.grad, rtol=1e-3, atol=1e-3)


def test_flash_attn_module():
    torch.manual_seed(42)
    B, H_q, H_kv, S, D = 1, 4, 2, 16, 64
    q = torch.randn(B, H_q, S, D)
    k = torch.randn(B, H_kv, S, D)
    v = torch.randn(B, H_kv, S, D)

    attn = FlashAttention(is_causal=True)
    out = attn(q, k, v)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_flash_attn_jax(backend_name):
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels.jax.attention import flash_attention as jax_attn

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (2, 4, 16, 64))
    k = jax.random.normal(key, (2, 2, 16, 64))
    v = jax.random.normal(key, (2, 2, 16, 64))

    out = jax_attn(q, k, v, is_causal=True)
    assert not jnp.isnan(out).any()
