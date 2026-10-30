"""Tests for Split-KV FlashDecode (q_len = 1, MHA/GQA/MQA)."""

import pytest
import torch

from stackkernels._reference.torch.attention import attention_reference
from stackkernels.torch.attention import flash_decode


@pytest.mark.parametrize("heads", [
    (8, 8),  # MHA
    (8, 2),  # GQA
    (8, 1),  # MQA
])
@pytest.mark.parametrize("kv_len", [32, 128, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flash_decode_forward(heads, kv_len, head_dim):
    torch.manual_seed(42)
    B = 2
    H_q, H_kv = heads
    S_q = 1
    S_k = kv_len
    D = head_dim

    q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
    k = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)
    v = torch.randn(B, H_kv, S_k, D, dtype=torch.float32)

    out = flash_decode(q, k, v)
    expected, _ = attention_reference(q, k, v, is_causal=False)

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_flash_decode_jax(backend_name):
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels.jax.attention import flash_decode as jax_decode

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (2, 4, 1, 64))
    k = jax.random.normal(key, (2, 2, 128, 64))
    v = jax.random.normal(key, (2, 2, 128, 64))

    out = jax_decode(q, k, v)
    assert not jnp.isnan(out).any()
