"""End-to-end tests for Rotary Position Embedding (RoPE) forward, backward, GQA, and start_pos."""

import pytest
import torch

from stackkernels._reference.torch.rope import (
    apply_rope as ref_apply_rope_torch,
    precompute_freqs,
)
from stackkernels.torch.embedding import RoPE, apply_rope as torch_apply_rope


@pytest.mark.parametrize("start_pos", [0, 15, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rope_forward_torch(start_pos, dtype, dtype_tolerances):
    """Test RoPE forward accuracy against reference implementation with start_pos offset."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, S, H, D = 2, 8, 4, 64
    x = torch.randn(B, S, H, D, dtype=dtype, device=device)
    cos, sin = precompute_freqs(D, max_seq_len=start_pos + S + 64, device=device, dtype=dtype)

    out = torch_apply_rope(x, cos=cos, sin=sin, start_pos=start_pos)
    ref_out = ref_apply_rope_torch(x, cos=cos, sin=sin, start_pos=start_pos)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.parametrize("start_pos", [0, 23])
def test_rope_gqa(start_pos, dtype_tolerances):
    """Test RoPE on Grouped Query Attention (Q and K with different head counts)."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, S, D = 2, 8, 128
    H_q = 16  # 16 Query heads
    H_k = 4   # 4 Key heads (4:1 GQA ratio)

    q = torch.randn(B, S, H_q, D, dtype=torch.float32, device=device)
    k = torch.randn(B, S, H_k, D, dtype=torch.float32, device=device)

    cos, sin = precompute_freqs(D, max_seq_len=start_pos + S + 64, device=device)

    q_rot, k_rot = torch_apply_rope(q, k, cos=cos, sin=sin, start_pos=start_pos)
    q_ref, k_ref = ref_apply_rope_torch(q, k, cos=cos, sin=sin, start_pos=start_pos)

    assert q_rot.shape == (B, S, H_q, D)
    assert k_rot.shape == (B, S, H_k, D)
    torch.testing.assert_close(q_rot, q_ref, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(k_rot, k_ref, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.parametrize("start_pos", [0, 31])
def test_rope_backward_torch(start_pos, dtype_tolerances):
    """Test RoPE backward gradients against autograd reference."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, S, H, D = 2, 4, 8, 64
    x_test = torch.randn(B, S, H, D, dtype=torch.float32, device=device, requires_grad=True)
    cos, sin = precompute_freqs(D, max_seq_len=start_pos + S + 64, device=device)

    y_test = torch_apply_rope(x_test, cos=cos, sin=sin, start_pos=start_pos)
    dy = torch.randn_like(y_test)
    y_test.backward(dy)

    x_ref = x_test.detach().clone().requires_grad_(True)
    y_ref = ref_apply_rope_torch(x_ref, cos=cos, sin=sin, start_pos=start_pos)
    y_ref.backward(dy)

    torch.testing.assert_close(x_test.grad, x_ref.grad, atol=tol["atol"], rtol=tol["rtol"])


def test_rope_module():
    """Verify RoPE nn.Module layer functionality with cached buffers."""
    dim = 64
    module = RoPE(dim=dim, max_seq_len=512)
    q = torch.randn(2, 8, 4, dim, requires_grad=True)
    k = torch.randn(2, 8, 2, dim, requires_grad=True)

    q_rot, k_rot = module(q, k, start_pos=10)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    loss = q_rot.sum() + k_rot.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None


def test_rope_forward_jax(backend_name):
    """Test JAX RoPE against reference (skipped if JAX not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.rope import apply_rope as ref_apply_rope_jax
    from stackkernels.jax.embedding import apply_rope as jax_apply_rope

    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (2, 8, 4, 64))
    k = jax.random.normal(key, (2, 8, 2, 64))

    q_rot, k_rot = jax_apply_rope(q, k, start_pos=16)
    q_ref, k_ref = ref_apply_rope_jax(q, k, start_pos=16)

    assert jnp.allclose(q_rot, q_ref, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(k_rot, k_ref, atol=1e-5, rtol=1e-5)

