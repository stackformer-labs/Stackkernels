"""Unit tests for dropout and fused dropout + residual add."""

import pytest
import torch

from stackkernels._reference.torch.dropout import (
    dropout as ref_dropout,
    fused_dropout_add as ref_fused_dropout_add,
)
from stackkernels.torch.dropout import (
    Dropout,
    FusedDropoutAdd,
    dropout,
    fused_dropout_add,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_eval_mode(dtype):
    """When training=False, dropout acts as identity."""
    x = torch.randn(4, 128, 256, dtype=dtype)
    out = dropout(x, p=0.5, training=False)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_zero_prob(dtype):
    """When p=0.0, dropout acts as identity even in training mode."""
    x = torch.randn(4, 128, 256, dtype=dtype)
    out = dropout(x, p=0.0, training=True)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dropout_full_prob(dtype):
    """When p=1.0, all activations are zeroed out."""
    x = torch.randn(4, 128, 256, dtype=dtype)
    out = dropout(x, p=1.0, training=True)
    expected = torch.zeros_like(x)
    torch.testing.assert_close(out, expected)


def test_dropout_stats():
    """Verify that empirical drop rate and expectation match theoretical properties."""
    torch.manual_seed(42)
    p = 0.3
    x = torch.ones(100, 1000, dtype=torch.float32)
    out = dropout(x, p=p, training=True, seed=123)

    # Check that dropped elements are exactly 0.0
    zero_fraction = (out == 0.0).float().mean().item()
    assert abs(zero_fraction - p) < 0.03, f"Expected drop rate ~{p}, got {zero_fraction}"

    # Check that scale factor 1/(1-p) preserves expected value
    assert abs(out.mean().item() - 1.0) < 0.03, f"Expected mean ~1.0, got {out.mean().item()}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_dropout_add_eval(dtype):
    """When training=False, fused_dropout_add is simply x + residual."""
    x = torch.randn(2, 64, 128, dtype=dtype)
    res = torch.randn(2, 64, 128, dtype=dtype)
    out = fused_dropout_add(x, res, p=0.5, training=False)
    torch.testing.assert_close(out, x + res)


def test_dropout_backward():
    """Test backward pass gradient propagation for dropout."""
    x = torch.randn(10, 100, dtype=torch.float32, requires_grad=True)
    p = 0.5
    out = dropout(x, p=p, training=True, seed=42)
    out.sum().backward()

    assert x.grad is not None
    # For kept elements: grad = 1 / (1 - p) = 2.0; for dropped elements: grad = 0.0
    kept_mask = out != 0.0
    dropped_mask = out == 0.0

    scale = 1.0 / (1.0 - p)
    torch.testing.assert_close(x.grad[kept_mask], torch.full_like(x.grad[kept_mask], scale))
    torch.testing.assert_close(x.grad[dropped_mask], torch.zeros_like(x.grad[dropped_mask]))


def test_fused_dropout_add_backward():
    """Test backward pass gradient propagation for fused_dropout_add."""
    x = torch.randn(10, 100, dtype=torch.float32, requires_grad=True)
    res = torch.randn(10, 100, dtype=torch.float32, requires_grad=True)
    p = 0.4
    out = fused_dropout_add(x, res, p=p, training=True, seed=99)
    out.sum().backward()

    # Gradient on residual must be exactly 1.0 everywhere
    assert res.grad is not None
    torch.testing.assert_close(res.grad, torch.ones_like(res))

    # Gradient on x must be scaled on kept elements and 0 on dropped elements
    assert x.grad is not None
    kept_mask = (out - res) != 0.0
    dropped_mask = (out - res) == 0.0
    scale = 1.0 / (1.0 - p)

    torch.testing.assert_close(x.grad[kept_mask], torch.full_like(x.grad[kept_mask], scale), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad[dropped_mask], torch.zeros_like(x.grad[dropped_mask]))


def test_dropout_modules():
    """Test nn.Module wrappers for Dropout and FusedDropoutAdd."""
    drop = Dropout(p=0.5)
    fused_drop = FusedDropoutAdd(p=0.5)

    x = torch.randn(2, 32, 64)
    res = torch.randn(2, 32, 64)

    # In eval mode
    drop.eval()
    fused_drop.eval()
    torch.testing.assert_close(drop(x), x)
    torch.testing.assert_close(fused_drop(x, res), x + res)

    # In train mode
    drop.train()
    fused_drop.train()
    out = drop(x)
    assert out.shape == x.shape
    out_fused = fused_drop(x, res)
    assert out_fused.shape == x.shape


def test_jax_dropout_parity():
    """Test JAX dropout and fused_dropout_add implementations and gradients."""
    try:
        import jax
        import jax.numpy as jnp
        from stackkernels.jax import dropout as jax_dropout, fused_dropout_add as jax_fused_dropout_add

        x = jnp.ones((100, 100), dtype=jnp.float32)
        res = jnp.ones((100, 100), dtype=jnp.float32)

        # Eval mode
        out_eval = jax_dropout(x, p=0.5, training=False)
        assert jnp.allclose(out_eval, x)

        out_fused_eval = jax_fused_dropout_add(x, res, p=0.5, training=False)
        assert jnp.allclose(out_fused_eval, x + res)

        # Train mode
        out_train = jax_dropout(x, p=0.5, training=True, key=42)
        assert out_train.shape == x.shape
        zero_fraction = float(jnp.mean(out_train == 0.0))
        assert abs(zero_fraction - 0.5) < 0.05

        # Backward / Grad test
        grad_fn = jax.grad(lambda a: jnp.sum(jax_dropout(a, p=0.5, training=True, key=42)))
        dx = grad_fn(x)
        assert dx.shape == x.shape
        # Elements kept should have grad 2.0, dropped 0.0
        assert jnp.all(jnp.isclose(dx, 0.0) | jnp.isclose(dx, 2.0))
    except ImportError:
        pytest.skip("JAX not installed")


def test_hub_registry_dropout():
    """Verify registry resolves dropout and fused_dropout_add."""
    from stackkernels.hub.registry import get_op

    op_torch = get_op("dropout", "torch")
    assert callable(op_torch)

    op_fused_torch = get_op("fused_dropout_add", "torch")
    assert callable(op_fused_torch)

    op_jax = get_op("dropout", "jax")
    assert callable(op_jax)

    op_fused_jax = get_op("fused_dropout_add", "jax")
    assert callable(op_fused_jax)

