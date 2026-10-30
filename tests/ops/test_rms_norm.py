"""End-to-end tests for RMSNorm forward, backward, and compilation."""

import pytest
import torch

from stackkernels._reference.torch.norm import rms_norm as ref_rms_norm_torch
from stackkernels.torch.norm import RMSNorm, rms_norm as torch_rms_norm


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 64),
        (1, 16, 512),
        (2, 8, 4096),
        (1, 2, 8192),
    ],
)
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_forward_torch(shape, has_weight, dtype, dtype_tolerances):
    """Test PyTorch RMSNorm forward accuracy against reference implementation."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device) if has_weight else None

    out = torch_rms_norm(x, w, eps=1e-6)
    ref_out = ref_rms_norm_torch(x, w, eps=1e-6)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])


def test_rms_norm_forward_jax(backend_name):
    """Test JAX RMSNorm forward against JAX reference (skipped if JAX is not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.norm import rms_norm as ref_rms_norm_jax
    from stackkernels.jax.norm import rms_norm as jax_rms_norm

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 8, 512))
    w = jax.random.normal(key, (512,))

    out = jax_rms_norm(x, w)
    ref_out = ref_rms_norm_jax(x, w)

    assert jnp.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("has_weight", [False, True])
def test_rms_norm_backward_torch(has_weight, dtype_tolerances):
    """Test PyTorch RMSNorm backward gradient accuracy against reference autograd."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (2, 8, 256)

    # 1. StackKernels RMSNorm
    x_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    w_test = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True) if has_weight else None
    y_test = torch_rms_norm(x_test, w_test, eps=1e-6)

    dy = torch.randn_like(y_test)
    y_test.backward(dy)

    # 2. Reference implementation
    x_ref = x_test.detach().clone().requires_grad_(True)
    w_ref = w_test.detach().clone().requires_grad_(True) if has_weight else None
    y_ref = ref_rms_norm_torch(x_ref, w_ref, eps=1e-6)
    y_ref.backward(dy)

    # Assert dx matches
    torch.testing.assert_close(x_test.grad, x_ref.grad, atol=tol["atol"], rtol=tol["rtol"])

    # Assert dw matches
    if has_weight:
        torch.testing.assert_close(w_test.grad, w_ref.grad, atol=tol["atol"], rtol=tol["rtol"])


def test_rms_norm_module():
    """Verify RMSNorm nn.Module layer functionality and backward parameter gradient."""
    dim = 512
    module = RMSNorm(dim=dim)
    x = torch.randn(2, 4, dim, requires_grad=True)

    y = module(x)
    assert y.shape == x.shape

    loss = y.sum()
    loss.backward()

    assert module.weight.grad is not None
    assert module.weight.grad.shape == module.weight.shape
    assert x.grad is not None


def test_rms_norm_torch_compile_no_graph_break():
    """Verify RMSNorm compiles cleanly under TorchDynamo without graph breaks."""
    try:
        import torch._dynamo as dynamo

        layer = RMSNorm(256)
        x = torch.randn(2, 4, 256)

        explanation = dynamo.explain(layer)(x)
        assert explanation.graph_break_count == 0
    except (ImportError, RuntimeError, AttributeError):
        pytest.skip("TorchDynamo is not available or unsupported on this environment")
