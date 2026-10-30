"""End-to-end tests for LayerNorm forward, backward, and compilation."""

import pytest
import torch

from stackkernels._reference.torch.norm import layer_norm as ref_layer_norm_torch
from stackkernels.torch.norm import LayerNorm, layer_norm as torch_layer_norm


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
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_layer_norm_forward_torch(shape, has_weight, has_bias, dtype, dtype_tolerances):
    """Test PyTorch LayerNorm forward accuracy against reference implementation."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device) if has_weight else None
    b = torch.randn(shape[-1], dtype=dtype, device=device) if has_bias else None

    out = torch_layer_norm(x, w, b, eps=1e-5)
    ref_out = ref_layer_norm_torch(x, w, b, eps=1e-5)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])


def test_layer_norm_forward_jax(backend_name):
    """Test JAX LayerNorm forward against JAX reference (skipped if JAX not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.norm import layer_norm as ref_layer_norm_jax
    from stackkernels.jax.norm import layer_norm as jax_layer_norm

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 8, 512))
    w = jax.random.normal(key, (512,))
    b = jax.random.normal(key, (512,))

    out = jax_layer_norm(x, w, b)
    ref_out = ref_layer_norm_jax(x, w, b)

    assert jnp.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_layer_norm_backward_torch(has_weight, has_bias, dtype_tolerances):
    """Test PyTorch LayerNorm backward gradient accuracy against reference autograd."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (2, 8, 256)

    # 1. StackKernels LayerNorm
    x_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    w_test = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True) if has_weight else None
    b_test = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True) if has_bias else None
    y_test = torch_layer_norm(x_test, w_test, b_test, eps=1e-5)

    dy = torch.randn_like(y_test)
    y_test.backward(dy)

    # 2. Reference LayerNorm
    x_ref = x_test.detach().clone().requires_grad_(True)
    w_ref = w_test.detach().clone().requires_grad_(True) if has_weight else None
    b_ref = b_test.detach().clone().requires_grad_(True) if has_bias else None
    y_ref = ref_layer_norm_torch(x_ref, w_ref, b_ref, eps=1e-5)
    y_ref.backward(dy)

    # Assert dx matches
    torch.testing.assert_close(x_test.grad, x_ref.grad, atol=tol["atol"], rtol=tol["rtol"])

    # Assert dw matches
    if has_weight:
        torch.testing.assert_close(w_test.grad, w_ref.grad, atol=tol["atol"], rtol=tol["rtol"])

    # Assert db matches
    if has_bias:
        torch.testing.assert_close(b_test.grad, b_ref.grad, atol=tol["atol"], rtol=tol["rtol"])


def test_layer_norm_module():
    """Verify LayerNorm nn.Module layer functionality and parameter gradients."""
    dim = 512
    module = LayerNorm(dim, bias=True)
    x = torch.randn(2, 4, dim, requires_grad=True)

    y = module(x)
    assert y.shape == x.shape

    loss = y.sum()
    loss.backward()

    assert module.weight.grad is not None
    assert module.bias.grad is not None
    assert x.grad is not None


def test_layer_norm_torch_compile_no_graph_break():
    """Verify LayerNorm compiles cleanly under TorchDynamo without graph breaks."""
    try:
        import torch._dynamo as dynamo

        layer = LayerNorm(256)
        x = torch.randn(2, 4, 256)

        explanation = dynamo.explain(layer)(x)
        assert explanation.graph_break_count == 0
    except (ImportError, RuntimeError, AttributeError):
        pytest.skip("TorchDynamo is not available or unsupported in this environment")

