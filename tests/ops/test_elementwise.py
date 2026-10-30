"""End-to-end tests for elementwise activations forward and backward."""

import pytest
import torch

from stackkernels._reference.torch.activation import (
    gelu as ref_gelu_torch,
    leaky_relu as ref_leaky_relu_torch,
    relu as ref_relu_torch,
    sigmoid as ref_sigmoid_torch,
    silu as ref_silu_torch,
)
from stackkernels.torch.activation import (
    gelu,
    leaky_relu,
    relu,
    sigmoid,
    silu,
)


@pytest.mark.parametrize(
    "act_fn, ref_fn, kwargs",
    [
        (silu, ref_silu_torch, {}),
        (relu, ref_relu_torch, {}),
        (sigmoid, ref_sigmoid_torch, {}),
        (gelu, ref_gelu_torch, {"approximate": "none"}),
        (gelu, ref_gelu_torch, {"approximate": "tanh"}),
        (leaky_relu, ref_leaky_relu_torch, {"negative_slope": 0.05}),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elementwise_forward_torch(act_fn, ref_fn, kwargs, dtype, dtype_tolerances):
    """Test elementwise activation forward accuracy against reference."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 8, 512, dtype=dtype, device=device)

    out = act_fn(x, **kwargs)
    ref_out = ref_fn(x, **kwargs)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.parametrize(
    "act_fn, ref_fn, kwargs",
    [
        (silu, ref_silu_torch, {}),
        (relu, ref_relu_torch, {}),
        (sigmoid, ref_sigmoid_torch, {}),
        (gelu, ref_gelu_torch, {"approximate": "none"}),
        (gelu, ref_gelu_torch, {"approximate": "tanh"}),
        (leaky_relu, ref_leaky_relu_torch, {"negative_slope": 0.05}),
    ],
)
def test_elementwise_backward_torch(act_fn, ref_fn, kwargs, dtype_tolerances):
    """Test elementwise activation backward gradients against autograd reference."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_test = torch.randn(2, 8, 256, dtype=torch.float32, device=device, requires_grad=True)
    y_test = act_fn(x_test, **kwargs)
    dy = torch.randn_like(y_test)
    y_test.backward(dy)

    x_ref = x_test.detach().clone().requires_grad_(True)
    y_ref = ref_fn(x_ref, **kwargs)
    y_ref.backward(dy)

    torch.testing.assert_close(x_test.grad, x_ref.grad, atol=tol["atol"], rtol=tol["rtol"])


def test_elementwise_forward_jax(backend_name):
    """Test JAX elementwise activations (skipped if JAX not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.activation import silu as ref_silu_jax
    from stackkernels.jax.activation import silu as jax_silu

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 8, 256))

    out = jax_silu(x)
    ref_out = ref_silu_jax(x)
    assert jnp.allclose(out, ref_out, atol=1e-5, rtol=1e-5)

