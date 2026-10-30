"""End-to-end tests for GLU activations (SwiGLU, GeGLU, ReGLU) forward and backward."""

import pytest
import torch

from stackkernels._reference.torch.activation import (
    geglu as ref_geglu_torch,
    reglu as ref_reglu_torch,
    swiglu as ref_swiglu_torch,
)
from stackkernels.torch.activation import geglu, reglu, swiglu


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 64),
        (1, 16, 512),
        (2, 8, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swiglu_forward_torch(shape, dtype, dtype_tolerances):
    """Test SwiGLU forward accuracy against reference."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gate = torch.randn(shape, dtype=dtype, device=device)
    up = torch.randn(shape, dtype=dtype, device=device)

    out = swiglu(gate, up)
    ref_out = ref_swiglu_torch(gate, up)

    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])


def test_swiglu_chunk_input():
    """Verify swiglu handles packed [gate, up] tensor along last dimension."""
    x = torch.randn(2, 4, 128)
    out = swiglu(x)
    assert out.shape == (2, 4, 64)

    gate, up = torch.chunk(x, 2, dim=-1)
    ref_out = ref_swiglu_torch(gate, up)
    torch.testing.assert_close(out, ref_out)


@pytest.mark.parametrize("glu_fn, ref_fn", [
    (swiglu, ref_swiglu_torch),
    (geglu, ref_geglu_torch),
    (reglu, ref_reglu_torch),
])
def test_glu_backward_torch(glu_fn, ref_fn, dtype_tolerances):
    """Test GLU backward gradients against reference autograd."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (2, 8, 256)

    gate_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    up_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    y_test = glu_fn(gate_test, up_test)

    dy = torch.randn_like(y_test)
    y_test.backward(dy)

    gate_ref = gate_test.detach().clone().requires_grad_(True)
    up_ref = up_test.detach().clone().requires_grad_(True)
    y_ref = ref_fn(gate_ref, up_ref)
    y_ref.backward(dy)

    torch.testing.assert_close(gate_test.grad, gate_ref.grad, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(up_test.grad, up_ref.grad, atol=tol["atol"], rtol=tol["rtol"])


def test_glu_forward_jax(backend_name):
    """Test JAX GLU activations (skipped if JAX not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.activation import swiglu as ref_swiglu_jax
    from stackkernels.jax.activation import swiglu as jax_swiglu

    key = jax.random.PRNGKey(42)
    gate = jax.random.normal(key, (2, 8, 256))
    up = jax.random.normal(key, (2, 8, 256))

    out = jax_swiglu(gate, up)
    ref_out = ref_swiglu_jax(gate, up)

    assert jnp.allclose(out, ref_out, atol=1e-5, rtol=1e-5)

