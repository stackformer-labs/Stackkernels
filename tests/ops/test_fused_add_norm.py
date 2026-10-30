"""End-to-end tests for Fused Add Norm forward, backward, and compilation."""

import pytest
import torch

from stackkernels._reference.torch.norm import fused_add_norm as ref_fused_add_norm_torch
from stackkernels.torch.norm import fused_add_norm as torch_fused_add_norm


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 64),
        (1, 16, 512),
        (2, 8, 2048),
    ],
)
@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_add_norm_forward_torch(shape, norm_type, has_weight, dtype, dtype_tolerances):
    """Test PyTorch fused_add_norm forward accuracy against reference implementation."""
    tol = dtype_tolerances(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(shape, dtype=dtype, device=device)
    res = torch.randn(shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device) if has_weight else None
    b = torch.randn(shape[-1], dtype=dtype, device=device) if (has_weight and norm_type == "layer_norm") else None

    out, res_out = torch_fused_add_norm(x, res, w, b, eps=1e-5, norm_type=norm_type)
    ref_out, ref_res_out = ref_fused_add_norm_torch(x, res, w, b, eps=1e-5, norm_type=norm_type)

    assert out.shape == x.shape
    assert res_out.shape == x.shape
    torch.testing.assert_close(out, ref_out, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(res_out, ref_res_out, atol=tol["atol"], rtol=tol["rtol"])


def test_fused_add_norm_forward_jax(backend_name):
    """Test JAX fused_add_norm against reference (skipped if JAX not installed)."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels._reference.jax.norm import fused_add_norm as ref_fused_add_norm_jax
    from stackkernels.jax.norm import fused_add_norm as jax_fused_add_norm

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 8, 512))
    res = jax.random.normal(key, (2, 8, 512))
    w = jax.random.normal(key, (512,))

    out, res_out = jax_fused_add_norm(x, res, w, norm_type="rms_norm")
    ref_out, ref_res_out = ref_fused_add_norm_jax(x, res, w, norm_type="rms_norm")

    assert jnp.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(res_out, ref_res_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("has_dres", [False, True])
def test_fused_add_norm_backward_torch(norm_type, has_weight, has_dres, dtype_tolerances):
    """Test PyTorch fused_add_norm backward gradients against autograd reference."""
    tol = dtype_tolerances(torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (2, 8, 256)

    x_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    res_test = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    w_test = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True) if has_weight else None
    b_test = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True) if (has_weight and norm_type == "layer_norm") else None

    out_test, res_out_test = torch_fused_add_norm(
        x_test, res_test, w_test, b_test, eps=1e-5, norm_type=norm_type
    )

    dy = torch.randn_like(out_test)
    dres = torch.randn_like(res_out_test) if has_dres else None

    # Backward on StackKernels
    if has_dres:
        torch.autograd.backward([out_test, res_out_test], [dy, dres])
    else:
        out_test.backward(dy)

    # Reference computation
    x_ref = x_test.detach().clone().requires_grad_(True)
    res_ref = res_test.detach().clone().requires_grad_(True)
    w_ref = w_test.detach().clone().requires_grad_(True) if has_weight else None
    b_ref = b_test.detach().clone().requires_grad_(True) if b_test is not None else None

    out_ref, res_out_ref = ref_fused_add_norm_torch(
        x_ref, res_ref, w_ref, b_ref, eps=1e-5, norm_type=norm_type
    )

    if has_dres:
        torch.autograd.backward([out_ref, res_out_ref], [dy, dres])
    else:
        out_ref.backward(dy)

    torch.testing.assert_close(x_test.grad, x_ref.grad, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(res_test.grad, res_ref.grad, atol=tol["atol"], rtol=tol["rtol"])

    if has_weight:
        torch.testing.assert_close(w_test.grad, w_ref.grad, atol=tol["atol"], rtol=tol["rtol"])
    if b_test is not None:
        torch.testing.assert_close(b_test.grad, b_ref.grad, atol=tol["atol"], rtol=tol["rtol"])

