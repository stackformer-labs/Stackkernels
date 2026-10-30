"""Tests for Fused Linear + Cross Entropy operator."""

import pytest
import torch

from stackkernels.torch.loss import (
    FusedLinearCrossEntropyLoss,
    fused_linear_cross_entropy,
)
import stackkernels._reference.torch.loss as ref_loss


@pytest.mark.parametrize("shape", [(16, 64, 256), (4, 16, 128, 512)])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_fused_linear_ce_forward(shape, has_bias, label_smoothing, reduction):
    torch.manual_seed(42)
    if len(shape) == 3:
        M, K, V = shape
        x = torch.randn(M, K, dtype=torch.float32)
        target = torch.randint(0, V, (M,), dtype=torch.long)
    else:
        B, S, K, V = shape
        x = torch.randn(B, S, K, dtype=torch.float32)
        target = torch.randint(0, V, (B, S), dtype=torch.long)

    weight = torch.randn(V, K, dtype=torch.float32)
    bias = torch.randn(V, dtype=torch.float32) if has_bias else None

    # Inject some ignore_index entries
    target[target < V // 10] = -100

    out = fused_linear_cross_entropy(
        x,
        weight,
        target,
        bias=bias,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    expected = ref_loss.fused_linear_cross_entropy(
        x,
        weight,
        target,
        bias=bias,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_fused_linear_ce_backward(has_bias, label_smoothing, reduction):
    torch.manual_seed(42)
    B, S, K, V = 2, 8, 32, 128

    x1 = torch.randn(B, S, K, dtype=torch.float32, requires_grad=True)
    w1 = torch.randn(V, K, dtype=torch.float32, requires_grad=True)
    b1 = torch.randn(V, dtype=torch.float32, requires_grad=True) if has_bias else None

    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True) if has_bias else None

    target = torch.randint(0, V, (B, S), dtype=torch.long)
    target[0, 0] = -100
    target[1, 2] = -100

    loss1 = fused_linear_cross_entropy(
        x1,
        w1,
        target,
        bias=b1,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    loss2 = ref_loss.fused_linear_cross_entropy(
        x2,
        w2,
        target,
        bias=b2,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    if reduction == "none":
        grad_out = torch.randn_like(loss1)
        loss1.backward(grad_out)
        loss2.backward(grad_out)
    else:
        loss1.backward()
        loss2.backward()

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-4, atol=1e-4)
    if has_bias:
        torch.testing.assert_close(b1.grad, b2.grad, rtol=1e-4, atol=1e-4)


def test_fused_linear_ce_module():
    torch.manual_seed(42)
    M, K, V = 8, 16, 64
    x = torch.randn(M, K)
    w = torch.randn(V, K)
    target = torch.randint(0, V, (M,))

    criterion = FusedLinearCrossEntropyLoss(ignore_index=-100, label_smoothing=0.0)
    loss = criterion(x, w, target)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_fused_linear_ce_jax(backend_name):
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels.jax.loss import fused_linear_cross_entropy as jax_flce

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4, 16))
    w = jax.random.normal(key, (32, 16))
    target = jax.random.randint(key, (4,), 0, 32)

    loss = jax_flce(x, w, target, reduction="mean")
    assert not jnp.isnan(loss)
