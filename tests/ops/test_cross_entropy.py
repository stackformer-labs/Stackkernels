"""Tests for Cross Entropy loss operator."""

import pytest
import torch

from stackkernels.torch.loss import cross_entropy
import stackkernels._reference.torch.loss as ref_loss


@pytest.mark.parametrize("shape", [(16, 256), (4, 16, 1024), (2, 8, 4096)])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_forward(shape, label_smoothing, reduction):
    torch.manual_seed(42)
    vocab_size = shape[-1]
    target_shape = shape[:-1]

    logits = torch.randn(shape, dtype=torch.float32)
    target = torch.randint(0, vocab_size, target_shape, dtype=torch.long)
    # Add some ignore_index entries
    target[target < vocab_size // 10] = -100

    out = cross_entropy(
        logits,
        target,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    expected = ref_loss.cross_entropy(
        logits,
        target,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_cross_entropy_backward(reduction, label_smoothing):
    torch.manual_seed(42)
    B, S, V = 2, 8, 512

    logits1 = torch.randn(B, S, V, dtype=torch.float32, requires_grad=True)
    logits2 = logits1.detach().clone().requires_grad_(True)

    target = torch.randint(0, V, (B, S), dtype=torch.long)
    target[0, 0] = -100
    target[1, 3] = -100

    loss1 = cross_entropy(
        logits1,
        target,
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    loss2 = ref_loss.cross_entropy(
        logits2,
        target,
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

    torch.testing.assert_close(logits1.grad, logits2.grad, rtol=1e-4, atol=1e-4)

    # Ignored targets should have 0 gradient
    flat_grad = logits1.grad.reshape(-1, V)
    flat_target = target.reshape(-1)
    ignored_mask = flat_target == -100
    if ignored_mask.any():
        assert torch.all(flat_grad[ignored_mask] == 0.0)


def test_cross_entropy_all_ignored():
    """Edge case: all tokens in batch are ignored."""
    logits = torch.randn(4, 128, requires_grad=True)
    target = torch.full((4,), -100, dtype=torch.long)

    loss = cross_entropy(logits, target, ignore_index=-100, reduction="mean")
    assert loss.item() == 0.0

    loss.backward()
    assert torch.all(logits.grad == 0.0)


def test_cross_entropy_jax(backend_name):
    """Test JAX cross entropy wrapper."""
    if backend_name != "jax":
        return

    import jax
    import jax.numpy as jnp
    from stackkernels.jax.loss import cross_entropy as jax_ce

    key = jax.random.PRNGKey(42)
    logits = jax.random.normal(key, (4, 128))
    target = jax.random.randint(key, (4,), 0, 128)

    loss = jax_ce(logits, target, reduction="mean")
    assert not jnp.isnan(loss)
