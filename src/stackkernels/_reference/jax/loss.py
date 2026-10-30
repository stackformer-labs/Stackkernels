"""Loss reference implementations in JAX."""

from typing import Any, Literal


def cross_entropy(
    logits: Any,
    target: Any,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Any:
    """Cross entropy loss in JAX with ignore_index and label smoothing."""
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size).astype(jnp.float32)
    target_flat = target.reshape(-1)

    # Compute log-softmax
    lse = jnn.logsumexp(logits_flat, axis=-1, keepdims=True)
    log_probs = logits_flat - lse

    valid_mask = target_flat != ignore_index
    safe_targets = jnp.where(valid_mask, target_flat, 0)

    # NLL loss
    nll = -jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)

    if label_smoothing > 0.0:
        smooth_loss = -jnp.mean(log_probs, axis=-1)
        loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
    else:
        loss = nll

    loss = jnp.where(valid_mask, loss, 0.0)

    if reduction == "none":
        return loss.reshape(target.shape)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        valid_count = jnp.maximum(jnp.sum(valid_mask), 1.0)
        return jnp.sum(loss) / valid_count


def fused_linear_cross_entropy(
    x: Any,
    weight: Any,
    target: Any,
    bias: Any | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Any:
    """Fused linear projection + cross entropy in JAX."""
    import jax.numpy as jnp

    logits = jnp.dot(x, weight.T)
    if bias is not None:
        logits = logits + bias
    return cross_entropy(
        logits,
        target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


__all__ = ["cross_entropy", "fused_linear_cross_entropy"]
