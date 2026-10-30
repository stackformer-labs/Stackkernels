"""Save versus recompute decisions for backward activation passes."""

from enum import Enum
from typing import Any


class SavePolicy(str, Enum):
    """Memory save policy for backward execution."""

    RECOMPUTE = "recompute"  # Recompute intermediate values in SRAM/registers to save HBM bandwidth
    SAVE = "save"  # Save intermediate values to HBM in forward and reload in backward


# Default policy per operator category
DEFAULT_OP_POLICIES: dict[str, SavePolicy] = {
    # Memory-bound activations: recomputing is strictly faster than 2x HBM traffic
    "glu": SavePolicy.RECOMPUTE,
    "swiglu": SavePolicy.RECOMPUTE,
    "geglu": SavePolicy.RECOMPUTE,
    "reglu": SavePolicy.RECOMPUTE,
    "elementwise": SavePolicy.RECOMPUTE,
    "gelu": SavePolicy.RECOMPUTE,
    "silu": SavePolicy.RECOMPUTE,
    "relu": SavePolicy.RECOMPUTE,
    "sigmoid": SavePolicy.RECOMPUTE,
}


def should_recompute(
    op: str,
    shape: Any = None,
    dtype: str = "float16",
    policy_override: SavePolicy | str | None = None,
) -> bool:
    """Decide whether to recompute intermediate activations in backward or load saved tensors.

    Args:
        op: Operator name (e.g., 'swiglu', 'gelu').
        shape: Tensor dimensions.
        dtype: Data type string.
        policy_override: Explicit user override.

    Returns:
        True if the backward pass should recompute on-chip; False to load from HBM.
    """
    if policy_override is not None:
        if isinstance(policy_override, str):
            return policy_override.lower().strip() == SavePolicy.RECOMPUTE.value
        return policy_override == SavePolicy.RECOMPUTE

    op_norm = op.strip().lower()
    policy = DEFAULT_OP_POLICIES.get(op_norm, SavePolicy.RECOMPUTE)
    return policy == SavePolicy.RECOMPUTE


__all__ = ["SavePolicy", "should_recompute"]
