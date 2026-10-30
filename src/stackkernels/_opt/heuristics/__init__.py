"""Heuristics registry for op kernel execution configurations."""

from collections.abc import Callable
from typing import Any

from stackkernels._opt.heuristics import (
    cross_entropy,
    elementwise,
    flash_attn,
    flash_decode,
    fused_add_norm,
    fused_linear_ce,
    glu,
    layer_norm,
    rms_norm,
    rope,
)

OP_HEURISTICS: dict[str, Callable[..., dict[str, Any]]] = {
    "rms_norm": rms_norm.heuristic,
    "rmsnorm": rms_norm.heuristic,
    "layer_norm": layer_norm.heuristic,
    "layernorm": layer_norm.heuristic,
    "fused_add_norm": fused_add_norm.heuristic,
    "elementwise": elementwise.heuristic,
    "gelu": elementwise.heuristic,
    "silu": elementwise.heuristic,
    "relu": elementwise.heuristic,
    "glu": glu.heuristic,
    "swiglu": glu.heuristic,
    "rope": rope.heuristic,
    "cross_entropy": cross_entropy.heuristic,
    "fused_linear_ce": fused_linear_ce.heuristic,
    "fused_linear_cross_entropy": fused_linear_ce.heuristic,
    "flash_attn": flash_attn.heuristic,
    "flash_attention": flash_attn.heuristic,
    "flash_decode": flash_decode.heuristic,
}


def register_heuristic(op_name: str, fn: Callable[..., dict[str, Any]]) -> None:
    """Register a custom heuristic function for an op."""
    OP_HEURISTICS[op_name.strip().lower()] = fn


def get_heuristic(op_name: str) -> Callable[..., dict[str, Any]] | None:
    """Retrieve the heuristic function for an op by name."""
    return OP_HEURISTICS.get(op_name.strip().lower())
