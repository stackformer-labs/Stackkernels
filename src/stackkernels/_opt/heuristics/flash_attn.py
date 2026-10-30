"""prefill and backward configs; returns kernel variant name"""

from typing import Any


def heuristic(shape: Any = None, **kwargs) -> dict:
    """Heuristic for Flash Attention (prefill & backward)."""
    arch = str(kwargs.get("arch", "sm_80")).lower()
    head_dim = kwargs.get("head_dim", 64)

    if arch in ("sm_90", "sm_100"):
        block_m, block_n = 128, 128
        stages = 3
    else:
        block_m, block_n = 64, 64
        stages = 2

    num_warps = 4 if head_dim <= 64 else 8

    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "num_warps": num_warps,
        "num_stages": stages,
        "kernel_variant": "flash_fwd",
    }
