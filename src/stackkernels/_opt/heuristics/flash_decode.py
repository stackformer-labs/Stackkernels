"""decode configs and split-KV count"""

from typing import Any


def heuristic(shape: Any = None, **kwargs) -> dict:
    """Heuristic for Flash Decode (split-KV attention for q_len=1)."""
    kv_len = kwargs.get("kv_len", 1024)
    split_kv = max(1, min(kv_len // 256, 16))

    return {
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "SPLIT_KV": split_kv,
        "num_warps": 4,
        "kernel_variant": "decode",
    }
