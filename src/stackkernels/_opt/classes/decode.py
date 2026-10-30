"""Seed configurations for FlashDecode (Split-KV single-token generation)."""

import math
from typing import Any

from stackkernels._opt.hardware import DeviceProfile


def seed_configs(
    profile: DeviceProfile,
    problem: Any = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ordered candidate configs (best guess first) for decode kernels."""
    kv_len = kwargs.get("kv_len", 1024)
    if isinstance(problem, dict) and "kv_len" in problem:
        kv_len = problem["kv_len"]
    elif isinstance(problem, (list, tuple)) and len(problem) >= 3:
        kv_len = problem[2]

    # If sequence is small, split-KV overhead dominates
    if kv_len <= 128:
        base_splits = 1
    else:
        sm_count = max(profile.sm_count, 1)
        # Aim for enough splits so SMs are saturated
        target_splits = max(1, min(kv_len // 256, 16))
        base_splits = target_splits

    seeds: list[dict[str, Any]] = [
        {
            "SPLIT_KV": base_splits,
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 1,
            "kernel_variant": "decode",
        }
    ]

    candidates = [
        base_splits // 2,
        base_splits * 2,
        4,
        8,
        16,
    ]

    for splits in candidates:
        if splits >= 1 and splits <= 32:
            cand_dict = {
                "SPLIT_KV": splits,
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "num_warps": 4,
                "num_stages": 1,
                "kernel_variant": "decode",
            }
            if cand_dict != seeds[0] and cand_dict not in seeds:
                seeds.append(cand_dict)

    return seeds


__all__ = ["seed_configs"]

