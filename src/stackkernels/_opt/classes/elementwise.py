"""Seed configurations for elementwise streaming kernels (GELU, SiLU, GLU, Dropout)."""

import math
from typing import Any

from stackkernels._opt.hardware import DeviceProfile


def seed_configs(
    profile: DeviceProfile,
    problem: Any = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ordered candidate configs (best guess first) for elementwise kernels."""
    # Determine total elements
    if isinstance(problem, int):
        total_elements = problem
    elif isinstance(problem, (list, tuple)) and len(problem) > 0:
        total_elements = math.prod(problem)
    else:
        total_elements = kwargs.get("total_elements", 1024 * 1024)

    # Calculate target program count so every SM gets at least 2 resident programs
    sm_count = max(profile.sm_count, 1)
    target_programs = sm_count * 2

    # Vectorized default block size: 1024
    if total_elements <= target_programs * 256:
        base_block = 256
        warps = 4
    elif total_elements <= target_programs * 512:
        base_block = 512
        warps = 4
    elif total_elements >= 4 * 1024 * 1024:
        base_block = 2048
        warps = 8
    else:
        base_block = 1024
        warps = 4

    seeds: list[dict[str, Any]] = [
        {"BLOCK_SIZE": base_block, "num_warps": warps, "num_stages": 1}
    ]

    # Additional candidates around the primary seed
    candidates = [
        {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 1},
        {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1},
        {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 1},
        {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1},
    ]

    for cand in candidates:
        if cand != seeds[0] and cand not in seeds:
            seeds.append(cand)

    return seeds


__all__ = ["seed_configs"]
