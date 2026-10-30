"""Seed configurations for row-reduction kernels (RMSNorm, LayerNorm, CrossEntropy, RoPE)."""

import math
from typing import Any

from stackkernels._opt.hardware import DeviceProfile


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def seed_configs(
    profile: DeviceProfile,
    problem: Any = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ordered candidate configs (best guess first) for row reduction kernels."""
    # Determine row length (cols)
    if isinstance(problem, int):
        cols = problem
    elif isinstance(problem, (list, tuple)) and len(problem) > 0:
        cols = problem[-1]
    else:
        cols = kwargs.get("cols", kwargs.get("vocab_size", 1024))

    base_block = _next_power_of_2(cols)

    # If row exceeds single-pass register budget on device, use chunked loop
    max_single_pass = 8192 if profile.is_cuda and profile.compute_capability >= (8, 0) else 4096
    is_chunked = cols > max_single_pass

    if is_chunked:
        block_size = 2048
        warps = 8
    else:
        block_size = min(max(base_block, 32), max_single_pass)
        if block_size <= 128:
            warps = 1 if block_size <= 64 else 2
        elif block_size <= 512:
            warps = 4
        elif block_size <= 2048:
            warps = 8
        else:
            warps = 16 if profile.is_cuda and profile.max_threads_per_sm >= 1024 else 8

    # Primary seed (heuristic choice)
    seeds: list[dict[str, Any]] = [
        {"BLOCK_SIZE": block_size, "num_warps": warps, "num_stages": 1}
    ]

    # Additional candidates around the primary seed
    if is_chunked:
        seeds.append({"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1})
        seeds.append({"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1})
        seeds.append({"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 1})
    else:
        # Half block size with fewer warps
        if block_size > 64:
            seeds.append({
                "BLOCK_SIZE": block_size // 2,
                "num_warps": max(1, warps // 2),
                "num_stages": 1,
            })

        # Same block size with double warps (if allowed)
        if warps < 16:
            seeds.append({
                "BLOCK_SIZE": block_size,
                "num_warps": min(16, warps * 2),
                "num_stages": 1,
            })

        # Double block size if well below limit
        if block_size * 2 <= max_single_pass:
            seeds.append({
                "BLOCK_SIZE": block_size * 2,
                "num_warps": min(16, warps * 2),
                "num_stages": 1,
            })

    return seeds


__all__ = ["seed_configs"]

