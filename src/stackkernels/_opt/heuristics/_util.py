"""next_pow2, warp-count rules"""

import triton

next_pow2 = triton.next_power_of_2

__all__ = ["next_pow2", "determine_num_warps", "get_num_warps"]


def determine_num_warps(block_size: int, max_warps: int = 16) -> int:
    """Determine number of warps dynamically based on block size (32 threads per warp)."""
    # 32 threads per warp; power-of-2 clamped between 1 and max_warps
    threads = max(1, block_size)
    warps = max(1, threads // 32)
    warps = min(next_pow2(warps), max_warps)
    return warps


get_num_warps = determine_num_warps
