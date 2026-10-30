"""Static constraint checking to prune invalid kernel candidate configurations."""

from typing import Any

from stackkernels._opt.hardware import DeviceProfile
from stackkernels._opt.occupancy import accumulator_regs_per_thread


def _get_itemsize(dtype: str) -> int:
    d = dtype.lower()
    if d in ("float16", "bfloat16", "fp16", "bf16", "half"):
        return 2
    if d in ("float32", "fp32", "int32"):
        return 4
    if d in ("int8", "uint8"):
        return 1
    return 2


def check(
    cfg: dict[str, Any],
    profile: DeviceProfile,
    kernel_class: str,
    dtype: str = "float16",
    problem: Any = None,
) -> list[str]:
    """Validate a candidate configuration against hardware limits and class rules.

    Returns:
        list of string rejection reasons. Empty list indicates candidate is valid.
    """
    rejections: list[str] = []
    d = dtype.lower()
    kclass = kernel_class.lower().strip()

    # 1. Dtype support check
    if profile.is_cuda and d in ("bfloat16", "bf16"):
        if not profile.has_bf16:
            rejections.append(
                f"bfloat16 is not natively supported on compute capability {profile.compute_capability} (requires sm80+)"
            )

    # 2. Thread count limit
    num_warps = cfg.get("num_warps", 4)
    threads = 32 * num_warps
    if profile.is_cuda:
        if threads > 1024:
            rejections.append(f"Thread count {threads} exceeds maximum hardware block limit 1024")
        elif threads > profile.max_threads_per_sm:
            rejections.append(f"Thread count {threads} exceeds SM limit {profile.max_threads_per_sm}")

    # 3. Architecture num_stages limit
    stages = cfg.get("num_stages", 1)
    if profile.is_cuda:
        if stages > 2 and not profile.has_cp_async:
            rejections.append(
                f"num_stages={stages} requires hardware cp.async (compute capability sm80+)"
            )
        elif stages > 4 and profile.compute_capability < (9, 0):
            rejections.append(
                f"num_stages={stages} exceeds pipeline depth limit for sm{profile.compute_capability[0]}{profile.compute_capability[1]}"
            )

    itemsize = _get_itemsize(dtype)

    # 4. Shared memory budget check (tiled kernels)
    if kclass == "tiled" and profile.is_cuda:
        block_m = cfg.get("BLOCK_M", 64)
        block_n = cfg.get("BLOCK_N", 64)
        head_dim = 64
        if isinstance(problem, dict) and "head_dim" in problem:
            head_dim = problem["head_dim"]
        elif isinstance(problem, (list, tuple)) and len(problem) >= 4:
            head_dim = problem[-1]

        # Shared memory: stages * (BLOCK_M * D + 2 * BLOCK_N * D) * bytes
        est_smem = stages * (block_m * head_dim + 2 * block_n * head_dim) * itemsize
        max_allowed_smem = int(profile.max_smem_per_block * 0.90)  # 10% safety margin
        if max_allowed_smem > 0 and est_smem > max_allowed_smem:
            rejections.append(
                f"Estimated shared memory {est_smem}B exceeds safe block limit {max_allowed_smem}B"
            )

        # 5. Accumulator register budget check
        acc_regs = accumulator_regs_per_thread(block_m, block_n, num_warps)
        max_acc_regs = int(profile.max_regs_per_thread * 0.45)
        if profile.max_regs_per_thread > 0 and acc_regs > max_acc_regs:
            rejections.append(
                f"Accumulator registers per thread {acc_regs} exceeds budget {max_acc_regs}"
            )

    # 6. Tile vs problem size
    if kclass == "row":
        block_size = cfg.get("BLOCK_SIZE", cfg.get("BLOCK_V", 1024))
        if block_size > 16384:
            rejections.append(f"BLOCK_SIZE {block_size} exceeds maximum row limit 16384")
        if problem is not None:
            row_size = (
                problem
                if isinstance(problem, int)
                else (problem[-1] if isinstance(problem, (list, tuple)) and len(problem) > 0 else None)
            )
            if row_size is not None and row_size >= 32:
                if block_size < row_size:
                    rejections.append(
                        f"BLOCK_SIZE {block_size} is smaller than row dimension {row_size}"
                    )
                elif block_size > 8 * row_size:
                    rejections.append(
                        f"BLOCK_SIZE {block_size} is absurdly larger than row dimension {row_size} (>8x)"
                    )

    return rejections


__all__ = ["check"]
