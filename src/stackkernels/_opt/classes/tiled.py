"""Seed configurations for 2D tiled kernels (Flash Attention, Fused Linear CE)."""

from typing import Any

from stackkernels._opt.hardware import DeviceProfile


def seed_configs(
    profile: DeviceProfile,
    problem: Any = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ordered candidate configs (best guess first) for 2D tiled kernels."""
    # Determine head_dim D
    head_dim = 64
    if isinstance(problem, dict):
        head_dim = problem.get("head_dim", 64)
    elif isinstance(problem, (list, tuple)) and len(problem) >= 4:
        head_dim = problem[-1]
    else:
        head_dim = kwargs.get("head_dim", 64)

    is_modern_gpu = profile.is_cuda and profile.has_cp_async

    # Select base tile size and stages based on architecture and head dimension
    if head_dim >= 256:
        if is_modern_gpu:
            base_m, base_n = 64, 32
            base_stages = 2
            base_warps = 8
        else:
            base_m, base_n = 32, 16
            base_stages = 1
            base_warps = 4
    elif is_modern_gpu:
        # Ampere/Hopper (sm80+) has cp.async and large shared memory
        base_m = 128 if head_dim <= 64 else 64
        base_n = 64
        base_stages = 3 if profile.compute_capability >= (9, 0) else 2
        base_warps = 4 if head_dim <= 64 else 4
    else:
        # Turing (sm75) or fallback: smaller shared memory per block (48KB)
        base_m = 64
        base_n = 32 if head_dim >= 128 else 64
        base_stages = 1
        base_warps = 4

    seeds: list[dict[str, Any]] = [
        {
            "BLOCK_M": base_m,
            "BLOCK_N": base_n,
            "num_warps": base_warps,
            "num_stages": base_stages,
            "kernel_variant": "flash_fwd",
        }
    ]

    # Additional candidates around the primary seed
    candidates = [
        {"BLOCK_M": 32, "BLOCK_N": 16, "num_warps": 4, "num_stages": 1},
        {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1},
        {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1 if not is_modern_gpu else 2},
        {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4 if head_dim <= 64 else 8, "num_stages": 2 if is_modern_gpu else 1},
        {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2 if is_modern_gpu else 1},
    ]

    for cand in candidates:
        cand_dict = dict(cand)
        cand_dict["kernel_variant"] = "flash_fwd"
        if cand_dict != seeds[0] and cand_dict not in seeds:
            seeds.append(cand_dict)

    return seeds


__all__ = ["seed_configs"]
