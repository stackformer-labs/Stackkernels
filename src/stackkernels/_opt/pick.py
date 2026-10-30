"""Resolution order: user override > user cache > shipped tuned file > class heuristic."""

from typing import Any

from stackkernels._opt.cache import (
    TuningCache,
    get_current_kernel_rev,
    get_current_triton_version,
    get_global_cache,
)
from stackkernels._opt.classes import decode, elementwise, row, tiled
from stackkernels._opt.constraints import check
from stackkernels._opt.hardware import DeviceProfile, get_device_profile
from stackkernels._opt.occupancy import programs_per_sm, waves
from stackkernels._opt.search_spaces import get_kernel_class


def pick(
    op: str,
    shape: Any = None,
    dtype: str = "float16",
    arch: str | None = None,
    override: dict[str, Any] | None = None,
    cache: TuningCache | None = None,
    profile: DeviceProfile | None = None,
    explain: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Pick kernel execution configuration following the resolution order:

    1. User Override (highest precedence)
    2. User Cache (user-tuned configurations)
    3. Shipped Tuned File (matching device fingerprint)
    4. Class Heuristic (seed rules derived from DeviceProfile)

    If explain=True, returns a tuple of (config, explanation_str).
    """
    op_name = str(op).strip().lower()
    prof = profile or get_device_profile()
    fingerprint = arch or prof.fingerprint()
    kernel_rev = get_current_kernel_rev(op_name)
    triton_ver = get_current_triton_version()
    target_cache = cache if cache is not None else get_global_cache()
    bucket = kwargs.get("bucket", shape)

    chosen_config: dict[str, Any] | None = None
    source: str | None = None
    rejections: list[tuple[dict[str, Any], list[str]]] = []
    kclass = get_kernel_class(op_name)

    # 1. User override
    if override:
        chosen_config = dict(override)
        source = "user_override"

    # 2. User cache
    if chosen_config is None:
        user_entry = target_cache.get_user(
            op=op_name,
            arch=fingerprint,
            dtype=dtype,
            bucket=bucket,
            triton_version=triton_ver,
            kernel_rev=kernel_rev,
        )
        if user_entry is not None:
            chosen_config = dict(user_entry)
            source = "user_cache"

    # 3. Shipped tuned file (matching fingerprint)
    if chosen_config is None:
        shipped_entry = target_cache.get_shipped(
            op=op_name,
            arch=fingerprint,
            dtype=dtype,
            bucket=bucket,
            triton_version=triton_ver,
            kernel_rev=kernel_rev,
        )
        if shipped_entry is not None:
            chosen_config = dict(shipped_entry)
            source = "shipped_tuned"

    # 4. Class heuristic
    if chosen_config is None:
        if kclass == "row":
            seeds = row.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
        elif kclass == "elementwise":
            seeds = elementwise.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
        elif kclass == "tiled":
            seeds = tiled.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
        elif kclass == "decode":
            seeds = decode.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
        else:
            seeds = [{"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1}]

        chosen_config = dict(seeds[0])
        source = f"heuristic ({kclass} seed)"

        # Inspect rejected variations if explain is requested
        if explain:
            # Test variations to record any rejected candidates and why
            test_candidates: list[dict[str, Any]] = []
            if kclass in ("row", "elementwise"):
                bs = chosen_config.get("BLOCK_SIZE", 1024)
                w = chosen_config.get("num_warps", 4)
                for test_bs in (bs * 4, bs * 8):
                    test_candidates.append({"BLOCK_SIZE": test_bs, "num_warps": w, "num_stages": 1})
                for test_w in (32, 64):
                    test_candidates.append({"BLOCK_SIZE": bs, "num_warps": test_w, "num_stages": 1})
            elif kclass == "tiled":
                bm = chosen_config.get("BLOCK_M", 64)
                bn = chosen_config.get("BLOCK_N", 64)
                test_candidates.append({"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 4, "num_stages": 3})
                test_candidates.append({"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": 32, "num_stages": 1})
            elif kclass == "decode":
                test_candidates.append({"SPLIT_KV": 128, "BLOCK_M": 16, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1})

            for cand in test_candidates:
                errs = check(cand, prof, kernel_class=kclass, dtype=dtype, problem=shape)
                if errs:
                    rejections.append((cand, errs))

    # If explain is not requested, return chosen config directly
    if not explain:
        return chosen_config

    # Build detailed human-readable explanation
    # Wave and grid calculation
    if kclass in ("row", "elementwise"):
        grid_size = shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 1
    elif kclass == "tiled":
        bm = chosen_config.get("BLOCK_M", 64)
        seq_q = shape[2] if isinstance(shape, (list, tuple)) and len(shape) >= 4 else 128
        bh = shape[0] * shape[1] if isinstance(shape, (list, tuple)) and len(shape) >= 4 else 1
        grid_size = max(1, (seq_q + bm - 1) // bm) * bh
    elif kclass == "decode":
        split = chosen_config.get("SPLIT_KV", 4)
        bh = shape[0] * shape[1] if isinstance(shape, (list, tuple)) and len(shape) >= 4 else 1
        grid_size = bh * split
    else:
        grid_size = 1

    num_warps = chosen_config.get("num_warps", 4)
    threads = num_warps * prof.warp_size

    # Estimate smem and regs based on class
    if kclass == "tiled":
        bm = chosen_config.get("BLOCK_M", 64)
        bn = chosen_config.get("BLOCK_N", 64)
        st = chosen_config.get("num_stages", 2)
        head_dim = shape[-1] if isinstance(shape, (list, tuple)) and len(shape) >= 4 else 64
        elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
        smem_bytes = st * (bm * head_dim + 2 * bn * head_dim) * elem_bytes
        n_regs = 48
    elif kclass == "decode":
        smem_bytes = 1024
        n_regs = 32
    else:
        smem_bytes = 0
        n_regs = 32

    est_occ = programs_per_sm(
        profile=prof,
        n_regs=n_regs,
        smem_bytes=smem_bytes,
        num_warps=num_warps,
    )
    wave_count = waves(
        grid=grid_size,
        profile=prof,
        programs_per_sm=est_occ,
    )

    lines = [
        f"=== StackKernels Pick Explanation: '{op_name}' ===",
        f"Decision Source: {source}",
        f"Device Profile: {prof.name} (fingerprint: {prof.fingerprint()})",
        f"Triton Version: {triton_ver}, Kernel Revision: {kernel_rev}",
        f"Hardware Facts:",
        f"  - SM Count: {prof.sm_count}",
        f"  - Max Threads / SM: {prof.max_threads_per_sm}",
        f"  - Max Regs / SM: {prof.max_regs_per_sm}",
        f"  - Max Regs / Thread: {prof.max_regs_per_thread}",
        f"  - Max Smem / Block: {prof.max_smem_per_block} bytes",
        f"Occupancy & Wave Math:",
        f"  - Problem Shape: {shape}, Dtype: {dtype}",
        f"  - Grid Size: {grid_size} program blocks",
        f"  - Waves: {wave_count:.2f} waves across {prof.sm_count} SMs",
        f"  - Estimated Occupancy: {est_occ} programs/SM ({est_occ * threads} active threads/SM)",
        f"Selected Configuration:",
        f"  - {chosen_config}",
    ]
    if rejections:
        lines.append("Rejected Candidates:")
        for cand, errs in rejections[:5]:
            lines.append(f"  - Config {cand}: {'; '.join(errs)}")
    else:
        lines.append("Rejected Candidates: None")

    explanation_str = "\n".join(lines)
    return chosen_config, explanation_str


__all__ = ["pick"]
