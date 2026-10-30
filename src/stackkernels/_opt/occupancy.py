"""Occupancy math, register estimation, and compiler probe adapter."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from stackkernels._opt.hardware import DeviceProfile, get_device_profile


@dataclass(frozen=True)
class ProbeResult:
    """Compiled kernel resource metrics."""

    n_regs: int
    n_spills: int
    smem_bytes: int
    programs_per_sm: int


def _safe_compile_probe(kernel: Any, args: tuple | list | None = None, kwargs: dict | None = None) -> dict[str, Any] | None:
    """Single adapter function to probe Triton compiler for registers and spills (Rule 5)."""
    if kernel is None:
        return None
    try:
        import triton

        args = tuple(args or ())
        kwargs = dict(kwargs or {})

        # Try kernel.warmup(*args, **kwargs)
        if hasattr(kernel, "warmup"):
            compiled = kernel.warmup(*args, **kwargs)
            n_regs = getattr(compiled, "n_regs", None)
            if n_regs is None and hasattr(compiled, "metadata"):
                n_regs = getattr(compiled.metadata, "num_regs", None)

            n_spills = getattr(compiled, "n_spills", None)
            if n_spills is None and hasattr(compiled, "metadata"):
                n_spills = getattr(compiled.metadata, "num_spills", 0)

            smem = getattr(compiled, "shared", None)
            if smem is None and hasattr(compiled, "metadata"):
                smem = getattr(compiled.metadata, "shared", 0)

            if n_regs is not None:
                return {
                    "n_regs": int(n_regs),
                    "n_spills": int(n_spills or 0),
                    "smem_bytes": int(smem or 0),
                }
    except Exception:
        pass
    return None


def programs_per_sm(
    profile: DeviceProfile,
    n_regs: int,
    smem_bytes: int,
    num_warps: int,
) -> int:
    """Calculate the number of concurrent programs (thread blocks) per SM."""
    if not profile.is_cuda:
        return 1

    threads_per_program = profile.warp_size * max(num_warps, 1)

    # 1. Thread limit per SM
    limit_threads = profile.max_threads_per_sm // max(threads_per_program, 1)

    # 2. Register limit per SM
    if n_regs > 0:
        regs_per_program = n_regs * threads_per_program
        limit_regs = profile.max_regs_per_sm // max(regs_per_program, 1)
    else:
        limit_regs = limit_threads

    # 3. Shared memory limit per SM
    if smem_bytes > 0:
        limit_smem = profile.max_smem_per_sm // smem_bytes
    else:
        limit_smem = limit_threads

    # 4. Hardware architectural block limit (16 for Turing, 32 for Ampere/Hopper)
    limit_blocks = 16 if profile.compute_capability < (8, 0) else 32

    return max(0, min(limit_threads, limit_regs, limit_smem, limit_blocks))


def waves(
    grid: int | Sequence[int],
    profile: DeviceProfile,
    programs_per_sm: int,
) -> int:
    """Calculate the number of waves needed to execute the full thread block grid."""
    if isinstance(grid, (list, tuple)):
        total_blocks = math.prod(grid)
    else:
        total_blocks = int(grid)

    capacity = profile.sm_count * max(programs_per_sm, 1)
    if capacity <= 0:
        return max(1, total_blocks)

    return math.ceil(total_blocks / capacity)


def accumulator_regs_per_thread(
    block_m: int,
    block_n: int,
    num_warps: int,
) -> int:
    """Estimate registers per thread dedicated to the accumulator tile in GEMM / tiled kernels."""
    threads = 32 * max(num_warps, 1)
    return math.ceil((block_m * block_n) / threads)


def probe(
    kernel: Any,
    args: tuple | list | None = None,
    kwargs: dict | None = None,
    profile: DeviceProfile | None = None,
    num_warps: int = 4,
) -> ProbeResult | None:
    """Probe compiler output and calculate occupancy."""
    res = _safe_compile_probe(kernel, args=args, kwargs=kwargs)
    if res is None:
        return None

    prof = profile or get_device_profile()
    n_regs = res["n_regs"]
    n_spills = res["n_spills"]
    smem_bytes = res["smem_bytes"]

    occ = programs_per_sm(prof, n_regs=n_regs, smem_bytes=smem_bytes, num_warps=num_warps)
    return ProbeResult(
        n_regs=n_regs,
        n_spills=n_spills,
        smem_bytes=smem_bytes,
        programs_per_sm=occ,
    )


__all__ = [
    "ProbeResult",
    "programs_per_sm",
    "waves",
    "accumulator_regs_per_thread",
    "probe",
]
