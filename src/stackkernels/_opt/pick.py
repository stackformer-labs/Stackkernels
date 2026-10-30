"""resolution order: user override, tuned cache, static heuristic"""

from typing import Any

from stackkernels._opt.cache import TuningCache, get_global_cache
from stackkernels._opt.hardware import get_gpu_arch
from stackkernels._opt.heuristics import get_heuristic


def pick(
    op: str,
    shape: Any = None,
    dtype: str = "float16",
    arch: str | None = None,
    override: dict[str, Any] | None = None,
    cache: TuningCache | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pick kernel execution configuration following the resolution order:

    1. User Override (highest precedence)
    2. Tuned Cache (shipped or user-saved)
    3. Static Heuristic (looked up by op name)
    4. Fallback default
    """
    op_name = str(op).strip().lower()

    # 1. User override
    if override:
        return dict(override)

    # Resolve GPU architecture
    effective_arch = (arch or get_gpu_arch()).lower()
    target_cache = cache if cache is not None else get_global_cache()

    # 2. Tuned cache lookup
    bucket = kwargs.get("bucket", shape)
    cached_config = target_cache.get(
        op=op_name,
        arch=effective_arch,
        dtype=dtype,
        bucket=bucket,
    )
    if cached_config is not None:
        return dict(cached_config)

    # 3. Static heuristic lookup by op name
    heuristic_fn = get_heuristic(op_name)
    if heuristic_fn is not None:
        return heuristic_fn(
            shape=shape,
            dtype=dtype,
            arch=effective_arch,
            **kwargs,
        )

    # 4. Fallback safe default
    return {
        "BLOCK_SIZE": 1024,
        "num_warps": 4,
    }
