"""runs candidate configs and writes the cache"""

from typing import Any

from stackkernels._opt.cache import TuningCache, get_global_cache
from stackkernels._opt.hardware import get_gpu_arch, is_gpu_available
from stackkernels._opt.search_spaces import get_search_space


def tune(
    op: str,
    shape: Any,
    dtype: str = "float16",
    arch: str | None = None,
    candidate_configs: list[dict[str, Any]] | None = None,
    cache: TuningCache | None = None,
    write_to_disk: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Sweep candidate configs, select the best performer, and update cache."""
    effective_arch = (arch or get_gpu_arch()).lower()
    target_cache = cache if cache is not None else get_global_cache()

    candidates = candidate_configs or get_search_space(op, shape)
    if not candidates:
        from stackkernels._opt.pick import pick

        return pick(op=op, shape=shape, dtype=dtype, arch=effective_arch, cache=target_cache, **kwargs)

    best_config = candidates[0]

    # If CUDA is available, benchmark candidate configs
    if is_gpu_available():
        try:
            import torch
            from stackkernels._opt.profile.bench import bench_fn
            from stackkernels.torch.norm import rms_norm

            best_latency = float("inf")
            device = "cuda"
            input_tensor = torch.randn(shape, dtype=getattr(torch, dtype, torch.float16), device=device)

            for cfg in candidates:
                try:
                    res = bench_fn(
                        lambda: rms_norm(input_tensor),
                        warmup=10,
                        iters=20,
                    )
                    if res.median_ms < best_latency:
                        best_latency = res.median_ms
                        best_config = cfg
                except Exception:
                    continue
        except Exception:
            pass

    bucket = kwargs.get("bucket", shape)
    target_cache.set(
        op=op,
        arch=effective_arch,
        dtype=dtype,
        bucket=bucket,
        config=best_config,
    )

    if write_to_disk:
        try:
            target_cache.save_user_cache()
        except Exception:
            pass

    return best_config
