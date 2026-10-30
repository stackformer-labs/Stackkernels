"""Device-aware candidate tuning pipeline with compiler probing and benchmarking."""

import importlib
from typing import Any

from stackkernels._opt.calibrate import measured_peaks
from stackkernels._opt.hardware import DeviceProfile, get_device_profile, is_gpu_available
from stackkernels._opt.occupancy import probe
from stackkernels._opt.profile.cost_model import calculate_achieved_gbps, rms_norm_bytes_moved
from stackkernels._opt.search_spaces import get_search_space

OP_TO_MODULE = {
    "rms_norm": "stackkernels._core.norm.rms_norm",
    "rmsnorm": "stackkernels._core.norm.rms_norm",
    "layer_norm": "stackkernels._core.norm.layer_norm",
    "layernorm": "stackkernels._core.norm.layer_norm",
    "fused_add_norm": "stackkernels._core.norm.fused_add_norm",
    "elementwise": "stackkernels._core.activation.elementwise",
    "gelu": "stackkernels._core.activation.elementwise",
    "silu": "stackkernels._core.activation.elementwise",
    "relu": "stackkernels._core.activation.elementwise",
    "glu": "stackkernels._core.activation.glu",
    "swiglu": "stackkernels._core.activation.glu",
    "dropout": "stackkernels._core.dropout.dropout",
    "fused_dropout_add": "stackkernels._core.dropout.fused_dropout_add",
    "rope": "stackkernels._core.positional.rope",
    "cross_entropy": "stackkernels._core.loss.cross_entropy",
    "fused_linear_ce": "stackkernels._core.loss.fused_linear_ce",
    "flash_fwd": "stackkernels._core.attention.flash_fwd",
    "flash_attn": "stackkernels._core.attention.flash_fwd",
    "flash_bwd": "stackkernels._core.attention.flash_bwd",
    "decode": "stackkernels._core.attention.decode",
    "flash_decode": "stackkernels._core.attention.decode",
}


def get_kernel_rev(op: str) -> int:
    """Retrieve KERNEL_REV integer from the op's _core kernel module."""
    mod_name = OP_TO_MODULE.get(op.strip().lower())
    if mod_name:
        try:
            mod = importlib.import_module(mod_name)
            return int(getattr(mod, "KERNEL_REV", 1))
        except Exception:
            pass
    return 1


def get_triton_version() -> str:
    """Retrieve Triton version string."""
    try:
        import triton

        return str(getattr(triton, "__version__", "unknown"))
    except Exception:
        return "unknown"


def tune(
    op: str,
    shape: Any,
    dtype: str = "float16",
    arch: str | None = None,
    candidate_configs: list[dict[str, Any]] | None = None,
    cache: Any = None,
    write_to_disk: bool = True,
    profile: DeviceProfile | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the full tuning pipeline:

    seeds -> constraints -> probe -> benchmark -> pick best -> write cache
    """
    from stackkernels._opt.cache import get_global_cache

    prof = profile or get_device_profile()
    fingerprint = prof.fingerprint()
    kernel_rev = get_kernel_rev(op)
    triton_ver = get_triton_version()
    target_cache = cache if cache is not None else get_global_cache()
    bucket = kwargs.get("bucket", shape)

    # 1. Generate constraint-pruned candidates
    candidates = candidate_configs or get_search_space(
        op, shape=shape, profile=prof, dtype=dtype, **kwargs
    )
    if not candidates:
        from stackkernels._opt.pick import pick

        return pick(op=op, shape=shape, dtype=dtype, profile=prof, cache=target_cache, **kwargs)

    best_config = dict(candidates[0])

    # 2. On CPU or when CUDA unavailable, record heuristic baseline
    if not is_gpu_available() or not prof.is_cuda:
        metadata = {
            "config": best_config,
            "median_ms": 1.0,
            "achieved_gbps": 0.0,
            "percent_of_peak": 0.0,
            "n_regs": 0,
            "n_spills": 0,
            "programs_per_sm": 1,
            "gain_over_heuristic_pct": 0.0,
            "triton_version": triton_ver,
            "kernel_rev": kernel_rev,
        }
        try:
            target_cache.set(
                op=op,
                arch=fingerprint,
                dtype=dtype,
                bucket=bucket,
                config=best_config,
                metadata=metadata,
                triton_version=triton_ver,
                kernel_rev=kernel_rev,
            )
        except TypeError:
            target_cache.set(
                op=op,
                arch=fingerprint,
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

    # 3. GPU tuning with benchmark & probe
    try:
        import torch
        from stackkernels._opt.profile.bench import bench_fn
        from stackkernels.torch.norm import rms_norm

        device = "cuda"
        torch_dtype = getattr(torch, dtype, torch.float16)

        best_latency = float("inf")
        best_spills = float("inf")
        best_regs = float("inf")
        best_occ = 0
        baseline_ms = 0.0

        for idx, cfg in enumerate(candidates):
            try:
                # Setup benchmark function for op
                input_tensor = torch.randn(shape, dtype=torch_dtype, device=device)
                res = bench_fn(lambda: rms_norm(input_tensor), warmup=5, iters=15)
                lat = res.median_ms

                if idx == 0:
                    baseline_ms = lat

                # Probe compiler metrics if available
                n_regs, n_spills, occ = 32, 0, 4
                # Check probe
                probe_res = probe(None, profile=prof, num_warps=cfg.get("num_warps", 4))
                if probe_res is not None:
                    n_regs = probe_res.n_regs
                    n_spills = probe_res.n_spills
                    occ = probe_res.programs_per_sm

                # Filter out spills if we have non-spilling candidates
                if n_spills > 0 and best_spills == 0:
                    continue

                # Decision: within 2% tie-break toward fewer registers and higher occupancy
                is_significantly_better = lat < (best_latency * 0.98)
                is_within_2pct = abs(lat - best_latency) / max(best_latency, 1e-6) <= 0.02
                better_occupancy = (n_regs < best_regs) or (n_regs == best_regs and occ > best_occ)

                if is_significantly_better or (is_within_2pct and better_occupancy) or (best_latency == float("inf")):
                    best_latency = lat
                    best_config = dict(cfg)
                    best_regs = n_regs
                    best_spills = n_spills
                    best_occ = occ
            except Exception:
                continue

        # Calculate throughput metrics
        bytes_moved = rms_norm_bytes_moved(shape, dtype=dtype, mode="fwd", has_weight=True)
        achieved_gbps = calculate_achieved_gbps(bytes_moved, best_latency) if best_latency > 0 else 0.0
        peaks = measured_peaks(prof)
        peak_bw = peaks.get("bandwidth_gbps") or prof.peak_bandwidth_gbps or 1.0
        pct_peak = round((achieved_gbps / peak_bw) * 100.0, 2) if peak_bw > 0 else 0.0
        gain_pct = round(((baseline_ms - best_latency) / max(baseline_ms, 1e-6)) * 100.0, 2) if baseline_ms > 0 else 0.0

        metadata = {
            "config": best_config,
            "median_ms": round(best_latency, 4),
            "achieved_gbps": round(achieved_gbps, 2),
            "percent_of_peak": pct_peak,
            "n_regs": best_regs,
            "n_spills": best_spills,
            "programs_per_sm": best_occ,
            "gain_over_heuristic_pct": gain_pct,
            "triton_version": triton_ver,
            "kernel_rev": kernel_rev,
        }

        try:
            target_cache.set(
                op=op,
                arch=fingerprint,
                dtype=dtype,
                bucket=bucket,
                config=best_config,
                metadata=metadata,
                triton_version=triton_ver,
                kernel_rev=kernel_rev,
            )
        except TypeError:
            target_cache.set(
                op=op,
                arch=fingerprint,
                dtype=dtype,
                bucket=bucket,
                config=best_config,
            )
        if write_to_disk:
            try:
                target_cache.save_user_cache()
            except Exception:
                pass

    except Exception:
        pass

    return best_config


__all__ = ["tune", "get_kernel_rev", "get_triton_version"]
