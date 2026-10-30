"""detect GPU arch, SM count, shared memory, peak bandwidth and FLOPs"""

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    """Hardware specifications detected dynamically from device or host."""

    name: str = "cpu"
    arch: str = "cpu"
    sm_count: int = 0
    shared_memory_per_sm: int = 0  # bytes
    peak_bandwidth_gbps: float = 0.0  # GB/s
    peak_flops_fp16: float = 0.0  # TFLOPs
    has_gpu: bool = False


def is_gpu_available() -> bool:
    """Check dynamically if a CUDA GPU device is available."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_gpu_arch(device_id: int = 0) -> str:
    """Return the active GPU architecture (e.g. 'sm_80', 'sm_90') or 'cpu'."""
    env_arch = os.environ.get("STACKKERNELS_GPU_ARCH")
    if env_arch:
        return env_arch.lower().strip()

    if not is_gpu_available():
        return "cpu"

    try:
        import torch

        major, minor = torch.cuda.get_device_capability(device_id)
        return f"sm_{major}{minor}"
    except Exception:
        return "cpu"


def get_hardware_info(device_id: int = 0, arch: str | None = None) -> HardwareInfo:
    """Dynamically detect hardware properties from runtime or system.

    Works cleanly on CPU without hardcoding device specifications.
    """
    target_arch = (arch or os.environ.get("STACKKERNELS_GPU_ARCH") or "").lower().strip()

    # If GPU is available and no forced CPU arch requested
    if is_gpu_available() and target_arch != "cpu":
        try:
            import torch

            props = torch.cuda.get_device_properties(device_id)
            major, minor = torch.cuda.get_device_capability(device_id)
            effective_arch = target_arch if target_arch else f"sm_{major}{minor}"

            # Calculate memory bandwidth dynamically from clock rate & bus width if reported
            bandwidth_gbps = 0.0
            clock_rate_khz = getattr(props, "memory_clock_rate", 0)
            bus_width_bits = getattr(props, "memory_bus_width", 0)
            if clock_rate_khz > 0 and bus_width_bits > 0:
                # 2 for DDR * clock * bus_width / 8 bytes per bit / 1e9 for GB/s
                bandwidth_gbps = round(2.0 * clock_rate_khz * 1e3 * (bus_width_bits / 8.0) / 1e9, 2)

            return HardwareInfo(
                name=getattr(props, "name", "NVIDIA GPU"),
                arch=effective_arch,
                sm_count=getattr(props, "multi_processor_count", 0),
                shared_memory_per_sm=getattr(
                    props, "max_shared_memory_per_multiprocessor", 0
                ),
                peak_bandwidth_gbps=bandwidth_gbps,
                peak_flops_fp16=0.0,
                has_gpu=True,
            )
        except Exception:
            pass

    # CPU dynamic fallback
    cpu_cores = os.cpu_count() or 1
    cpu_arch = platform.machine().lower() or "cpu"
    cpu_name = platform.processor() or platform.machine() or "Host CPU"

    return HardwareInfo(
        name=cpu_name,
        arch=target_arch if target_arch else cpu_arch,
        sm_count=cpu_cores,
        shared_memory_per_sm=0,
        peak_bandwidth_gbps=0.0,
        peak_flops_fp16=0.0,
        has_gpu=False,
    )
