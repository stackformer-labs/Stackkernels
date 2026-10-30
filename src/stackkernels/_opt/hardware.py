"""Device-aware hardware profiles, query pipeline, and specs."""

import os
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    """Device specifications used to guide optimization decisions."""

    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    warp_size: int
    max_regs_per_sm: int
    max_regs_per_thread: int
    max_threads_per_sm: int
    max_smem_per_block: int
    max_smem_per_sm: int
    l2_bytes: int
    total_memory_bytes: int
    peak_bandwidth_gbps: float | None = None
    peak_tflops: dict[str, float] | None = None
    has_cp_async: bool = False
    has_bf16: bool = False

    def fingerprint(self) -> str:
        """Return a stable normalized fingerprint string for cache keys."""
        if self.compute_capability == (0, 0):
            return "cpu"
        clean_name = self.name.lower().replace(" ", "-").replace("/", "-")
        maj, min_ = self.compute_capability
        mem_mb = self.total_memory_bytes // (1024 * 1024)
        return f"{clean_name}_sm{maj}{min_}_{self.sm_count}sm_{mem_mb}mb"

    @property
    def is_cuda(self) -> bool:
        return self.compute_capability != (0, 0)


@dataclass(frozen=True)
class HardwareInfo:
    """Legacy hardware info preserved for backwards compatibility."""

    name: str = "cpu"
    arch: str = "cpu"
    sm_count: int = 0
    shared_memory_per_sm: int = 0
    peak_bandwidth_gbps: float = 0.0
    peak_flops_fp16: float = 0.0
    has_gpu: bool = False


# Fallback specification table keyed strictly by compute capability (major, minor)
_FALLBACK_TABLE: dict[tuple[int, int], dict[str, Any]] = {
    # Turing (e.g. Tesla T4)
    (7, 5): {
        "name": "NVIDIA Turing GPU",
        "sm_count": 40,
        "warp_size": 32,
        "max_regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "max_threads_per_sm": 1024,
        "max_smem_per_block": 49152,
        "max_smem_per_sm": 65536,
        "l2_bytes": 4 * 1024 * 1024,
        "total_memory_bytes": 16 * 1024 * 1024 * 1024,
        "peak_bandwidth_gbps": 320.0,
        "peak_tflops": {"fp16": 65.0, "fp32": 8.1},
        "has_cp_async": False,
        "has_bf16": False,
    },
    # Ampere (e.g. A100-SXM4)
    (8, 0): {
        "name": "NVIDIA A100-SXM4",
        "sm_count": 108,
        "warp_size": 32,
        "max_regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "max_threads_per_sm": 2048,
        "max_smem_per_block": 167936,
        "max_smem_per_sm": 167936,
        "l2_bytes": 40 * 1024 * 1024,
        "total_memory_bytes": 40 * 1024 * 1024 * 1024,
        "peak_bandwidth_gbps": 1555.0,
        "peak_tflops": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
        "has_cp_async": True,
        "has_bf16": True,
    },
    # Hopper (e.g. H100-SXM5)
    (9, 0): {
        "name": "NVIDIA H100-SXM5",
        "sm_count": 132,
        "warp_size": 32,
        "max_regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "max_threads_per_sm": 2048,
        "max_smem_per_block": 232448,
        "max_smem_per_sm": 232448,
        "l2_bytes": 50 * 1024 * 1024,
        "total_memory_bytes": 80 * 1024 * 1024 * 1024,
        "peak_bandwidth_gbps": 3350.0,
        "peak_tflops": {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0},
        "has_cp_async": True,
        "has_bf16": True,
    },
}

_DEVICE_PROFILE_CACHE: dict[Any, DeviceProfile] = {}


def _safe_query_triton_driver_properties(device_idx: int) -> dict[str, Any] | None:
    """Safe adapter to query Triton active driver device properties (Rule 5)."""
    try:
        import triton

        driver = getattr(triton.runtime, "driver", None)
        if driver is not None:
            active = getattr(driver, "active", None)
            if active is not None and hasattr(active, "utils"):
                utils = active.utils
                if hasattr(utils, "get_device_properties"):
                    props = utils.get_device_properties(device_idx)
                    if isinstance(props, dict):
                        return props
    except Exception:
        pass
    return None


def is_gpu_available() -> bool:
    """Check dynamically if a CUDA GPU device is available."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def get_gpu_arch(device_id: int = 0) -> str:
    """Return active GPU architecture string ('sm_80', 'sm_75', etc.) or 'cpu'."""
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


def _build_cpu_profile() -> DeviceProfile:
    """Build a clearly marked CPU profile."""
    cpu_cores = os.cpu_count() or 1
    cpu_name = platform.processor() or platform.machine() or "Host CPU"
    return DeviceProfile(
        name=f"CPU ({cpu_name})",
        compute_capability=(0, 0),
        sm_count=cpu_cores,
        warp_size=32,
        max_regs_per_sm=0,
        max_regs_per_thread=0,
        max_threads_per_sm=1,
        max_smem_per_block=0,
        max_smem_per_sm=0,
        l2_bytes=0,
        total_memory_bytes=0,
        peak_bandwidth_gbps=None,
        peak_tflops=None,
        has_cp_async=False,
        has_bf16=False,
    )


def get_device_profile(device: Any = None) -> DeviceProfile:
    """Get the DeviceProfile for a device with multi-tiered resolution:

    1. torch.cuda.get_device_properties
    2. Triton driver active device properties
    3. Fallback table keyed by compute capability
    4. Conservative defaults
    5. Clean CPU profile if no GPU.
    """
    cache_key = str(device) if device is not None else "default"
    if cache_key in _DEVICE_PROFILE_CACHE:
        return _DEVICE_PROFILE_CACHE[cache_key]

    env_arch = os.environ.get("STACKKERNELS_GPU_ARCH", "").lower().strip()
    if env_arch == "cpu" or not is_gpu_available():
        profile = _build_cpu_profile()
        _DEVICE_PROFILE_CACHE[cache_key] = profile
        return profile

    try:
        import torch

        if isinstance(device, int):
            device_idx = device
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                profile = _build_cpu_profile()
                _DEVICE_PROFILE_CACHE[cache_key] = profile
                return profile
            device_idx = device.index or 0
        elif isinstance(device, str):
            if device == "cpu":
                profile = _build_cpu_profile()
                _DEVICE_PROFILE_CACHE[cache_key] = profile
                return profile
            device_idx = int(device.replace("cuda:", "").replace("cuda", "0") or 0)
        else:
            device_idx = torch.cuda.current_device()

        # Tier 1: PyTorch device properties
        props = torch.cuda.get_device_properties(device_idx)
        cc = torch.cuda.get_device_capability(device_idx)
        major, minor = cc
        name = getattr(props, "name", f"NVIDIA GPU sm_{major}{minor}")
        sm_count = getattr(props, "multi_processor_count", 0)
        total_mem = getattr(props, "total_memory", 0)
        max_threads = getattr(props, "max_threads_per_multi_processor", 1024)

        # Tier 2: Query Triton driver device properties if available
        triton_props = _safe_query_triton_driver_properties(device_idx) or {}
        max_smem_sm = (
            triton_props.get("max_shared_mem")
            or getattr(props, "max_shared_memory_per_multiprocessor", 0)
        )
        max_regs_sm = triton_props.get("max_num_regs", 65536)

        # Tier 3: Compute capability fallback table
        fallback = _FALLBACK_TABLE.get((major, minor), {})

        # Merge properties, preferring runtime queried values, then fallback, then defaults
        final_sm_count = sm_count or fallback.get("sm_count", 32)
        final_warp_size = triton_props.get("warp_size") or fallback.get("warp_size", 32)
        final_max_regs_sm = max_regs_sm or fallback.get("max_regs_per_sm", 65536)
        final_max_regs_th = triton_props.get("max_regs_per_thread") or fallback.get("max_regs_per_thread", 255)
        final_max_threads = max_threads or fallback.get("max_threads_per_sm", 1024)
        final_max_smem_block = triton_props.get("max_shared_mem_per_block") or fallback.get(
            "max_smem_per_block", 49152
        )
        final_max_smem_sm = max_smem_sm or fallback.get("max_smem_per_sm", 65536)
        final_l2 = triton_props.get("l2_cache_size") or fallback.get("l2_bytes", 4 * 1024 * 1024)
        final_mem = total_mem or fallback.get("total_memory_bytes", 8 * 1024 * 1024 * 1024)
        peak_bw = fallback.get("peak_bandwidth_gbps")
        peak_tflops = fallback.get("peak_tflops")

        has_cp_async = major >= 8
        has_bf16 = (major, minor) >= (8, 0)

        profile = DeviceProfile(
            name=name,
            compute_capability=(major, minor),
            sm_count=final_sm_count,
            warp_size=final_warp_size,
            max_regs_per_sm=final_max_regs_sm,
            max_regs_per_thread=final_max_regs_th,
            max_threads_per_sm=final_max_threads,
            max_smem_per_block=final_max_smem_block,
            max_smem_per_sm=final_max_smem_sm,
            l2_bytes=final_l2,
            total_memory_bytes=final_mem,
            peak_bandwidth_gbps=peak_bw,
            peak_tflops=peak_tflops,
            has_cp_async=has_cp_async,
            has_bf16=has_bf16,
        )
    except Exception:
        profile = _build_cpu_profile()

    _DEVICE_PROFILE_CACHE[cache_key] = profile
    return profile


def get_hardware_info(device_id: int = 0, arch: str | None = None) -> HardwareInfo:
    """Legacy HardwareInfo function for backwards compatibility."""
    profile = get_device_profile(device_id)
    if not profile.is_cuda:
        return HardwareInfo(
            name=profile.name,
            arch="cpu",
            sm_count=profile.sm_count,
            shared_memory_per_sm=0,
            peak_bandwidth_gbps=0.0,
            peak_flops_fp16=0.0,
            has_gpu=False,
        )

    maj, min_ = profile.compute_capability
    return HardwareInfo(
        name=profile.name,
        arch=f"sm_{maj}{min_}",
        sm_count=profile.sm_count,
        shared_memory_per_sm=profile.max_smem_per_sm,
        peak_bandwidth_gbps=profile.peak_bandwidth_gbps or 0.0,
        peak_flops_fp16=(profile.peak_tflops or {}).get("fp16", 0.0),
        has_gpu=True,
    )


__all__ = [
    "DeviceProfile",
    "HardwareInfo",
    "get_device_profile",
    "get_hardware_info",
    "get_gpu_arch",
    "is_gpu_available",
]
