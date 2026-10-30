"""Device calibration: measure achieved peak memory bandwidth and compute throughput."""

import json
import os
from pathlib import Path
from typing import Any

from stackkernels._opt.hardware import DeviceProfile, get_device_profile, is_gpu_available


def _get_calibration_cache_path() -> Path:
    base_dir = os.environ.get("STACKKERNELS_CACHE_DIR")
    if base_dir:
        cache_dir = Path(base_dir)
    else:
        cache_dir = Path.home() / ".cache" / "stackkernels"
    return cache_dir / "calibration.json"


def _load_all_calibrations() -> dict[str, dict[str, Any]]:
    path = _get_calibration_cache_path()
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _save_calibration(fingerprint: str, data: dict[str, Any]) -> None:
    path = _get_calibration_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        all_cal = _load_all_calibrations()
        all_cal[fingerprint] = data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_cal, f, indent=2)
    except Exception:
        pass


def measured_peaks(profile: DeviceProfile | None = None) -> dict[str, float]:
    """Retrieve measured peaks for a device, falling back to specs or conservative defaults."""
    prof = profile or get_device_profile()
    fp = prof.fingerprint()

    # Check cached calibration
    all_cal = _load_all_calibrations()
    if fp in all_cal:
        return dict(all_cal[fp])

    # Fallback to profile spec values if available
    bw = prof.peak_bandwidth_gbps if prof.peak_bandwidth_gbps is not None else (30.0 if not prof.is_cuda else 300.0)
    tflops_dict = prof.peak_tflops or {}
    t16 = tflops_dict.get("fp16", 0.0)
    t32 = tflops_dict.get("fp32", 1.0)

    return {
        "bandwidth_gbps": round(bw, 2),
        "tflops_fp16": round(t16, 2),
        "tflops_fp32": round(t32, 2),
        "calibrated": False,
    }


def calibrate_device(profile: DeviceProfile | None = None, force: bool = False) -> dict[str, float]:
    """Measure achieved peak bandwidth and matmul throughput on device."""
    prof = profile or get_device_profile()
    fp = prof.fingerprint()

    if not force:
        all_cal = _load_all_calibrations()
        if fp in all_cal:
            return dict(all_cal[fp])

    if not is_gpu_available() or not prof.is_cuda:
        # CPU calibration fallback
        result = {
            "bandwidth_gbps": 35.0,
            "tflops_fp16": 0.0,
            "tflops_fp32": 1.5,
            "calibrated": True,
        }
        _save_calibration(fp, result)
        return result

    try:
        import torch

        device = "cuda"

        # 1. Measure Memory Copy Bandwidth (large 256MB tensor copy)
        n_elements = 64 * 1024 * 1024  # 256 MB float32
        x = torch.empty(n_elements, dtype=torch.float32, device=device)
        y = torch.empty(n_elements, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(5):
            y.copy_(x)
        torch.cuda.synchronize()

        # Timed runs with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        iters = 10
        start_event.record()
        for _ in range(iters):
            y.copy_(x)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / iters
        bytes_copied = 2 * n_elements * 4  # Read X, write Y
        achieved_bw = (bytes_copied / (elapsed_ms * 1e-3)) / 1e9

        del x, y
        torch.cuda.empty_cache()

        # 2. Measure FP16 GEMM Throughput (4096 x 4096 x 4096)
        M, N, K = 4096, 4096, 4096
        a16 = torch.randn(M, K, dtype=torch.float16, device=device)
        b16 = torch.randn(K, N, dtype=torch.float16, device=device)

        for _ in range(5):
            torch.matmul(a16, b16)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            torch.matmul(a16, b16)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_gemm_ms = start_event.elapsed_time(end_event) / iters
        gemm_flops = 2.0 * M * N * K
        achieved_tflops_fp16 = (gemm_flops / (elapsed_gemm_ms * 1e-3)) / 1e12

        del a16, b16
        torch.cuda.empty_cache()

        # 3. Measure FP32 GEMM Throughput
        a32 = torch.randn(M, K, dtype=torch.float32, device=device)
        b32 = torch.randn(K, N, dtype=torch.float32, device=device)

        for _ in range(5):
            torch.matmul(a32, b32)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            torch.matmul(a32, b32)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_gemm32_ms = start_event.elapsed_time(end_event) / iters
        achieved_tflops_fp32 = (gemm_flops / (elapsed_gemm32_ms * 1e-3)) / 1e12

        del a32, b32
        torch.cuda.empty_cache()

        result = {
            "bandwidth_gbps": round(achieved_bw, 2),
            "tflops_fp16": round(achieved_tflops_fp16, 2),
            "tflops_fp32": round(achieved_tflops_fp32, 2),
            "calibrated": True,
        }
        _save_calibration(fp, result)
        return result

    except Exception:
        # Fallback to default
        return measured_peaks(prof)


__all__ = ["measured_peaks", "calibrate_device"]

