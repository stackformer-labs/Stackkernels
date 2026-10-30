"""bytes moved, FLOPs, memory-bound or compute-bound"""

import math
from typing import Any

import torch


def get_dtype_bytes(dtype: Any) -> int:
    """Return number of bytes per element for a given dtype."""
    if isinstance(dtype, str):
        dt = dtype.lower()
        if "32" in dt:
            return 4
        elif "16" in dt or "bf16" in dt:
            return 2
        elif "64" in dt:
            return 8
        elif "8" in dt:
            return 1
        return 2

    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()

    return 2


def rms_norm_bytes_moved(
    shape: tuple[int, ...] | list[int],
    dtype: Any = torch.float16,
    mode: str = "fwd",
    has_weight: bool = True,
) -> int:
    """Calculate exact memory traffic in bytes for RMSNorm forward or backward pass.

    Args:
        shape: Tensor shape (*, D).
        dtype: Data type of tensor elements.
        mode: 'fwd' or 'bwd'.
        has_weight: Whether weight scaling parameter is used.

    Returns:
        Total bytes transferred across memory bus.
    """
    elem_size = get_dtype_bytes(dtype)
    rstd_size = 4  # fp32 reciprocal standard deviation

    total_elements = math.prod(shape)
    N = shape[-1]
    M = total_elements // N

    if mode == "fwd":
        # Read X (M*N), Write Y (M*N), Write rstd (M)
        bytes_x = M * N * elem_size
        bytes_y = M * N * elem_size
        bytes_rstd = M * rstd_size
        bytes_w = (N * elem_size) if has_weight else 0
        return bytes_x + bytes_y + bytes_rstd + bytes_w
    else:
        # Backward: Read dY, X, rstd, W (if present); Write dX, dW (if present)
        bytes_dy = M * N * elem_size
        bytes_x = M * N * elem_size
        bytes_rstd = M * rstd_size
        bytes_dx = M * N * elem_size
        bytes_w_read = (N * elem_size) if has_weight else 0
        bytes_dw_write = (N * elem_size) if has_weight else 0
        return bytes_dy + bytes_x + bytes_rstd + bytes_dx + bytes_w_read + bytes_dw_write


def calculate_achieved_gbps(bytes_moved: int, latency_ms: float) -> float:
    """Calculate achieved memory bandwidth in GB/s from bytes moved and latency."""
    if latency_ms <= 0.0:
        return 0.0
    # bytes / (ms * 1e-3) / 1e9 = bytes / (ms * 1e6)
    return (bytes_moved / (latency_ms * 1e6))


def calculate_memory_efficiency(achieved_gbps: float, peak_bandwidth_gbps: float) -> float:
    """Calculate percentage of peak hardware memory bandwidth achieved."""
    if peak_bandwidth_gbps <= 0.0:
        return 0.0
    return round((achieved_gbps / peak_bandwidth_gbps) * 100.0, 2)
