"""Heuristic configuration for Cross Entropy Loss."""

from typing import Any

from stackkernels._opt.classes.row import seed_configs
from stackkernels._opt.hardware import DeviceProfile, get_device_profile


def heuristic(shape: Any = None, profile: DeviceProfile | None = None, dtype: str = "float16", **kwargs: Any) -> dict[str, Any]:
    """Determine optimal block size and warps for Cross Entropy."""
    prof = profile or get_device_profile()
    cfg = dict(seed_configs(prof, problem=shape, dtype=dtype, **kwargs)[0])
    cfg["BLOCK_V"] = cfg["BLOCK_SIZE"]
    return cfg


__all__ = ["heuristic"]
