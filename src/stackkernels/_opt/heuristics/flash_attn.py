"""Heuristic configuration for Flash Attention (prefill forward & backward)."""

from typing import Any

from stackkernels._opt.classes.tiled import seed_configs
from stackkernels._opt.hardware import DeviceProfile, get_device_profile


def heuristic(shape: Any = None, profile: DeviceProfile | None = None, dtype: str = "float16", **kwargs: Any) -> dict[str, Any]:
    """Determine optimal block sizes, warps, and stages for Flash Attention."""
    prof = profile or get_device_profile()
    return dict(seed_configs(prof, problem=shape, dtype=dtype, **kwargs)[0])


__all__ = ["heuristic"]
