"""Heuristic configuration for Fused Linear Cross Entropy."""

from typing import Any

from stackkernels._opt.classes.tiled import seed_configs
from stackkernels._opt.hardware import DeviceProfile, get_device_profile


def heuristic(shape: Any = None, profile: DeviceProfile | None = None, dtype: str = "float16", **kwargs: Any) -> dict[str, Any]:
    """Determine optimal chunk size, block size, and warps for Fused Linear Cross Entropy."""
    prof = profile or get_device_profile()
    vocab_size = kwargs.get("vocab_size")
    if vocab_size is None:
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            vocab_size = shape[-1]
        elif isinstance(shape, int):
            vocab_size = shape
        else:
            vocab_size = 32000

    chunk_size = max(min(134217728 // (max(vocab_size, 1) * 2), 2048), 128)
    chunk_size = 1 << (chunk_size.bit_length() - 1)
    chunk_size = max(chunk_size, 128)

    cfg = dict(seed_configs(prof, problem=shape, dtype=dtype, **kwargs)[0])
    cfg["CHUNK_SIZE"] = chunk_size
    cfg["BLOCK_V"] = min(max(vocab_size, 64), 2048)
    return cfg


__all__ = ["heuristic"]
