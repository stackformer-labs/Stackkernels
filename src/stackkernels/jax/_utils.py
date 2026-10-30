"""Shape and array helpers for JAX backend."""

from typing import Any


def flatten_to_2d(x: Any) -> tuple[Any, tuple[int, ...]]:
    """Flatten leading batch dimensions into a 2D matrix (M, N)."""
    orig_shape = x.shape
    if len(orig_shape) == 2:
        return x, orig_shape
    return x.reshape(-1, orig_shape[-1]), orig_shape


def unflatten_from_2d(x_2d: Any, orig_shape: tuple[int, ...]) -> Any:
    """Restore 2D array to original multidimensional shape."""
    if x_2d.shape == orig_shape:
        return x_2d
    return x_2d.reshape(orig_shape)
