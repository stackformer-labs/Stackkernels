"""contiguity checks, reshape helpers"""

import torch


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous in memory."""
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten leading dimensions of tensor x into a 2D matrix (M, N).

    Returns:
        Flattened 2D tensor of shape (M, N), and original shape tuple.
    """
    orig_shape = x.shape
    x_contig = ensure_contiguous(x)
    if x.ndim == 2:
        return x_contig, orig_shape

    return x_contig.view(-1, orig_shape[-1]), orig_shape


def unflatten_from_2d(x_2d: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
    """Restore 2D tensor back to its original multidimensional shape."""
    if x_2d.shape == orig_shape:
        return x_2d
    return x_2d.view(orig_shape)
