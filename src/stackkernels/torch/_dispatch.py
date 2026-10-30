"""chooses Triton or _reference"""

import os
import torch
from stackkernels._options import get_backend


def should_use_triton(tensor: torch.Tensor) -> bool:
    """Determine whether to use Triton kernel or eager PyTorch reference."""
    backend = get_backend()

    if backend == "reference":
        return False

    if backend == "triton":
        # Force Triton if requested
        return True

    # 'auto' backend: use Triton on CUDA GPUs, PyTorch reference on CPU
    if tensor.is_cuda:
        return True

    # If CPU but TRITON_INTERPRET mode is enabled and requested
    if os.environ.get("TRITON_INTERPRET") == "1" and backend == "triton":
        return True

    return False
