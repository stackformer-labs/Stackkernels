"""heuristic for elementwise activations"""

from stackkernels._opt.heuristics._util import determine_num_warps, next_pow2


def heuristic(shape: int | tuple | None = None, **kwargs) -> dict:
    """Determine optimal block size and warps for elementwise kernels."""
    block_size = 1024
    return {
        "BLOCK_SIZE": block_size,
        "num_warps": determine_num_warps(block_size),
    }

