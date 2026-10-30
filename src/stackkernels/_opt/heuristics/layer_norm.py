"""heuristic for layer_norm"""

from stackkernels._opt.heuristics._util import determine_num_warps, next_pow2


def heuristic(shape: int | tuple | None = None, **kwargs) -> dict:
    """Determine optimal block size and warps for LayerNorm."""
    n_cols = kwargs.get("n_cols")
    if n_cols is None:
        if isinstance(shape, (list, tuple)):
            n_cols = shape[-1]
        elif isinstance(shape, int):
            n_cols = shape
        else:
            n_cols = 1024

    block_size = min(max(next_pow2(n_cols), 32), 8192)
    return {
        "BLOCK_SIZE": block_size,
        "num_warps": determine_num_warps(block_size, max_warps=8),
    }