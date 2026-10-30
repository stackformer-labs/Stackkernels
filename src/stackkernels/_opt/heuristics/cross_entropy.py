"""heuristic for cross entropy loss"""

from stackkernels._opt.heuristics._util import determine_num_warps, next_pow2


def heuristic(shape: int | tuple | None = None, **kwargs) -> dict:
    """Determine optimal block size and warps for Cross Entropy."""
    vocab_size = kwargs.get("vocab_size")
    if vocab_size is None:
        if isinstance(shape, (list, tuple)):
            vocab_size = shape[-1]
        elif isinstance(shape, int):
            vocab_size = shape
        else:
            vocab_size = 32000

    block_size = min(max(next_pow2(vocab_size), 64), 8192)
    return {
        "BLOCK_SIZE": block_size,
        "num_warps": determine_num_warps(block_size),
    }

