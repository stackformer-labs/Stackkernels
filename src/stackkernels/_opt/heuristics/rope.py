"""heuristic for rotary position embedding (RoPE)"""

from stackkernels._opt.heuristics._util import determine_num_warps, next_pow2


def heuristic(shape: int | tuple | None = None, **kwargs) -> dict:
    """Determine optimal block size and warps for RoPE."""
    head_dim = kwargs.get("head_dim")
    if head_dim is None:
        if isinstance(shape, (list, tuple)):
            head_dim = shape[-1]
        elif isinstance(shape, int):
            head_dim = shape
        else:
            head_dim = 128

    block_size = max(next_pow2(head_dim), 32)
    return {
        "BLOCK_SIZE": block_size,
        "num_warps": 4,
    }

