"""Heuristic for fused linear cross entropy."""

from stackkernels._opt.heuristics._util import determine_num_warps, next_pow2


def heuristic(shape: int | tuple | None = None, **kwargs) -> dict:
    """Determine optimal chunk size, block size, and warps for Fused Linear Cross Entropy."""
    vocab_size = kwargs.get("vocab_size")
    if vocab_size is None:
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            vocab_size = shape[-1]
        elif isinstance(shape, int):
            vocab_size = shape
        else:
            vocab_size = 32000

    # Determine chunk size for M to keep chunk logits <= 128MB
    # 128MB / (vocab_size * 2 bytes) = max tokens in chunk
    chunk_size = max(min(134217728 // (max(vocab_size, 1) * 2), 2048), 128)
    chunk_size = 1 << (chunk_size.bit_length() - 1)
    chunk_size = max(chunk_size, 128)

    block_v = min(max(next_pow2(vocab_size), 64), 2048)
    return {
        "CHUNK_SIZE": chunk_size,
        "BLOCK_V": block_v,
        "num_warps": determine_num_warps(block_v),
    }


__all__ = ["heuristic"]
