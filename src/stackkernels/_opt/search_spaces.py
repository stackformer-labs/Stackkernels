"""candidate configs per op, as plain data"""

from typing import Any
import triton


def get_rms_norm_search_space(n_cols: int) -> list[dict[str, Any]]:
    """Generate search space for RMSNorm tuning."""
    base_block = triton.next_power_of_2(n_cols)
    block_sizes = sorted(list({max(32, base_block // 2), base_block, min(8192, base_block * 2)}))
    warp_counts = [2, 4, 8, 16]

    configs = []
    for bs in block_sizes:
        for w in warp_counts:
            configs.append({"BLOCK_SIZE": bs, "num_warps": w})
    return configs


def get_search_space(op: str, shape: Any = None) -> list[dict[str, Any]]:
    """Retrieve candidate configuration search space for a given op."""
    op_name = op.strip().lower()
    if op_name in ("rms_norm", "rmsnorm", "layer_norm", "layernorm"):
        n_cols = 1024
        if isinstance(shape, (list, tuple)):
            n_cols = shape[-1]
        elif isinstance(shape, int):
            n_cols = shape
        return get_rms_norm_search_space(n_cols)

    # General default candidates
    return [
        {"BLOCK_SIZE": 256, "num_warps": 4},
        {"BLOCK_SIZE": 512, "num_warps": 4},
        {"BLOCK_SIZE": 1024, "num_warps": 8},
        {"BLOCK_SIZE": 2048, "num_warps": 8},
    ]
