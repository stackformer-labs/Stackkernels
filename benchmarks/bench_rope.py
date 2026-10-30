"""Benchmark Rotary Position Embedding (RoPE) achieved GB/s against peak."""

import torch
from benchmarks._common import run_op_benchmark, save_benchmark_results
from stackkernels.torch.embedding import apply_rope, precompute_freqs


def benchmark_rope():
    # Shapes formatted as (B, S, H, D)
    shapes = [
        (2, 512, 32, 128),
        (2, 1024, 32, 128),
        (2, 2048, 32, 128),
        (1, 4096, 32, 128),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    cos, sin = precompute_freqs(128, max_seq_len=8192, device=device, dtype=dtype)

    def run_fn(shape):
        q = torch.randn(shape, dtype=dtype, device=device)
        return apply_rope(q, cos=cos, sin=sin, start_pos=0)

    dtype_str = "float16" if dtype == torch.float16 else "float32"
    records = run_op_benchmark(
        op="rope",
        fn=run_fn,
        shapes=shapes,
        dtype_str=dtype_str,
    )
    save_benchmark_results("rope", records)


if __name__ == "__main__":
    benchmark_rope()
