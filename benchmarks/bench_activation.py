"""Benchmark activation functions (SwiGLU, SiLU, GELU) achieved GB/s against peak."""

import torch
from benchmarks._common import run_op_benchmark, save_benchmark_results
from stackkernels.torch.activation import gelu, silu, swiglu


def benchmark_activations():
    shapes = [
        (2, 512, 4096),
        (2, 1024, 4096),
        (2, 2048, 4096),
        (1, 4096, 8192),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Benchmark SwiGLU
    def run_swiglu(shape):
        gate = torch.randn(shape, dtype=dtype, device=device)
        up = torch.randn(shape, dtype=dtype, device=device)
        return swiglu(gate, up)

    dtype_str = "float16" if dtype == torch.float16 else "float32"
    records_swiglu = run_op_benchmark(
        op="swiglu",
        fn=run_swiglu,
        shapes=shapes,
        dtype_str=dtype_str,
    )
    save_benchmark_results("swiglu", records_swiglu)

    # Benchmark SiLU
    def run_silu(shape):
        x = torch.randn(shape, dtype=dtype, device=device)
        return silu(x)

    records_silu = run_op_benchmark(
        op="silu",
        fn=run_silu,
        shapes=shapes,
        dtype_str=dtype_str,
    )
    save_benchmark_results("silu", records_silu)


if __name__ == "__main__":
    benchmark_activations()

