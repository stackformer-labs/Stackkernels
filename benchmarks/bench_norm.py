"""Benchmark RMSNorm achieved GB/s against hardware peak."""

import torch
from benchmarks._common import run_op_benchmark, save_benchmark_results
from stackkernels.torch.norm import rms_norm


def benchmark_rms_norm():
    shapes = [
        (2, 512, 1024),
        (2, 1024, 2048),
        (2, 2048, 4096),
        (1, 4096, 8192),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def run_fn(shape):
        x = torch.randn(shape, dtype=dtype, device=device)
        w = torch.ones(shape[-1], dtype=dtype, device=device)
        return rms_norm(x, w)

    dtype_str = "float16" if dtype == torch.float16 else "float32"
    records = run_op_benchmark(
        op="rms_norm",
        fn=run_fn,
        shapes=shapes,
        dtype_str=dtype_str,
    )
    save_benchmark_results("rms_norm", records)


if __name__ == "__main__":
    benchmark_rms_norm()
