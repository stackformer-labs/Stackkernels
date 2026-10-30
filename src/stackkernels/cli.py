"""command line: tune, bench, info"""

import argparse
import sys
from typing import Sequence

import stackkernels
from stackkernels._opt.hardware import get_hardware_info
from stackkernels._opt.tune import tune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stackkernels",
        description="StackKernels CLI: high-performance Triton kernels for transformer models.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: info
    subparsers.add_parser("info", help="Display system, hardware, and backend information.")

    # Subcommand: tune
    tune_parser = subparsers.add_parser("tune", help="Autotune an operator and write to cache.")
    tune_parser.add_argument("--op", type=str, default="rms_norm", help="Operator name (e.g. rms_norm)")
    tune_parser.add_argument("--shape", type=int, nargs="+", default=[2, 2048, 4096], help="Problem shape")
    tune_parser.add_argument("--dtype", type=str, default="float16", help="Data type")

    # Subcommand: bench
    bench_parser = subparsers.add_parser("bench", help="Benchmark an operator.")
    bench_parser.add_argument("--op", type=str, default="rms_norm", help="Operator name")
    bench_parser.add_argument("--shape", type=int, nargs="+", default=[2, 2048, 4096], help="Problem shape")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "info":
        hw = get_hardware_info()
        print("=== StackKernels System Info ===")
        print(f"StackKernels version: {stackkernels.__version__}")
        print(f"Active backend:       {stackkernels.get_backend()}")
        print(f"Device name:          {hw.name}")
        print(f"Architecture:         {hw.arch}")
        print(f"GPU Available:        {hw.has_gpu}")
        print(f"Compute Units / SMs:  {hw.sm_count}")
        return 0

    elif args.subcommand == "tune":
        shape = tuple(args.shape)
        print(f"Tuning op '{args.op}' for shape {shape} ({args.dtype})...")
        best = tune(op=args.op, shape=shape, dtype=args.dtype)
        print(f"Best config: {best}")
        return 0

    elif args.subcommand == "bench":
        shape = tuple(args.shape)
        print(f"Benchmarking {args.op} with shape {shape}...")
        try:
            from benchmarks.bench_norm import benchmark_rms_norm

            benchmark_rms_norm()
        except Exception as e:
            print(f"Benchmark error: {e}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
