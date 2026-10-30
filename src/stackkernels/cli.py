"""Command line interface: info, calibrate, tune, bench."""

import argparse
import sys
from typing import Sequence

import stackkernels
from stackkernels._opt.calibrate import calibrate_device, measured_peaks
from stackkernels._opt.hardware import get_device_profile
from stackkernels._opt.tune import tune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stackkernels",
        description="StackKernels CLI: high-performance Triton kernels for transformer models.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: info
    subparsers.add_parser("info", help="Display system, hardware, versions, and calibrated peaks.")

    # Subcommand: calibrate
    subparsers.add_parser("calibrate", help="Benchmark and record device peak bandwidth and throughput.")

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
        prof = get_device_profile()
        peaks = measured_peaks(prof)

        # Query versions
        torch_ver = "not installed"
        try:
            import torch
            torch_ver = torch.__version__
        except Exception:
            pass

        triton_ver = "not installed"
        try:
            import triton
            triton_ver = getattr(triton, "__version__", "unknown")
        except Exception:
            pass

        print("=== StackKernels System & Hardware Info ===")
        print(f"StackKernels version: {stackkernels.__version__}")
        print(f"PyTorch version:      {torch_ver}")
        print(f"Triton version:       {triton_ver}")
        print(f"Active backend:       {stackkernels.get_backend()}")
        print("--- Device Profile ---")
        print(f"Device name:          {prof.name}")
        maj, min_ = prof.compute_capability
        arch_str = f"sm_{maj}{min_}" if prof.is_cuda else "cpu"
        print(f"Compute capability:   {arch_str}")
        print(f"SM / Core count:      {prof.sm_count}")
        mem_gb = round(prof.total_memory_bytes / (1024**3), 2)
        print(f"Total memory:         {mem_gb} GB" if prof.is_cuda else "Total memory:         Host RAM")
        print(f"Has cp.async:         {prof.has_cp_async}")
        print(f"Has native BF16:      {prof.has_bf16}")
        print(f"Device fingerprint:   {prof.fingerprint()}")
        print("--- Calibrated Peaks ---")
        status_str = "measured" if peaks.get("calibrated") else "spec / uncalibrated"
        print(f"Status:               {status_str}")
        print(f"Peak bandwidth:       {peaks.get('bandwidth_gbps', 0.0)} GB/s")
        print(f"FP16 compute:         {peaks.get('tflops_fp16', 0.0)} TFLOPs")
        print(f"FP32 compute:         {peaks.get('tflops_fp32', 0.0)} TFLOPs")
        return 0

    elif args.subcommand == "calibrate":
        prof = get_device_profile()
        print(f"Calibrating device '{prof.name}' (fingerprint: {prof.fingerprint()})...")
        peaks = calibrate_device(prof, force=True)
        print("Calibration complete:")
        print(f"  Achieved Bandwidth: {peaks.get('bandwidth_gbps', 0.0)} GB/s")
        print(f"  FP16 Throughput:    {peaks.get('tflops_fp16', 0.0)} TFLOPs")
        print(f"  FP32 Throughput:    {peaks.get('tflops_fp32', 0.0)} TFLOPs")
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
