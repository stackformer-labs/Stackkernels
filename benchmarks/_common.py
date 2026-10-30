"""shared harness and JSON output"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stackkernels._opt.hardware import get_hardware_info
from stackkernels._opt.profile.bench import bench_fn
from stackkernels._opt.profile.cost_model import (
    calculate_achieved_gbps,
    calculate_memory_efficiency,
    rms_norm_bytes_moved,
)


@dataclass
class BenchmarkRecord:
    op: str
    shape: list[int]
    dtype: str
    median_ms: float
    bytes_moved: int
    achieved_gbps: float
    peak_gbps: float
    efficiency_pct: float


def run_op_benchmark(
    op: str,
    fn: Any,
    shapes: list[tuple[int, ...]],
    dtype_str: str = "float16",
    has_weight: bool = True,
    warmup: int = 25,
    iters: int = 50,
) -> list[BenchmarkRecord]:
    """Benchmark an op across shapes and calculate achieved GB/s vs peak."""
    records = []
    hw = get_hardware_info()
    peak_gbps = hw.peak_bandwidth_gbps

    print(f"\n=== Benchmarking {op} [{dtype_str}] (Device: {hw.name}, Arch: {hw.arch}) ===")
    print(f"{'Shape':<20} | {'Latency (ms)':<12} | {'Bytes Moved':<14} | {'Achieved GB/s':<14} | {'% Peak':<8}")
    print("-" * 80)

    for shape in shapes:
        bench_res = bench_fn(fn, shape, warmup=warmup, iters=iters)
        bytes_moved = rms_norm_bytes_moved(shape, dtype=dtype_str, mode="fwd", has_weight=has_weight)
        achieved_gbps = calculate_achieved_gbps(bytes_moved, bench_res.median_ms)
        eff_pct = calculate_memory_efficiency(achieved_gbps, peak_gbps)

        shape_str = "x".join(str(s) for s in shape)
        print(f"{shape_str:<20} | {bench_res.median_ms:<12.4f} | {bytes_moved:<14} | {achieved_gbps:<14.2f} | {eff_pct:<8.2f}%")

        records.append(
            BenchmarkRecord(
                op=op,
                shape=list(shape),
                dtype=dtype_str,
                median_ms=round(bench_res.median_ms, 4),
                bytes_moved=bytes_moved,
                achieved_gbps=round(achieved_gbps, 2),
                peak_gbps=peak_gbps,
                efficiency_pct=eff_pct,
            )
        )

    return records


def save_benchmark_results(name: str, records: list[BenchmarkRecord]) -> Path:
    """Save benchmark records to benchmarks/results/<name>.json."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = results_dir / f"{name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    print(f"\nSaved results to {out_file}")
    return out_file
