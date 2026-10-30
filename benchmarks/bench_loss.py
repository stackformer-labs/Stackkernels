"""Benchmark plain cross entropy and fused linear cross entropy.

Measures both execution speed and peak memory consumption across vocabulary sizes.
"""

import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn.functional as F

from stackkernels.torch.loss import cross_entropy, fused_linear_cross_entropy


@dataclass
class LossBenchmarkRecord:
    m: int
    k: int
    v: int
    dtype: str
    unfused_ms: float
    fused_ms: float
    speedup: float
    unfused_peak_mb: float
    fused_peak_mb: float
    memory_saved_pct: float


def measure_peak_memory_and_latency(fn: Any, warmup: int = 5, iters: int = 15) -> tuple[float, float]:
    """Measure median latency in ms and peak memory allocated in MB."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warmup
    for _ in range(warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Reset memory tracking
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]

    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 * 1024)
    else:
        peak_mb = 0.0

    return median_ms, peak_mb


def benchmark_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_str = "float16" if dtype == torch.float16 else "float32"

    if device.type == "cuda":
        configs = [
            (1024, 2048, 16384),
            (1024, 4096, 32000),
            (2048, 4096, 32000),
            (2048, 4096, 64000),
            (4096, 4096, 128000),
        ]
        warmup, iters = 5, 15
    else:
        configs = [
            (64, 128, 1024),
            (128, 256, 2048),
            (256, 256, 4096),
        ]
        warmup, iters = 2, 5

    records = []

    print(f"\n=== Benchmarking Loss: Unfused vs Fused Linear CE [{dtype_str}] (Device: {device}) ===")
    print(
        f"{'Shape (MxKxV)':<22} | {'Unfused (ms)':<12} | {'Fused (ms)':<12} | {'Speedup':<8} | "
        f"{'Unfused MB':<11} | {'Fused MB':<10} | {'Mem Saved':<10}"
    )
    print("-" * 96)

    for M, K, V in configs:
        x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
        w = torch.randn(V, K, dtype=dtype, device=device, requires_grad=True)
        target = torch.randint(0, V, (M,), dtype=torch.long, device=device)

        # Baseline: Unfused Linear + Cross Entropy
        def run_unfused():
            logits = F.linear(x, w)
            loss = F.cross_entropy(logits.float(), target)
            loss.backward()

        # StackKernels: Fused Linear + Cross Entropy
        def run_fused():
            loss = fused_linear_cross_entropy(x, w, target)
            loss.backward()

        try:
            unfused_ms, unfused_mb = measure_peak_memory_and_latency(run_unfused, warmup=warmup, iters=iters)
            fused_ms, fused_mb = measure_peak_memory_and_latency(run_fused, warmup=warmup, iters=iters)

            speedup = round(unfused_ms / max(fused_ms, 1e-6), 2)
            if unfused_mb > 0:
                mem_saved_pct = round((1.0 - (fused_mb / unfused_mb)) * 100.0, 1)
            else:
                # Theoretical calculation when on CPU
                unfused_theoretical = (M * V * 2 * 2 + M * K * 2 + V * K * 2) / (1024 * 1024)
                fused_theoretical = (min(M, 512) * V * 2 + M * K * 2 + V * K * 2) / (1024 * 1024)
                mem_saved_pct = round((1.0 - (fused_theoretical / unfused_theoretical)) * 100.0, 1)
                unfused_mb = round(unfused_theoretical, 2)
                fused_mb = round(fused_theoretical, 2)

            shape_str = f"{M}x{K}x{V}"
            print(
                f"{shape_str:<22} | {unfused_ms:<12.2f} | {fused_ms:<12.2f} | {speedup:<8.2f}x | "
                f"{unfused_mb:<11.2f} | {fused_mb:<10.2f} | {mem_saved_pct:<9.1f}%"
            )

            records.append(
                LossBenchmarkRecord(
                    m=M,
                    k=K,
                    v=V,
                    dtype=dtype_str,
                    unfused_ms=round(unfused_ms, 3),
                    fused_ms=round(fused_ms, 3),
                    speedup=speedup,
                    unfused_peak_mb=unfused_mb,
                    fused_peak_mb=fused_mb,
                    memory_saved_pct=mem_saved_pct,
                )
            )
        except Exception as e:
            print(f"Skipping {M}x{K}x{V} due to: {e}")

    # Save results to benchmarks/results/loss.json
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "loss.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    benchmark_loss()
