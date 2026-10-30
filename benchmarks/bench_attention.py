"""Benchmark FlashAttention and FlashDecode against baseline attention."""

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

from stackkernels._reference.torch.attention import attention_reference
from stackkernels.torch.attention import flash_attention, flash_decode


@dataclass
class AttentionBenchmarkRecord:
    mode: str
    shape_q: list[int]
    shape_kv: list[int]
    dtype: str
    is_causal: bool
    median_ms: float
    tflops: float


def measure_latency(fn: Any, warmup: int = 5, iters: int = 20) -> float:
    """Measure median execution time in ms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(warmup):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    return latencies[len(latencies) // 2]


def benchmark_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype_str = "float16" if dtype == torch.float16 else "float32"

    if device.type == "cuda":
        configs = [
            # (mode, B, H_q, H_kv, S_q, S_k, D, is_causal)
            ("prefill_mha", 2, 16, 16, 1024, 1024, 64, True),
            ("prefill_mha", 2, 16, 16, 2048, 2048, 64, True),
            ("prefill_mha", 1, 32, 32, 2048, 2048, 128, True),
            ("prefill_gqa", 2, 32, 8, 2048, 2048, 128, True),
            ("prefill_mqa", 2, 32, 1, 2048, 2048, 128, True),
            ("decode_mha", 4, 32, 32, 1, 2048, 128, False),
            ("decode_gqa", 4, 32, 8, 1, 4096, 128, False),
            ("decode_mqa", 4, 32, 1, 1, 8192, 128, False),
        ]
        warmup, iters = 5, 20
    else:
        configs = [
            ("prefill_mha", 1, 4, 4, 64, 64, 64, True),
            ("prefill_gqa", 1, 4, 2, 64, 64, 64, True),
            ("prefill_mqa", 1, 4, 1, 64, 64, 64, True),
            ("decode_gqa", 1, 4, 2, 1, 128, 64, False),
        ]
        warmup, iters = 2, 5

    records = []

    print(f"\n=== Benchmarking FlashAttention & FlashDecode [{dtype_str}] (Device: {device}) ===")
    print(f"{'Mode':<14} | {'Q Shape':<18} | {'KV Shape':<18} | {'Causal':<7} | {'Latency (ms)':<12} | {'TFLOPs':<8}")
    print("-" * 88)

    for mode, B, H_q, H_kv, S_q, S_k, D, is_causal in configs:
        q = torch.randn(B, H_q, S_q, D, dtype=dtype, device=device)
        k = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)
        v = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)

        if "decode" in mode and S_q == 1:
            fn = lambda: flash_decode(q, k, v)
        else:
            fn = lambda: flash_attention(q, k, v, is_causal=is_causal)

        try:
            latency_ms = measure_latency(fn, warmup=warmup, iters=iters)
            # 4 * B * H_q * S_q * S_k * D FLOPs
            flops = 4.0 * B * H_q * S_q * S_k * D
            if is_causal and S_q == S_k:
                flops /= 2.0  # causal triangular mask cuts FLOPs in half
            tflops = round(flops / (latency_ms * 1e9), 2)

            q_shape_str = f"({B},{H_q},{S_q},{D})"
            kv_shape_str = f"({B},{H_kv},{S_k},{D})"
            print(
                f"{mode:<14} | {q_shape_str:<18} | {kv_shape_str:<18} | {str(is_causal):<7} | "
                f"{latency_ms:<12.3f} | {tflops:<8.2f}"
            )

            records.append(
                AttentionBenchmarkRecord(
                    mode=mode,
                    shape_q=[B, H_q, S_q, D],
                    shape_kv=[B, H_kv, S_k, D],
                    dtype=dtype_str,
                    is_causal=is_causal,
                    median_ms=round(latency_ms, 3),
                    tflops=tflops,
                )
            )
        except Exception as e:
            print(f"Skipping {mode} due to: {e}")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "attention.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    benchmark_attention()
