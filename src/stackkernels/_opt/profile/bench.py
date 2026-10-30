"""timing harness: warmup, CUDA events, L2 flush"""

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class BenchResult:
    """Benchmark timing result statistics."""

    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def __repr__(self) -> str:
        return f"BenchResult(median={self.median_ms:.4f} ms, mean={self.mean_ms:.4f} ms, min={self.min_ms:.4f} ms)"


def get_l2_cache_tensor(size_mb: int = 40, device: torch.device | str = "cuda") -> torch.Tensor | None:
    """Allocate a buffer large enough to flush L2 cache."""
    try:
        num_elements = (size_mb * 1024 * 1024) // 4
        return torch.empty(num_elements, dtype=torch.int32, device=device)
    except Exception:
        return None


def bench_fn(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 20,
    iters: int = 50,
    l2_flush: bool = True,
    **kwargs: Any,
) -> BenchResult:
    """Measure execution latency with warmups, CUDA events, and L2 cache flushing."""
    is_cuda = False
    try:
        is_cuda = torch.cuda.is_available() and any(
            isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args
        )
    except Exception:
        is_cuda = False

    # 1. Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()

    # 2. Timing
    times_ms: list[float] = []

    if is_cuda:
        device = args[0].device
        l2_tensor = get_l2_cache_tensor(40, device) if l2_flush else None

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            if l2_tensor is not None:
                l2_tensor.zero_()
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args, **kwargs)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    mean_val = statistics.mean(times_ms)
    median_val = statistics.median(times_ms)
    min_val = min(times_ms)
    max_val = max(times_ms)
    std_val = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    return BenchResult(
        mean_ms=mean_val,
        median_ms=median_val,
        min_ms=min_val,
        max_ms=max_val,
        std_ms=std_val,
    )
