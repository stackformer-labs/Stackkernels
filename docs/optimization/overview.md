# Optimization Overview: The Three Execution Levels

This page explains how StackKernels selects block configurations and tuning parameters for your hardware across three user-controlled levels: automatic heuristics, manual overrides, and offline autotuning.

---

## Why Kernel Configuration Matters

Triton kernels execute in parallel across GPU Streaming Multiprocessors (SMs). To achieve high performance, each kernel needs concrete decisions about:

- **Block Sizes**: How many rows, columns, or tokens a single thread block processes in SRAM.
- **Warp Counts**: How many 32-thread warps execute concurrently per block.
- **Pipeline Stages**: How many memory read operations are prefetched into shared memory simultaneously.

An improper block size can cause register spills to slow local memory, exhaust shared memory, or leave multiprocessors idle. StackKernels eliminates trial-and-error tuning through three progressive levels.

---

## Level 1: Automatic Heuristics (Zero Configuration)

By default, every call to `stackkernels.torch` or `stackkernels.jax` resolves its execution configuration automatically. You do not need to pass configurations or launch autotuning scripts:

```python
from stackkernels.torch import rms_norm

# Works out of the box with zero manual configuration
out = rms_norm(x, weight)
```

### What Happens Behind the Scenes
When an operator runs, StackKernels queries the hardware profile of your active GPU (such as its SM count, shared memory limits, and register budget) and calculates a rule-based block configuration tailored to your tensor dimensions. It prunes configurations that would cause shared memory exhaustion or register spilling before launching the kernel.

---

## Level 2: Manual Configuration Overrides

If you want to benchmark specific tile dimensions or force experimental block parameters, you can bypass the heuristic by passing an `override` dictionary:

```python
from stackkernels._opt import pick

# Explicitly test a custom configuration
cfg = pick("rms_norm", shape=(2048, 1024), override={"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1})
```

Overrides take highest precedence in the resolution pipeline and are executed directly without heuristic adjustment.

---

## Level 3: Offline Autotuning (`sk.tune`)

For production training runs on fixed model architectures and dedicated GPU clusters, you can run offline autotuning to measure achieved execution speeds across candidate configurations and persist the best configuration to disk:

```python
from stackkernels._opt import tune

# Benchmark candidate configurations on the active GPU and save the winner
best_config = tune("rms_norm", shape=(2048, 4096), dtype="float16")
print(f"Optimal configuration: {best_config}")
```

Once tuned, all subsequent calls for that operator, hardware fingerprint, and tensor shape immediately load the saved configuration from cache with zero benchmarking overhead.

---

## Summary Comparison

| Level | When to Use | Setup Required | Runtime Overhead |
|:---|:---|:---|:---|
| **Level 1: Auto Heuristic** | Development, interactive notebooks, dynamic shapes. | None | Microseconds (cached in-memory lookup). |
| **Level 2: Override** | Performance debugging, kernel experimentation. | Pass explicit dictionary. | Zero. |
| **Level 3: Autotune** | Production training jobs, fixed cluster hardware. | One-time benchmark run (`sk.tune`). | Zero at runtime; loads cached winner. |

