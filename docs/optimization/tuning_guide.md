# Tuning Guide: Autotuning, Explain Mode, and Cache Management

This guide explains how to use StackKernels autotuning tools, inspect decision rationale with `pick(explain=True)`, calibrate hardware baselines, and manage persistent tuning caches.

For the underlying mathematical cost models and kernel class taxonomy, see [Design & Pipeline Architecture](design.md).

---

## 1. Inspecting Decisions with `explain=True`

To understand why a particular block configuration was chosen, invoke `pick()` with `explain=True`. It returns a tuple of the chosen configuration and a human-readable explanation trace:

```python
from stackkernels._opt.pick import pick

config, explanation = pick("rms_norm", shape=(2048, 1024), dtype="float16", explain=True)
print(explanation)
```

Example output:

```text
=== Resolution Decision for 'rms_norm' ===
Device Fingerprint: tesla-t4_sm75_40sm_14912mb
Source:             class_heuristic
Selected Config:    {'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 1}
Rationale:          Heuristic 'row' selected for shape (2048, 1024) on sm_75.
```

The explanation trace reveals:
1. The active hardware fingerprint (e.g. `tesla-t4_sm75_40sm_14912mb`).
2. The resolution source (`user_override`, `user_cache`, `shipped_tuned`, or `class_heuristic`).
3. Any candidate configurations rejected during constraint evaluation (such as shared memory limit violations).

---

## 2. Running Autotuning with `tune()`

To autotune an operation from Python:

```python
from stackkernels._opt.tune import tune

# Runs microbenchmarks across valid candidates and writes winner to cache
best_config = tune(
    op="rms_norm",
    shape=(2048, 4096),
    dtype="float16",
    write_to_disk=True,
)
print("Best configuration:", best_config)
```

### Tuning via the Command Line Interface

You can also trigger autotuning directly from your terminal:

```bash
python -m stackkernels tune --op rms_norm --shape 2 2048 4096 --dtype float16
```

---

## 3. Persistent Cache Architecture

### Where the Cache Lives

StackKernels stores persistent tuning data in JSON format:
- **Default Location**: `~/.cache/stackkernels/user_cache.json`
- **Custom Location**: Configure the `STACKKERNELS_CACHE_DIR` environment variable:
  ```bash
  export STACKKERNELS_CACHE_DIR=/workspace/cache/stackkernels
  ```

### The 6-Tuple Cache Key

Each entry in the cache is uniquely identified by a 6-tuple string:

```text
op:device_fingerprint:triton_version:kernel_rev:dtype:shape_bucket
```

Example:
```text
rms_norm:tesla-t4_sm75_40sm_14912mb:3.6.0:1:float16:2048x1024
```

- **`op`**: Operator name (`rms_norm`, `swiglu`, `flash_attention`).
- **`device_fingerprint`**: Hardware descriptor including GPU model, compute capability, SM count, and VRAM.
- **`triton_version`**: Triton release version. If Triton is upgraded, stale cache entries are automatically invalidated.
- **`kernel_rev`**: Internal revision integer defined in each `_core` kernel (`KERNEL_REV = 1`). If the kernel math or register allocation changes, bumping this integer invalidates obsolete cache entries.
- **`dtype`**: Data type (`float16`, `float32`, `bfloat16`).
- **`shape_bucket`**: Canonical problem dimensions (e.g. `2048x1024`).

### Resetting the Cache

To clear all cached autotuning data:

```bash
# Delete the user cache file directly
rm ~/.cache/stackkernels/user_cache.json
```

Or programmatically in Python:

```python
from stackkernels._opt.cache import get_global_cache

cache = get_global_cache()
cache.clear()
```

---

## 4. Hardware Calibration CLI

To establish baseline hardware bandwidth and arithmetic throughput numbers for your machine, run `calibrate`:

```bash
python -m stackkernels calibrate
```

Example output:

```text
Calibrating device 'Tesla T4' (fingerprint: tesla-t4_sm75_40sm_14912mb)...
Calibration complete:
  Achieved Bandwidth: 242.3 GB/s
  FP16 Throughput:    58.4 TFLOPs
  FP32 Throughput:    7.8 TFLOPs
```

These measured peaks are stored alongside your device profile and allow the optimization cost model to calculate achieved percentage of hardware peak for every kernel run.

