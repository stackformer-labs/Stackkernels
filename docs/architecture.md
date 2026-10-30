# StackKernels Architecture

StackKernels is structured into cleanly isolated layers with strict boundary contracts.

```text
       ┌────────────────────────────────────────────────────────┐
       │                   User Applications                    │
       └───────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌───────────────────────┐               ┌───────────────────────┐
   │ stackkernels.torch    │               │ stackkernels.jax      │
   │ (Autograd functions,  │               │ (custom_vjp,          │
   │  torch.compile,       │               │  triton_call,         │
   │  nn.Modules)          │               │  JIT transforms)      │
   └──────────┬────────────┘               └──────────┬────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ stackkernels._opt       │
                     │ (pick, tune, occupancy, │
                     │  classes, constraints)  │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ stackkernels._core      │
                     │ (Pure Triton kernels,   │
                     │  KERNEL_REV integers)   │
                     └─────────────────────────┘
```

---

## Layers

### Layer 1: Core (`src/stackkernels/_core/`)
- Contains pure Triton kernels.
- **Strict Rule**: `_core` modules **never** import `_opt`, `torch`, or `jax`.
- Receives tuning parameters as explicit kernel arguments. Never uses `@triton.autotune`.
- Each kernel module defines a module-level `KERNEL_REV = 1` integer for cache invalidation.

### Layer 2: Optimization (`src/stackkernels/_opt/`)
- Determines *how* to run kernels on specific hardware.
- Public entry points are strictly `pick()` and `tune()`.
- **Submodules**:
  - `hardware.py`: Queries device properties into a frozen `DeviceProfile` with fallback tables and stable `fingerprint()`.
  - `classes/`: Pure Python seed rules for the 4 kernel classes (`row`, `elementwise`, `tiled`, `decode`).
  - `constraints.py`: Static validation (`check()`) pruning invalid configurations (smem 10% safety margin, accumulator register budgets, thread limits).
  - `occupancy.py`: Analytical SM occupancy calculations (`programs_per_sm`, `waves`) and compiler adapter `probe()`.
  - `search_spaces.py`: Generates candidate neighbors around seeds, filtered by constraints (capped at 12).
  - `calibrate.py`: Hardware calibration measuring achieved peak copy bandwidth and matmul TFLOPs.
  - `cache.py`: Manages in-memory and disk cache keyed by 6-tuple `(op, fingerprint, triton_ver, kernel_rev, dtype, bucket)`.
  - `heuristics/`: Backward-compatible thin wrappers delegating to `_opt/classes/`.
- **Strict Rule**: Heuristics and class rules are pure Python with no framework imports at module level (unit-testable on CPU).

### Layer 3a: PyTorch Bindings (`src/stackkernels/torch/`)
- PyTorch frontend implementations.
- Implements `torch.autograd.Function`, `torch.library.custom_op`, and `nn.Module`.
- Resolves inputs to contiguous 2D tensors, calls `_opt.pick()` for device configuration, and launches `_core` kernels or dispatches to reference fallback on CPU.

### Layer 3b: JAX Bindings (`src/stackkernels/jax/`)
- JAX frontend implementations.
- Implements `jax.custom_vjp` and `jax.triton_call`.
- Matches PyTorch numerical behavior and shapes.

### Hub (`src/stackkernels/hub/`)
- Op name to implementation registry per backend.

---

## Architectural Rules

1. **`_core` Isolation**: `_core` kernels contain only computation logic. No dispatch decisions, no hardware querying, no framework dependencies.
2. **Framework Isolation**: PyTorch and JAX bindings are mutually independent; blocking one framework does not prevent importing or using the other.
3. **CPU Compatibility**: All heuristics, profiles, search spaces, and constraints run and pass unit tests on CPU environments.
4. **No Hidden Constants**: Hardware constants are confined strictly to fallback tables and test fixtures.
