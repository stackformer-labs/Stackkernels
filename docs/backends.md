# Backends and Fallback System

This page explains how StackKernels selects execution backends, manages fallback behavior across heterogeneous environments, and enables debugging with the reference implementations.

---

## Backend Selection Overview

StackKernels supports three operational backends configured through `stackkernels.options`:

| Backend Name | Behavior | Target Use Case |
|:---|:---|:---|
| `"auto"` (default) | Automatically executes Triton kernels on CUDA devices; routes CPU tensors to pure framework reference code. | Production training and inference on GPU clusters and local workstations. |
| `"triton"` | Forces execution of Triton kernels. Raises an exception if CUDA is unavailable, unless `TRITON_INTERPRET=1` is set. | Kernel development, validation, and benchmarking of GPU pipelines. |
| `"reference"` | Bypasses Triton kernels entirely, executing pure PyTorch or JAX eager reference code. | Unit testing on CPU, gradient debugging, and verifying numerical consistency. |

---

## Configuring the Backend

You can configure the active backend in Python code using `stackkernels.options()` or `stackkernels.set_backend()`:

```python
import stackkernels

# Configure backend using options()
stackkernels.options(backend="reference")

# Check current backend
print(stackkernels.get_backend())  # "reference"

# Switch back to automatic mode
stackkernels.set_backend("auto")
```

Alternatively, set the backend globally using the `STACKKERNELS_BACKEND` environment variable before executing your application:

```bash
export STACKKERNELS_BACKEND=reference
pytest tests/ops/test_rms_norm.py
```

---

## How Dispatch Works Under the Hood

### PyTorch Dispatch Pipeline

In `stackkernels.torch`, every functional call delegates to a custom `torch.autograd.Function`. During the forward pass:

1. The function evaluates `should_use_triton(tensor)`:
    - If `get_backend() == "reference"`, returns `False`.
    - If `get_backend() == "triton"`, returns `True`.
    - If `get_backend() == "auto"`, returns `True` if `tensor.is_cuda` is true, otherwise `False`.
2. When Triton is selected:
    - Inputs are verified for memory contiguity using `ensure_contiguous()`.
    - Higher-dimensional tensors are flattened to 2D matrices where required by the kernel.
    - `stackkernels._opt.pick()` resolves the optimal block dimensions and pipeline stages for the device.
    - The core Triton kernel executes.
3. When reference fallback is selected:
    - Execution routes directly to `src/stackkernels/_reference/torch/`.
    - Calculations run in FP32 precision where appropriate for maximum numerical fidelity.
    - Autograd graphs are tracked natively via standard PyTorch operations.

### JAX Dispatch Pipeline

In `stackkernels.jax`:

- Operations are wrapped with `jax.custom_vjp` to ensure seamless automatic differentiation.
- For GPU compilation, kernels bind to Triton through `jax.triton_call`.
- On CPU platforms, JAX primitives invoke pure JAX implementations registered in `src/stackkernels/_reference/jax/`, supporting `jax.jit`, `jax.vmap`, and `jax.grad`.

---

## Environment Variables Reference

The following environment variables control library behavior:

- `STACKKERNELS_BACKEND`: Sets default backend mode (`auto`, `triton`, or `reference`).
- `STACKKERNELS_CACHE_DIR`: Directory path for saving and loading autotuning cache files. Defaults to `~/.cache/stackkernels/`.
- `TRITON_INTERPRET`: When set to `"1"`, instructs OpenAI Triton to execute kernels in software emulation mode on CPU. Used primarily for inspecting Triton kernel internals when no GPU is present.

