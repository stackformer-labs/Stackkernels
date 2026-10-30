# Getting Started with StackKernels

This guide walks you through installing StackKernels across supported platforms, verifying your GPU environment, and running your first kernel.

---

## Installation

StackKernels requires Python 3.9 or newer. Hardware GPU execution requires NVIDIA CUDA and an environment capable of running OpenAI Triton (Triton 2.3 or newer).

### Linux and WSL2

On Linux or Windows Subsystem for Linux (WSL2), install StackKernels with framework extras using pip:

```bash
# For PyTorch support
pip install "stackkernels[torch]"

# For JAX support
pip install "stackkernels[jax]"

# For development and testing tools
pip install "stackkernels[torch,dev,docs]"
```

To install from source in editable mode:

```bash
git clone https://github.com/stackformer-labs/stackkernels.git
cd stackkernels
pip install -e ".[torch,dev]"
```

### Google Colab and Kaggle

Google Colab and Kaggle Linux environments ship with preinstalled CUDA drivers and PyTorch. In a notebook cell, install the package directly:

```bash
!pip install --no-deps -e .
```

You can run the bundled validation suite directly in a notebook to verify kernel operations against your allocated GPU:

```python
import stackkernels
print(stackkernels.__version__)
```

### Windows Native Notes

OpenAI Triton does not officially support native Windows without custom binary distributions. When you install or run StackKernels on native Windows without a working Triton toolchain, the library automatically falls back to pure PyTorch and JAX reference implementations.

To execute GPU Triton kernels on Windows, use **WSL2 (Ubuntu 22.04 or 24.04)** with NVIDIA CUDA drivers passed through from the host.

### Compute Capability and BF16

- **FP16 and FP32**: Supported across all CUDA compute capabilities sm70+ (Volta, Turing, Ampere, Ada, Hopper).
- **BF16**: Native bfloat16 hardware instructions require compute capability **sm80 or newer** (Ampere A100/RTX 3090, Ada RTX 4090, Hopper H100). On sm75 (such as Tesla T4), BF16 operations are rejected by optimization constraints or routed to reference implementations.

---

## Verifying Your Setup with the CLI

Run the `info` command to inspect your system environment, detect active hardware profiles, and review calibrated performance limits:

```bash
python -m stackkernels info
```

Example output on a GPU-enabled machine (Tesla T4):

```text
=== StackKernels System & Hardware Info ===
StackKernels version: 0.1.0
PyTorch version:      2.3.0+cu121
Triton version:       3.0.0
Active backend:       auto
--- Device Profile ---
Device name:          Tesla T4
Compute capability:   sm_75
SM / Core count:      40
Total memory:         14.56 GB
Has cp.async:         False
Has native BF16:      False
Device fingerprint:   tesla-t4_sm75_40sm_14912mb
--- Calibrated Peaks ---
Status:               spec / uncalibrated
Peak bandwidth:       320.0 GB/s
FP16 compute:         65.0 TFLOPs
FP32 compute:         8.1 TFLOPs
```

On CPU-only environments, the command displays the host CPU core count, indicates that `is_cuda` is false, and reports the active fallback backend.

---

## Your First Kernel Call

Below is a complete script demonstrating Root Mean Square Normalization (`rms_norm`). You can execute this script on CPU or GPU without modifications:

```python
import torch
from stackkernels.torch import rms_norm

# Allocate sample tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 512, 1024, device=device, dtype=torch.float32)
weight = torch.ones(1024, device=device, dtype=torch.float32)

# Execute normalized forward pass
y = rms_norm(x, weight=weight, eps=1e-6)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
```

If CUDA is available, `rms_norm` automatically launches the Triton kernel configured for your specific GPU. If CUDA is unavailable, it runs the PyTorch reference implementation.

---

## Controlling Execution Backends

StackKernels provides three backend modes:

1. `"auto"` (default): Uses Triton kernels whenever a CUDA tensor is provided; falls back to reference code on CPU.
2. `"triton"`: Enforces Triton execution. Raises an error or invokes the Triton interpreter if executed without a CUDA device.
3. `"reference"`: Enforces pure PyTorch or JAX eager reference code regardless of device.

You can set the backend programmatically or via environment variables:

```python
import stackkernels

# Set globally in Python code
stackkernels.set_backend("reference")
print(stackkernels.get_backend())  # "reference"

# Or configure via context-friendly helper
stackkernels.options(backend="auto")
```

To configure from your terminal before starting a workload:

```bash
export STACKKERNELS_BACKEND=reference
python train.py
```

