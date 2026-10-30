# Troubleshooting

This guide details common installation and runtime issues encountered with StackKernels and how to resolve them.

---

## 1. Triton Installation Issues on Windows

### Problem
Attempting to install or run Triton directly on native Windows generates compiler errors or warnings:
```text
UserWarning: Failed to find CUDA. Install CUDA Toolkit or set CUDA_HOME / CUDA_PATH
```

### Cause
OpenAI Triton is designed for Linux environments and does not provide native Windows compiler toolchains in official releases.

### Solution
- **For GPU execution on Windows**: Install **WSL2** with Ubuntu 22.04 or 24.04 and install NVIDIA drivers with CUDA support.
- **For CPU execution / testing**: Run with `stackkernels.options(backend="reference")`. StackKernels will automatically execute reference implementations without invoking Triton.

---

## 2. BF16 Unsupported on Compute Capability < sm80 (e.g. Tesla T4)

### Problem
Running kernels with `torch.bfloat16` on NVIDIA Tesla T4 (sm75) results in a constraint rejection or runtime failure:
```text
Config rejected by constraint: Native BF16 requires compute capability >= sm80
```

### Cause
NVIDIA Turing GPUs (sm75, including Tesla T4 and RTX 2080) lack native hardware tensor core instructions for bfloat16. Native BF16 requires Ampere (sm80+) or newer.

### Solution
- Convert inputs to `torch.float16` or `torch.float32` when executing on Tesla T4 or Turing architectures:
  ```python
  x = x.to(torch.float16)
  ```
- If BF16 is required for parity tests, switch to the reference backend:
  ```python
  import stackkernels
  stackkernels.options(backend="reference")
  ```

---

## 3. CUDA Out of Memory or Block Size Rejections

### Problem
The optimization layer rejects large block sizes with shared memory warnings:
```text
Shared memory requirement exceeds device limit with safety margin
```

### Cause
StackKernels enforces a 10% safety margin on GPU shared memory to prevent hardware allocation failures. For wide hidden dimensions ($D > 8192$) or large attention sequence tiles, single-pass tiles may exceed SM limits.

### Solution
- Allow `pick()` to select chunked or multi-pass configurations automatically.
- Check current allocations using `python -m stackkernels info`.
- Reduce sequence chunk sizes ($S_q$ or $S_k$) or hidden dimension block sizes.

---

## 4. PyTorch and Triton Version Mismatches

### Problem
Import errors or kernel launch failures occur after upgrading PyTorch:
```text
RuntimeError: Triton compiler version mismatch or symbol not found
```

### Cause
PyTorch bundles or pairs with specific minor releases of Triton. Upgrading PyTorch independently of Triton can cause ABI mismatches.

### Solution
Reinstall matching PyTorch and Triton versions specified in `pyproject.toml`:
```bash
pip install "torch>=2.3,<3.0" "triton>=2.3"
```
Check active versions using:
```bash
python -m stackkernels info
```

---

## 5. Non-Contiguous Tensors in Custom Pipelines

### Problem
Kernel launches fail or output gradients show unexpected strides.

### Cause
Custom operations or permuted views (such as `.transpose(1, 2)`) can produce non-contiguous memory layouts that Triton pointer arithmetic cannot traverse without stride adjustments.

### Solution
StackKernels includes internal contiguous guards (`ensure_contiguous()`), but when interacting directly with low-level `_core` kernels, always call `.contiguous()` on input tensors before passing them:
```python
x = x.contiguous()
```

