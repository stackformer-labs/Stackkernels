# Architecture

## Layers
- **Layer 1 (`_core/`)**: Triton kernels only; no torch, jax, or _opt imports.
- **Layer 2 (`_opt/`)**: How to run kernels; public API is only pick() and tune().
- **Layer 3a (`torch/`)**: Public PyTorch API.
- **Layer 3b (`jax/`)**: Public JAX API.
- **Hub (`hub/`)**: Op name to implementation registry per backend.

## Import Rules
- _core never imports _opt, torch, or jax.
- _opt/heuristics never imports kernels or any framework (pure and testable on CPU).
- Public wrappers (torch/, jax/) import from _core and from _opt (only pick and tune).
- Package root __init__.py never imports torch or jax at import time.

