# stackkernels

Fast, fused Triton kernels for transformer models — built as the kernel backbone for [Stackformer](#).

Supports **PyTorch** and **JAX**, with a shared, framework-agnostic Triton core.

## Status
🚧 Early development. Build order: RMSNorm → LayerNorm → Flash Attention → Dropout → Activations (SiLU/GELU/SwiGLU) → Cross-Entropy → RoPE.

## Install
```bash
pip install stackkernels
```

## Quick start
```python
from stackkernels.torch import RMSNorm
```

## License
MIT