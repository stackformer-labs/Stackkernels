<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo-light.png" alt="StackKernels logo" width="560">
  </picture>
</p>
<p align="center">
  <a href="https://pypi.org/project/stackkernels/"><img src="https://img.shields.io/pypi/v/stackkernels.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/stackkernels/"><img src="https://img.shields.io/pypi/pyversions/stackkernels.svg" alt="Python versions" /></a>
  <a href="https://github.com/stackformer-labs/stackkernels/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
  <a href="https://github.com/stackformer-labs/stackkernels/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/stackformer-labs/stackkernels/test.yml?branch=main&label=CI" alt="CI status" />
  </a>
</p>

# StackKernels
 
**Fast, fused Triton kernels for transformer models — PyTorch and JAX backends.**

---

## What is StackKernels?

StackKernels is a library of fused GPU kernels written in OpenAI Triton for transformer training and inference. It provides clean, autograd-compatible frontends for both PyTorch and JAX. While originally engineered as the high-throughput kernel engine for the Stackformer architecture, StackKernels can be used as a standalone dependency in any deep learning project.

---

## Why StackKernels?

Standard deep learning frameworks evaluate multi-step neural network layers by launching separate GPU kernels for each elementary arithmetic operation. This pattern writes intermediate tensors to high-bandwidth memory (HBM) only to read them back immediately in the subsequent kernel, saturating memory bandwidth and driving up peak memory consumption.

StackKernels addresses these limitations:
- **Memory Fusion**: Fuses residual additions with normalizations, non-linearities with gating multiplications, and linear projections with cross-entropy loss directly inside GPU register files and SRAM.
- **Drastic Peak Memory Reduction**: By computing linear projection and online softmax cross-entropy in a single fused tiled loop, large language model training avoids materializing full vocabulary logit matrices in global memory.
- **Device-Aware Execution**: Block dimensions, warp counts, and pipeline stages are derived directly from the physical characteristics of your GPU (SM count, register limits, shared memory) rather than fragile static constants.
- **CPU Reference Fallback**: Automatically routes execution to pure PyTorch or JAX reference code on CPU, allowing local debugging and CI test execution without requiring a CUDA GPU.

---

## Supported Operations Status

The table below reflects the real implementation and validation status of operations in the repository:

| Operation | PyTorch Wrapper | JAX Wrapper | Backward Pass | Validated on T4 | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| `rms_norm` | Yes | Yes | Yes | Yes | Functional + `RMSNorm` module. Single-pass row reduction up to $D=8192$. |
| `layer_norm` | Yes | Yes | Yes | Yes | Functional + `LayerNorm` module. Single-pass mean and variance reduction. |
| `fused_add_norm` | Yes | Yes | Yes | Yes | Fused residual add + norm (RMSNorm or LayerNorm). Dual output gradients. |
| `swiglu` | Yes | Yes | Yes | Yes | Fused `silu(gate) * up`. Automatic chunking if `up` is None. |
| `geglu` | Yes | Yes | Yes | Yes | Fused `gelu(gate) * up` (tanh approximation). |
| `reglu` | Yes | Yes | Yes | Yes | Fused `relu(gate) * up`. |
| `glu` | Yes | Yes | Yes | Yes | General GLU dispatcher (`silu`, `gelu`, `relu`). |
| `gelu` | Yes | Yes | Yes | Yes | Elementwise GELU (exact erf or tanh approximation). |
| `silu` | Yes | Yes | Yes | Yes | Elementwise SiLU (Swish). |
| `relu` | Yes | Yes | Yes | Yes | Elementwise ReLU. |
| `leaky_relu` | Yes | Yes | Yes | Yes | Elementwise LeakyReLU with configurable negative slope. |
| `sigmoid` | Yes | Yes | Yes | Yes | Elementwise Sigmoid. |
| `apply_rope` | Yes | Yes | Yes | Yes | Functional + `RoPE` module. Supports GQA/MQA, `start_pos > 0`. |
| `cross_entropy` | Yes | Yes | Yes | Yes | Online softmax CE, `ignore_index`, `label_smoothing`, reductions. |
| `fused_linear_cross_entropy` | Yes | Yes | Yes | Yes | Fused GEMM + CE without materializing logits. |
| `flash_attention` | Yes | Yes | Yes | Yes | FlashAttention-2 fwd & bwd. MHA, GQA, MQA, causal bottom-right mask. |
| `flash_decode` | Yes | Yes | No (inf. only) | Yes | Split-KV FlashDecode for single-token generation ($S_q = 1$). |
| `dropout` | Yes | Yes | Yes | Yes | Low-memory Philox PRNG dropout without saving mask. |
| `fused_dropout_add` | Yes | Yes | Yes | Yes | Fused dropout + residual addition without extra tensor allocations. |

### Operations in Development (Not Yet Implemented)

The following operators currently exist as roadmap stubs and are planned for future releases:
- In-place KV-cache update (`src/stackkernels/torch/cache.py`)
- Fused Top-K / Top-P sampling (`src/stackkernels/torch/sampling.py`)

---

## Installation

StackKernels requires Python 3.9 or newer and Triton 2.3 or newer.

```bash
# PyTorch extra
pip install "stackkernels[torch]"

# JAX extra
pip install "stackkernels[jax]"

# Editable source install for development
git clone https://github.com/stackformer-labs/stackkernels.git
cd stackkernels
pip install -e ".[torch,dev,docs]"
```

### Platform & Hardware Requirements

- **Linux**: Supported natively with CUDA 11.8+ or 12.0+.
- **Windows**: OpenAI Triton does not officially support native Windows compilers. When running on native Windows without a custom Triton toolchain, StackKernels automatically routes execution to CPU reference implementations. To run Triton GPU kernels on Windows, use **WSL2** with CUDA driver passthrough.
- **Google Colab & Kaggle**: Works out of the box on Linux GPU runtimes.
- **Data Types & Compute Capabilities**:
  - `float16` and `float32`: Supported on all compute capabilities sm70+ (Volta, Turing, Ampere, Ada, Hopper).
  - `bfloat16`: Requires compute capability **sm80 or newer** (Ampere A100/RTX 3090, Hopper H100). On sm75 (Tesla T4), BF16 operations fall back to reference implementations.

---

## Quick Start

### 1. Zero-Configuration Usage (Functional API)

```python
import torch
from stackkernels.torch import flash_attention, rms_norm, swiglu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fused RMSNorm
x = torch.randn(2, 512, 1024, device=device, requires_grad=True)
weight = torch.ones(1024, device=device)
y = rms_norm(x, weight)
y.sum().backward()

# Fused SwiGLU
gate = torch.randn(2, 512, 2048, device=device)
up = torch.randn(2, 512, 2048, device=device)
act = swiglu(gate, up)

# FlashAttention (MHA / GQA / MQA)
q = torch.randn(2, 8, 512, 64, device=device)
k = torch.randn(2, 2, 512, 64, device=device)  # 2 KV heads (GQA)
v = torch.randn(2, 2, 512, 64, device=device)
attn_out = flash_attention(q, k, v, is_causal=True)
```

### 2. Object-Oriented Drop-in Modules

```python
import torch
import torch.nn as nn
from stackkernels.torch import CrossEntropyLoss, RMSNorm, RoPE

# Replace nn.LayerNorm
norm = RMSNorm(dim=1024)

# Rotary Position Embedding module caching frequencies
rope = RoPE(dim=64, max_seq_len=4096)

# Online Cross-Entropy Loss
criterion = CrossEntropyLoss(ignore_index=-100)
```

### 3. Optimization Overrides and Autotuning

```python
from stackkernels._opt import pick, tune

# Inspect why a configuration was chosen
cfg, explanation = pick("rms_norm", shape=(2048, 1024), dtype="float16", explain=True)
print(explanation)

# Or trigger offline autotuning on active hardware
best_config = tune("rms_norm", shape=(2048, 1024), dtype="float16")
```

### 4. JAX Frontend

```python
import jax
import jax.numpy as jnp
from stackkernels.jax import rms_norm, swiglu

x = jnp.ones((2, 128, 512), dtype=jnp.float32)
w = jnp.ones((512,), dtype=jnp.float32)

# Full JIT and autodiff compatibility
grad_x = jax.grad(lambda x_in: jnp.sum(rms_norm(x_in, w)))(x)
```

---

## How It Works: Layered Architecture

StackKernels strictly enforces isolation between computational layers:

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

- **Layer 1: Core (`_core/`)**: Pure Triton kernels. Modules in `_core` never import `torch`, `jax`, or `_opt`.
- **Layer 2: Optimization (`_opt/`)**: Pure Python heuristics, constraint verification, occupancy models, and persistent tuning cache.
- **Layer 3: Bindings (`torch/`, `jax/`)**: Framework adapters handling tensor layout, autograd graphs, and fallback routing.

---

## Tuning and Device Awareness

Configuration decisions in StackKernels follow a strict 4-stage resolution hierarchy:
1. **User Override**: Explicit parameters passed directly by the user.
2. **User Cache**: Winning configurations tuned offline on your specific device (`~/.cache/stackkernels/user_cache.json`).
3. **Shipped Defaults**: Validated configurations shipped with the package for known GPU models.
4. **Class Heuristics**: Pure Python rules deriving block sizes and warp counts from your GPU's active `DeviceProfile`.

To inspect your hardware profile and calibrated bandwidth peaks from your terminal:

```bash
python -m stackkernels info
```

For detailed instructions, see the [Tuning Guide](docs/optimization/tuning_guide.md).

---

## Performance

<!-- PERFORMANCE:START -->
Benchmarks have not been published yet. They will appear here after the T4 benchmark run, generated from `docs/benchmarks.md`.
<!-- PERFORMANCE:END -->

---

## Stackformer Integration

You can replace transformer blocks in Stackformer with StackKernels in a few lines of code:

```python
import torch
import torch.nn as nn
from stackkernels.torch import RMSNorm, flash_attention, swiglu

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.post_attn_norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused Attention
        normed = self.input_norm(x)
        q, k, v = self.qkv_proj(normed).chunk(3, dim=-1)
        # Reshape to (B, H, S, D)
        b, s, d = x.shape
        q = q.view(b, s, 8, d // 8).transpose(1, 2)
        k = k.view(b, s, 8, d // 8).transpose(1, 2)
        v = v.view(b, s, 8, d // 8).transpose(1, 2)
        attn = flash_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, d)
        h = x + self.out_proj(attn)

        # Fused SwiGLU MLP
        normed_h = self.post_attn_norm(h)
        mlp = self.down_proj(swiglu(self.gate_proj(normed_h), self.up_proj(normed_h)))
        return h + mlp
```

For the complete runnable example, see `examples/stackformer_integration.py` and the [Integration Guide](docs/stackformer_integration.md).

---

## Repository Structure

```text
stackkernels/
├── .github/workflows/          # CI and documentation workflows
├── benchmarks/                 # Microbenchmarks and result JSON artifacts
├── docs/                       # MkDocs documentation site source
│   ├── api/                    # Autogenerated API references
│   ├── optimization/           # Tuning guides and design documentation
│   └── user_guide/             # Per-operator guides and signatures
├── examples/                   # Runnable PyTorch, JAX, and Stackformer examples
├── src/stackkernels/
│   ├── _core/                  # Pure Triton kernels (Layer 1)
│   ├── _opt/                   # Device-aware optimization and cache (Layer 2)
│   ├── _reference/             # Pure PyTorch and JAX reference fallbacks
│   ├── torch/                  # PyTorch autograd bindings and modules (Layer 3a)
│   └── jax/                    # JAX custom_vjp bindings (Layer 3b)
├── tests/                      # Unit tests covering CPU, GPU, and optimization logic
├── mkdocs.yml                  # Material for MkDocs configuration
└── pyproject.toml              # Build specifications and optional dependencies
```

---

## Roadmap & Known Limitations

- **Hardware Validation**: Production validation is currently complete for NVIDIA Tesla T4 (sm75). Validation on Ampere (A100) and Hopper (H100) is in progress.
- **Native BF16**: Requires compute capability sm80 or newer; sm75 (T4) falls back to reference code.
- **Unimplemented Kernels**: In-place KV-cache update and generation sampling are roadmap milestones and exist currently as stubs.

---

## Contributing and License

- Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and testing workflows.
- Changes across versions are tracked in [CHANGELOG.md](CHANGELOG.md).
- Distributed under the [MIT License](LICENSE).
