# Optimization Layer Design

This document details the architecture and contract of the StackKernels optimization layer (`src/stackkernels/_opt/`).

## 1. Kernel Inventory & Class Assignment

Every kernel in `_core` maps to one of four kernel execution classes based on its memory access pattern and thread grid structure.

| Op / Kernel | Kernel Class | Actual Tuning Knobs Accepted | Description |
|---|---|---|---|
| `rms_norm` | `row` | `BLOCK_SIZE`, `num_warps`, `variant` | Normalizes across the inner dimension (row). Single-pass or chunked loop. |
| `layer_norm` | `row` | `BLOCK_SIZE`, `num_warps` | Normalizes across hidden dimension with mean and variance reductions. |
| `fused_add_norm` | `row` | `BLOCK_SIZE`, `num_warps` | Fuses residual addition with RMS or LayerNorm row reduction. |
| `cross_entropy` | `row` | `BLOCK_V`, `BLOCK_SIZE`, `num_warps` | Online-softmax reduction per token across the vocabulary row. |
| `rope` | `row` | `BLOCK_D_HALF`, `num_warps` | Rotary embedding rotation per token-head row. |
| `topk_topp` | `row` | `BLOCK_SIZE`, `num_warps` | Sampling reductions across vocabulary row per token. |
| `elementwise` | `elementwise` | `BLOCK_SIZE`, `num_warps` | Streaming activations (GELU, SiLU, ReLU, LeakyReLU, Sigmoid). |
| `glu` | `elementwise` | `BLOCK_SIZE`, `num_warps` | Gated activations (SwiGLU, GeGLU, ReGLU) elementwise fusion. |
| `dropout` / `fused_dropout_add` | `elementwise` | `BLOCK_SIZE`, `num_warps` | Random bit-masking and streaming addition. |
| `kv_update` | `elementwise` | `BLOCK_SIZE`, `num_warps` | KV-cache slice streaming updates. |
| `flash_fwd` | `tiled` | `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` | 2D tiled attention with online-softmax in SRAM. |
| `flash_bwd` | `tiled` | `BLOCK_M`, `BLOCK_N`, `num_warps` | 2D tiled attention backward for dQ and dK/dV. |
| `fused_linear_ce` | `tiled` | `CHUNK_SIZE`, `BLOCK_M`, `BLOCK_V`, `BLOCK_K`, `num_warps` | 2D tiled GEMM fused with vocabulary online softmax. |
| `flash_decode` | `decode` | `SPLIT_KV` (num_splits), `BLOCK_N`, `num_warps` | Split-KV parallelization for single query token over long context. |

---

## 2. Optimization Pipeline Overview

```text
DeviceProfile ──► seed configs ──► prune ──► probe-compile ──► benchmark ──► cache
(torch/triton)    (kernel-class     (static    (read regs,       (few        (keyed by
                   rules)            limits)    spills, smem)     survivors)  device+triton)
```

1. **DeviceProfile**: Hardware facts queried dynamically (compute capability, SM count, max registers per SM, shared memory, max threads per SM).
2. **Seed configs**: Class rules turn `(DeviceProfile, problem, dtype)` into candidate configurations (first seed is the heuristic default).
3. **Prune**: Static validation against hardware constraints (shared memory limits, thread limits, accumulator register budgets).
4. **Probe-compile**: Non-executing compilation via single guarded adapter to inspect register usage, spills, and shared memory.
5. **Benchmark**: Benchmarks surviving candidates with early stopping.
6. **Cache**: Persists best configuration keyed by `(op, device_fingerprint, triton_version, kernel_rev, dtype, shape_bucket)`.
