# Optimization Layer Design

This document details the architecture, design principles, and contracts of the StackKernels optimization layer (`src/stackkernels/_opt/`).

---

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

## 2. End-to-End Optimization Pipeline

The optimization workflow makes decisions based on the target GPU architecture and compiler feedback:

```text
  ┌────────────────┐
  │ DeviceProfile  │  (Hardware facts: SM count, regs/SM, max smem, max threads, cc)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │  Seed Rules    │  (Kernel classes: row, elementwise, tiled, decode)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │ Search Space   │  (Generates candidate neighbors around seeds, capped at 12)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │ Constraints    │  (Static pruning: smem 10% safety margin, acc regs, stages)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │ Compiler Probe │  (Extracts n_regs, n_spills, smem; rejects spills)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │ GPU Benchmark  │  (2% tie-break toward fewer registers / higher occupancy)
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │  Tuning Cache  │  (6-tuple key: op, fingerprint, triton_ver, kernel_rev, dtype, bucket)
  └────────────────┘
```

1. **Hardware Resolution (`hardware.py`)**:
   - Queries `torch.cuda.get_device_properties`, Triton driver properties, or CC fallback tables.
   - Produces a frozen `DeviceProfile` with SM counts, max threads, shared memory, and capabilities (`has_cp_async`, `has_bf16`).
   - Generates a stable hardware `fingerprint()` string (e.g., `tesla-t4_sm75_40sm_16384mb`).

2. **Class Seed Rules (`_opt/classes/`)**:
   - Pure Python rules that generate ordered candidate configurations from `(DeviceProfile, problem, dtype)`.
   - The first candidate is always the seed heuristic.

3. **Candidate Neighbors (`search_spaces.py`)**:
   - Varies tile dimensions, warp counts, and pipeline stages around the primary seed.
   - Limits search space to at most 12 valid candidates.

4. **Static Constraint Checking (`constraints.py`)**:
   - Validates candidates before any compilation or execution.
   - Enforces thread counts, architecture stages, shared memory limits, and accumulator budgets.

5. **Compiler Probing (`occupancy.py`)**:
   - Uses a single adapter function `_safe_compile_probe` (`kernel.warmup`) to extract register count, spill count, and shared memory.
   - Any configuration causing register spills (`n_spills > 0`) is rejected.

6. **Benchmarking & Tie-Breaking (`tune.py`)**:
   - Measures median kernel latency.
   - Candidates within 2% of the fastest latency are tie-broken in favor of fewer registers and higher SM occupancy.

7. **Caching & Persistence (`cache.py`)**:
   - Records the selected configuration and detailed metadata (latency, GB/s, % of peak, compiler metrics) into disk cache.

---

## 3. The 4 Kernel Classes and Seed Rules

### 1. Row Class (`_opt/classes/row.py`)
- **Operations**: `rms_norm`, `layer_norm`, `fused_add_norm`, `cross_entropy`, `rope`, `topk_topp`.
- **Seed Rule**:
  - Sets `BLOCK_SIZE = next_power_of_2(cols)`.
  - Warps scale with block size: 1 warp ($\le 64$), 2 warps ($128$), 4 warps ($512$), 8 warps ($2048$), 16 warps ($>2048$ if supported).
  - When $cols$ exceeds the single-pass register limit (4096 on sm75, 8192 on sm80+), chunked execution is activated.

### 2. Elementwise Class (`_opt/classes/elementwise.py`)
- **Operations**: `elementwise` (GELU, SiLU, ReLU, etc.), `glu` (SwiGLU, GeGLU), `dropout`, `kv_update`.
- **Seed Rule**:
  - Memory-bound 1D streaming kernels.
  - Defaults to large tiles (`BLOCK_SIZE = 1024`, `num_warps = 4` for sm80+; `BLOCK_SIZE = 512`, `num_warps = 2` for sm75 or small vectors).
  - Maximizes global memory bus saturation.

### 3. Tiled Class (`_opt/classes/tiled.py`)
- **Operations**: `flash_fwd`, `flash_bwd`, `fused_linear_ce`.
- **Seed Rule**:
  - Dynamically sized 2D tiles $(BLOCK\_M, BLOCK\_N)$ balancing SRAM consumption and accumulator register file usage.
  - **sm80+ (Ampere/Hopper)**:
    - $D \le 64$: `BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2` (or 3 on Hopper).
    - $D = 128$: `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2`.
    - $D \ge 256$: `BLOCK_M=64, BLOCK_N=32, num_warps=8, num_stages=2`.
  - **sm75 (Turing / T4)**:
    - Max 48KB shared memory per block.
    - $D \le 64$: `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1`.
    - $D = 128$: `BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=1`.
    - $D = 256$: `BLOCK_M=32, BLOCK_N=16, num_warps=4, num_stages=1`.

### 4. Decode Class (`_opt/classes/decode.py`)
- **Operations**: `decode`, `flash_decode`.
- **Seed Rule**:
  - Dedicated Split-KV parallelization for single query token generation.
  - Dynamically scales `SPLIT_KV` based on SM count and KV cache sequence length.
  - Tile size `BLOCK_N = 64` (or 32 for very short contexts), `BLOCK_M = 16`, `num_warps = 4`.

---

## 4. Hardware Constraints and Pruning Logic

Static constraints defined in `constraints.py` prune invalid configurations before running or probing:

1. **Native Dtype Support**:
   - `bfloat16` is rejected on compute capability $< (8, 0)$ (e.g. Turing / T4) where native hardware support is absent.

2. **Hardware Thread Limits**:
   - Blocks with $> 1024$ threads ($num\_warps > 32$) or exceeding `profile.max_threads_per_sm` are rejected.

3. **Pipeline Stages (`num_stages`)**:
   - `num_stages > 2` requires hardware asynchronous memory copy (`cp.async`, sm80+).
   - `num_stages > 4` is restricted to Hopper (sm90+) architectures.

4. **Shared Memory Safety Margin (10%)**:
   - Tile shared memory consumption is estimated as:
     $$\text{Estimated Smem} = \text{stages} \times (BLOCK\_M \cdot D + 2 \cdot BLOCK\_N \cdot D) \times \text{itemsize}$$
   - Must not exceed $90\%$ of `profile.max_smem_per_block`. The 10% safety margin prevents driver launch errors caused by internal compiler alignments and scalar arguments.

5. **Accumulator Register Budget**:
   - In GEMM/attention tiles, accumulators are distributed across active threads:
     $$\text{Acc Regs / Thread} = \left\lceil \frac{BLOCK\_M \times BLOCK\_N}{32 \times \text{num\_warps}} \right\rceil$$
   - Must not exceed $45\%$ of `profile.max_regs_per_thread` (114 regs on 255-reg GPUs), leaving $\ge 55\%$ for address arithmetic, inputs, and compiler temporaries.

6. **Tile vs. Problem Size Bounds**:
   - Row block size cannot exceed 16384.
   - Tiles cannot be $> 8\times$ larger than the actual problem dimension (preventing excessive padding overhead).

---

## 5. Cache Key & Invalidation Policy

Configurations are saved with a 6-tuple canonical cache key:

```python
CacheKey = tuple[
    str,  # op: op name (e.g. "rms_norm")
    str,  # device_fingerprint: hardware fingerprint (e.g. "tesla-t4_sm75_40sm_16384mb")
    str,  # triton_version: active Triton version (e.g. "3.7.1")
    int,  # kernel_rev: integer revision from _core module
    str,  # dtype: data type string (e.g. "float16")
    str,  # shape_bucket: canonical bucket string (e.g. "2048" or "1024x4096")
]
```

### Invalidation Policy
- **Triton Compiler Version**: Any update or change in the Triton compiler version cleanly misses existing cache entries, prompting fresh compilation and tuning.
- **Kernel Revision (`KERNEL_REV`)**: Modifying kernel math or memory layouts increments `KERNEL_REV` in `_core/<category>/<op>.py`, cleanly invalidating previously cached configurations.
- **Device Fingerprint**: Completely isolates configurations between different GPU architectures (e.g. T4 vs. A100 vs. H100).

---

## 6. Resolution Order

When `pick()` is invoked, configurations resolve in strict priority:

1. **User Override**: Explicit dictionary passed to `pick(..., override={...})`.
2. **User Cache**: Tuned configuration persisted in `~/.cache/stackkernels/user_cache.json`.
3. **Shipped Tuned Defaults**: Shipped configurations matching device fingerprint in `_opt/tuned/`.
4. **Class Heuristic**: Dynamic seed rule evaluated using `DeviceProfile`.

### Explanation Output (`pick(..., explain=True)`)
Returns a tuple `(config, explanation_str)` documenting:
- Decision source (`user_override`, `user_cache`, `shipped_tuned`, `heuristic`).
- Target device facts (SM count, max threads, max registers).
- Wave and occupancy math (grid blocks, wave count, programs per SM).
- Selected configuration knobs.
- Pruned/rejected candidate configurations and specific constraint rejection reasons.

---

## 7. Checklist for Adding a New Op

Follow this contract when adding a new operator:

- [ ] **1. Kernel Implementation**: Implement Triton kernel in `src/stackkernels/_core/<category>/<op>.py`. Set module-level integer `KERNEL_REV = 1`. Do NOT import `_opt`, `torch`, or `jax` inside `_core`.
- [ ] **2. Module Registration**: Register op name and module path in `src/stackkernels/_opt/tune.py` in `OP_TO_MODULE`.
- [ ] **3. Kernel Class Assignment**: Map op to its execution class (`row`, `elementwise`, `tiled`, `decode`) in `src/stackkernels/_opt/search_spaces.py` `OP_TO_CLASS`.
- [ ] **4. Heuristic Wrapper**: Add wrapper in `src/stackkernels/_opt/heuristics/<op>.py` delegating to the appropriate class in `_opt/classes/`.
- [ ] **5. Framework Wrappers**:
  - Implement PyTorch wrapper in `src/stackkernels/torch/<category>.py` calling `pick(op_name, shape=..., dtype=...)`.
  - Implement JAX wrapper in `src/stackkernels/jax/<category>.py`.
- [ ] **6. Unit Tests**:
  - Add CPU heuristic test in `tests/test_opt/test_heuristics.py`.
  - Add search space test in `tests/test_opt/test_search_spaces.py`.
  - Add forward/backward numerical tests in `tests/ops/test_<op>.py` with CUDA skip fixtures.
