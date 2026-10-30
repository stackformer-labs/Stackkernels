# RMSNorm Optimization Notes

## 1. Roofline & Arithmetic Intensity

Root Mean Square Normalization (RMSNorm) computes:
$$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}$$
$$y_i = \frac{x_i}{\text{RMS}(x)} \cdot w_i$$

### Compute & Memory Operations (Forward)
For a tensor of shape $(M, N)$:
- **Floating Point Operations (FLOPs)**:
  - $M \times N$ multiplications ($x_i^2$)
  - $M \times (N - 1)$ additions (sum reduction)
  - $1$ division + $1$ square root + $1$ reciprocal per row ($3 M$)
  - $M \times N$ multiplications ($x_i \cdot \text{rstd}$)
  - $M \times N$ multiplications ($\text{normed} \cdot w_i$)
  - **Total FLOPs** $\approx 4 M N$
- **Memory Traffic (Bytes Moved)**:
  - Read input $X$: $M \times N \times \text{elem\_size}$
  - Write output $Y$: $M \times N \times \text{elem\_size}$
  - Write saved $R_{\text{std}}$: $M \times 4$ (fp32)
  - Read weight $W$: $N \times \text{elem\_size}$
  - **Total Bytes** $\approx 2 M N \times \text{elem\_size} + 4 M + N \times \text{elem\_size}$

### Arithmetic Intensity
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Moved}} \approx \frac{4 M N}{2 M N \times 2} \approx 1.0 \text{ FLOP/byte (in fp16/bf16)}$$
Because modern GPUs deliver over 100 TFLOPs but have $\sim 1\text{–}3 \text{ TB/s}$ memory bandwidth, RMSNorm is **heavily memory-bandwidth bound**. The goal of the Triton kernel is to maximize achieved memory bandwidth (GB/s) towards theoretical peak.

---

## 2. Backward Pass: Multi-Pass vs. Single Fused Pass

### The Issue with Multi-Pass Backward
In a naive backward implementation, two separate kernels are used:
1. **Pass 1 (`dx` kernel)**: Reads $X$, $dY$, and $R_{\text{std}}$ from High Bandwidth Memory (HBM) to compute and write $dX$.
2. **Pass 2 (`dw` kernel)**: Reads $X$, $dY$, and $R_{\text{std}}$ **a second time** from HBM to compute and write $dW$.

#### Memory Bottleneck:
- For a sequence with $M \times N = 2048 \times 4096 = 8.38 \text{M}$ elements in fp16 ($16.8 \text{ MB}$):
  - Reading $X$ twice = $33.6 \text{ MB}$ transferred.
  - Reading $dY$ twice = $33.6 \text{ MB}$ transferred.
  - Total memory traffic is nearly **doubled** ($2\times$) simply re-reading the exact same inputs from DRAM across the PCIe/memory bus.
- In memory-bound operations, doubling memory traffic halves your kernel's effective bandwidth and throughput.

### The Single-Pass Fused Backward Solution
In our single-pass backward kernel (`_rms_norm_bwd_kernel`):
- Each program/thread block loads its assigned row of $X$, $dY$, and $R_{\text{std}}$ **once** into fast on-chip registers/SRAM.
- Using these register-resident values, the thread block computes:
  1. $\hat{x} = x \cdot R_{\text{std}}$
  2. $dx = R_{\text{std}} \cdot \left( dy \cdot w - \frac{\hat{x}}{N} \cdot \sum (dy \cdot w \cdot \hat{x}) \right)$
  3. $dw$ gradient contribution ($dy \cdot \hat{x}$)
- The kernel writes $dx$ back to HBM and accumulates $dw$ directly using `tl.atomic_add` on the $W$ vector in the same pass.
- **Result**: Zero redundant global memory loads; memory traffic drops from $\approx 5 M N$ bytes to $\approx 3 M N$ bytes.

---

## 3. Kernel Optimization Techniques

1. **Row-Parallel Execution**:
   - Each Triton block processes a full row of $N$ columns.
   - Preserves locality and eliminates inter-block synchronization for row-level reductions.

2. **2D Matrix Flattening**:
   - Flattening arbitrary multidimensional shapes $(B, S, D)$ into $(M, D)$ ensures clean sequential strides.

3. **Power-of-2 Column Padding**:
   - `BLOCK_SIZE = triton.next_power_of_2(N)`.
   - Bounds-checked with column masks (`col_offsets < N`).

4. **Float32 Accumulation**:
   - Sum of squares, variance, reciprocal standard deviation, and intermediate gradient projections are computed in `fp32` to eliminate underflow/overflow risks when activations are in `float16` or `bfloat16`.

5. **Saving $R_{\text{std}}$ for Backward**:
   - Saving the 1D tensor of reciprocal standard deviations ($M \times 4$ bytes) avoids recomputing $\text{mean}(x^2)$ during backward pass, saving an entire forward pass computation.

---

## 4. Persistent Grid-Stride Forward Variant

### Motivation
In standard row-parallel execution, a separate thread block is scheduled for each row $m \in [0, M-1]$. When $M$ is very large (e.g., $M > 4 \times \text{SM count}$) and row length $N$ is small to moderate ($N \le 2048$), launching thousands of transient thread blocks incurs hardware thread block scheduling overhead and suffers from tail-wave imbalance.

### Implementation
The persistent forward kernel (`_rms_norm_fwd_persistent_kernel`) launches a fixed grid of $2 \times \text{SM count}$ thread blocks. Each persistent block:
1. Stays resident on its allocated Streaming Multiprocessor (SM).
2. Preloads the weight vector $W$ once outside the row processing loop into fast registers/SRAM.
3. Processes multiple rows via grid-stride iteration (`while row_idx < M: ... row_idx += num_programs`).

### Device-Aware Selection Rule
The optimizer's `pick()` automatically selects `{"variant": "persistent"}` when:
- Row count $M > 4 \times \text{SM count}$
- Column dimension $N \le 2048$

### Hardware Performance Assessment
- **CPU / Local Environment**: Evaluated functionally in CPU simulation; verified zero discrepancy in numerical normalization and variance calculation.
- **Cloud T4 GPU Validation**: *Pending user validation on cloud T4 GPU instance*. Persistent threads amortize block dispatch latency across large batch dimensions ($B \cdot S \ge 2048$) while keeping SM occupancy uniformly high.
