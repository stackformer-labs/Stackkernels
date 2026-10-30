# Flash Attention Optimization

This document outlines the architecture, kernel design, tiling strategies, and memory efficiency models for `StackKernels` Flash Attention (prefill forward & backward) and FlashDecode.

---

## 1. Background & Memory Bottleneck

Standard scaled dot-product attention computes:
$$S = \frac{Q K^T}{\sqrt{D}}, \quad P = \text{softmax}(S), \quad O = P V$$

In standard PyTorch, intermediate matrices $S \in \mathbb{R}^{B \times H \times S_q \times S_k}$ and $P \in \mathbb{R}^{B \times H \times S_q \times S_k}$ are materialized in High-Bandwidth Memory (HBM).
For a context length of $S = 4096$, $B = 2$, $H = 32$, in `bfloat16`:
$$\text{Memory} = 2 \times 32 \times 4096 \times 4096 \times 2 \text{ bytes} \approx 2.14 \text{ GB}$$
Reading and writing these $O(S^2)$ intermediate matrices saturates the GPU memory bus, making attention severely memory-bandwidth bound.

---

## 2. Flash Attention Forward: Tiling & Online Softmax

StackKernels computes attention block by block in GPU SRAM using the **Online Softmax** algorithm.

### 2.1 Online Softmax Formulation
For query block $Q_m \in \mathbb{R}^{B_M \times D}$ and key/value blocks $K_n, V_n \in \mathbb{R}^{B_N \times D}$:

1. **Compute tile dot product**:
   $$S_{m, n} = (Q_m K_n^T) \times \frac{1}{\sqrt{D}}$$

2. **Update running max**:
   $$\tilde{m}_{\text{new}} = \max(\tilde{m}_{\text{old}}, \max_n(S_{m, n}))$$

3. **Compute scale and tile exponential**:
   $$\alpha = \exp(\tilde{m}_{\text{old}} - \tilde{m}_{\text{new}}), \quad P_{m, n} = \exp(S_{m, n} - \tilde{m}_{\text{new}})$$

4. **Update running sum of exponentials**:
   $$\tilde{l}_{\text{new}} = \tilde{l}_{\text{old}} \cdot \alpha + \sum_n P_{m, n}$$

5. **Accumulate output**:
   $$O_m = O_m \cdot \alpha + P_{m, n} V_n$$

6. **Final normalization**:
   $$O_m = \frac{O_m}{\tilde{l}_{\text{final}}}, \quad \text{LSE}_m = \tilde{m}_{\text{final}} + \log(\tilde{l}_{\text{final}})$$

$S$ and $P$ are never written to HBM; memory traffic is reduced from $O(S^2)$ to $O(S \cdot D)$.

---

## 3. Unified MHA, GQA, and MQA Support

In StackKernels, Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA) are unified with zero memory copying:

- Query heads: $H_q$
- Key/Value heads: $H_{kv}$
- Head group ratio: $\text{group\_size} = H_q / H_{kv}$

| Architecture | $H_q$ | $H_{kv}$ | $\text{group\_size}$ | KV Head Mapping |
|---|---|---|---|---|
| **MHA** | 32 | 32 | 1 | $h_{kv} = h_q$ |
| **GQA** | 32 | 8 | 4 | $h_{kv} = h_q // 4$ |
| **MQA** | 32 | 1 | 32 | $h_{kv} = 0$ |

Inside the Triton kernel, query head $h_q$ loads its corresponding KV block directly using:
```python
head_kv = head_q // group_size
k_offset = batch_id * stride_kb + head_kv * stride_kh + ...
```
No repeat/interleave allocations are made in GPU memory.

---

## 4. Bottom-Right Causal Mask Alignment

When query length $S_q$ and key length $S_k$ differ (e.g. prefix LM, cross-attention, chunked prefill):
- **Diagonal shift**: $\delta = S_k - S_q$
- **Causal condition**: Query $i \in [0, S_q)$ can attend to key $j \in [0, S_k)$ if and only if:
  $$j \le i + (S_k - S_q)$$

This aligns the causal boundary to the bottom-right corner of the attention matrix, guaranteeing that the newest query token attends to all available historical keys.

---

## 5. Backward Pass: $\Delta$ Scaling & Gradient Fusing

The backward pass computes gradients $dQ, dK, dV$ without ever storing the $O(S^2)$ attention matrix $P$:

1. **Preprocessing Pass**:
   Computes scaling vector $\Delta \in \mathbb{R}^{B \times H_q \times S_q}$:
   $$\Delta_i = \sum_{d} dO_{i, d} \cdot O_{i, d}$$

2. **$dQ$ Kernel**:
   Tiles over queries $M$. Loops over $K, V$ blocks, recomputing:
   $$P_{ij} = \exp(S_{ij} - \text{LSE}_i)$$
   $$dP_{ij} = dO_i \cdot V_j^T$$
   $$dS_{ij} = P_{ij} \cdot (dP_{ij} - \Delta_i) \cdot \frac{1}{\sqrt{D}}$$
   $$dQ_i = \sum_j dS_{ij} K_j$$

3. **$dK, dV$ Kernel**:
   Tiles over keys/values $N$. Loops over queries $M$ and loops across all query heads $h_q$ in the GQA/MQA group:
   $$dV_j = \sum_{h \in \text{group}} \sum_i P_{ij}^T \cdot dO_i$$
   $$dK_j = \sum_{h \in \text{group}} \sum_i dS_{ij}^T \cdot Q_i$$

---

## 6. FlashDecode: Split-KV for Single-Token Generation

When generating tokens during LLM inference ($S_q = 1$ and $S_k \gg 1$), parallelizing across batch and heads alone leaves GPU Streaming Multiprocessors (SMs) severely underutilized.

### Split-KV Architecture
1. **Split Phase**:
   Divides the sequence $S_k$ into `SPLIT_KV` chunks.
   Grid size is `(SPLIT_KV, B * H_q)`.
   Each thread block computes partial output:
   $$\text{partial\_out} \in \mathbb{R}^{B \times H_q \times \text{SPLIT\_KV} \times D}, \quad \text{partial\_lse} \in \mathbb{R}^{B \times H_q \times \text{SPLIT\_KV}}$$

2. **Reduction Phase**:
   A single lightweight reduction kernel combines the splits:
   $$m_{\text{global}} = \max_s(\text{partial\_max}_s)$$
   $$w_s = \text{partial\_lse}_s \cdot \exp(\text{partial\_max}_s - m_{\text{global}})$$
   $$O = \frac{\sum_s w_s \cdot \text{partial\_out}_s}{\sum_s w_s}$$

This parallelizes token generation across 100+ GPU SMs, achieving near-linear speedups for long-context inference.
