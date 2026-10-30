"""Split-KV FlashDecode kernel for q_len = 1 with arbitrary cache lengths."""

KERNEL_REV = 1

import math
import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics.flash_decode import heuristic as decode_heuristic
from stackkernels._opt.memory.layout import get_padded_head_dim, validate_attention_layout
from stackkernels._opt.memory.workspace import allocate_split_kv_buffers


@triton.jit
def _flash_decode_split_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    PartialOut_ptr,
    PartialLse_ptr,
    PartialMax_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_plsb,
    stride_plsh,
    stride_plss,
    stride_pmb,
    stride_pmh,
    stride_pms,
    B,
    H_q,
    H_kv,
    S_k,
    group_size,
    chunk_size,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACTUAL_D: tl.constexpr,
):
    split_id = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // H_q
    head_q = pid_bh % H_q
    head_kv = head_q // group_size

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < ACTUAL_D

    # Load 1 query vector: (BLOCK_D,)
    q_offset = batch_id * stride_qb + head_q * stride_qh + offs_d * stride_qd
    q = tl.load(Q_ptr + q_offset, mask=mask_d, other=0.0)

    # Slice range for this split
    n_start_split = split_id * chunk_size
    n_end_split = tl.minimum(n_start_split + chunk_size, S_k)

    m_val = -float("inf")
    l_val = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    k_base = batch_id * stride_kb + head_kv * stride_kh
    v_base = batch_id * stride_vb + head_kv * stride_vh

    for n_start in range(n_start_split, n_end_split, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_end_split

        k_offset = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(K_ptr + k_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # qk: (BLOCK_N,)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(mask_n, qk, -float("inf"))

        local_max = tl.max(qk, axis=0)
        new_m = tl.maximum(m_val, local_max)

        scale = tl.exp(m_val - new_m)
        p = tl.exp(qk - new_m)

        l_val = l_val * scale + tl.sum(p, axis=0)
        acc = acc * scale

        v_offset = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(V_ptr + v_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # acc += P @ V
        acc += tl.sum(p[:, None] * v, axis=0)
        m_val = new_m

    # Store partial output and stats for this split
    po_offset = (
        batch_id * stride_pob
        + head_q * stride_poh
        + split_id * stride_pos
        + offs_d * stride_pod
    )
    tl.store(PartialOut_ptr + po_offset, acc, mask=mask_d)

    pls_offset = batch_id * stride_plsb + head_q * stride_plsh + split_id * stride_plss
    pm_offset = batch_id * stride_pmb + head_q * stride_pmh + split_id * stride_pms

    tl.store(PartialLse_ptr + pls_offset, l_val)
    tl.store(PartialMax_ptr + pm_offset, m_val)


@triton.jit
def _flash_decode_reduce_kernel(
    PartialOut_ptr,
    PartialLse_ptr,
    PartialMax_ptr,
    O_ptr,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_plsb,
    stride_plsh,
    stride_plss,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B,
    H_q,
    SPLIT_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACTUAL_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // H_q
    head_q = pid_bh % H_q

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < ACTUAL_D

    offs_s = tl.arange(0, SPLIT_KV)

    # Load partial max and sum of exp across splits
    pm_offset = batch_id * stride_pmb + head_q * stride_pmh + offs_s * stride_pms
    pls_offset = batch_id * stride_plsb + head_q * stride_plsh + offs_s * stride_plss

    partial_max = tl.load(PartialMax_ptr + pm_offset)
    partial_lse = tl.load(PartialLse_ptr + pls_offset)

    global_max = tl.max(partial_max, axis=0)

    # Weight per split: l_val * exp(m_val - global_max)
    weights = partial_lse * tl.exp(partial_max - global_max)
    total_weight = tl.sum(weights, axis=0)
    total_safe = tl.where(total_weight == 0.0, 1.0, total_weight)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for s in range(0, SPLIT_KV):
        po_offset = (
            batch_id * stride_pob
            + head_q * stride_poh
            + s * stride_pos
            + offs_d * stride_pod
        )
        part_out = tl.load(PartialOut_ptr + po_offset, mask=mask_d, other=0.0)
        w = tl.exp(tl.load(PartialMax_ptr + batch_id * stride_pmb + head_q * stride_pmh + s * stride_pms) - global_max)
        acc += part_out * w

    out = acc / total_safe

    o_offset = batch_id * stride_ob + head_q * stride_oh + offs_d * stride_od
    tl.store(O_ptr + o_offset, out.to(O_ptr.dtype.element_ty), mask=mask_d)


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Launch Split-KV FlashDecode for q_len = 1.

    Supports MHA, GQA, MQA, and arbitrary KV lengths.

    Args:
        q: Query tensor of shape (B, H_q, 1, D).
        k: Key tensor of shape (B, H_kv, S_k, D).
        v: Value tensor of shape (B, H_kv, S_k, D).
        sm_scale: Scaling factor (default: 1 / sqrt(D)).

    Returns:
        Output tensor of shape (B, H_q, 1, D).
    """
    B, H_q, H_kv, S_q, S_k, D = validate_attention_layout(q, k, v)
    assert S_q == 1, f"flash_decode requires q_len == 1, got {S_q}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    group_size = H_q // H_kv

    # Get split-kv count from heuristic
    config = decode_heuristic(shape=q.shape, kv_len=S_k)
    split_kv = config["SPLIT_KV"]
    if S_k < 128:
        split_kv = 1
    split_kv = min(split_kv, max(1, S_k // 64))

    chunk_size = math.ceil(S_k / split_kv)

    BLOCK_D = get_padded_head_dim(D)
    BLOCK_N = 64 if S_k >= 64 else triton.next_power_of_2(S_k)
    BLOCK_N = max(min(BLOCK_N, 64), 16)

    partial_out = torch.empty((B, H_q, split_kv, BLOCK_D), dtype=torch.float32, device=q.device)
    partial_lse = torch.empty((B, H_q, split_kv), dtype=torch.float32, device=q.device)
    partial_max = torch.empty((B, H_q, split_kv), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)

    # Launch split kernel
    grid_split = (split_kv, B * H_q)
    _flash_decode_split_kernel[grid_split](
        q,
        k,
        v,
        partial_out,
        partial_lse,
        partial_max,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        B,
        H_q,
        H_kv,
        S_k,
        group_size,
        chunk_size,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        ACTUAL_D=D,
        num_warps=4,
    )

    # Launch reduce kernel
    grid_reduce = (B * H_q,)
    # Round split_kv to next power of 2 for reduction kernel constexpr
    SPLIT_KV_CONST = triton.next_power_of_2(split_kv)
    _flash_decode_reduce_kernel[grid_reduce](
        partial_out,
        partial_lse,
        partial_max,
        out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H_q,
        SPLIT_KV=split_kv,
        BLOCK_D=BLOCK_D,
        ACTUAL_D=D,
        num_warps=2,
    )

    return out


__all__ = ["flash_decode", "_flash_decode_split_kernel", "_flash_decode_reduce_kernel"]
