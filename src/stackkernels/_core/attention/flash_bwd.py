"""Backward kernel for Flash Attention: computes dQ, dK, dV with MHA, GQA, and MQA support."""

KERNEL_REV = 1

import math
import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps
from stackkernels._opt.memory.layout import get_padded_head_dim, validate_attention_layout
from stackkernels._opt.memory.workspace import allocate_delta_buffer


@triton.jit
def _bwd_preprocess_kernel(
    O_ptr,
    dO_ptr,
    Delta_ptr,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dm,
    S_q,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute Delta = sum(dO * O, dim=-1) for backward scaling."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S_q
    mask_d = offs_d < D

    o_offset = pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    do_offset = pid_bh * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod

    o = tl.load(O_ptr + o_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(dO_ptr + do_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    delta_offset = pid_bh * stride_dh + offs_m * stride_dm
    tl.store(Delta_ptr + delta_offset, delta, mask=mask_m)


@triton.jit
def _flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    dQ_ptr,
    Lse_ptr,
    Delta_ptr,
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
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_db,
    stride_dh,
    stride_dm,
    B,
    H_q,
    H_kv,
    S_q,
    S_k,
    group_size,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACTUAL_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // H_q
    head_q = pid_bh % H_q
    head_kv = head_q // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S_q
    mask_d = offs_d < ACTUAL_D

    # Load Q and dO blocks: (BLOCK_M, BLOCK_D)
    q_offset = (
        batch_id * stride_qb
        + head_q * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    do_offset = (
        batch_id * stride_dob
        + head_q * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )

    q = tl.load(Q_ptr + q_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do = tl.load(dO_ptr + do_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load LSE and Delta: (BLOCK_M,)
    lse_offset = batch_id * stride_lseb + head_q * stride_lseh + offs_m * stride_lsem
    delta_offset = batch_id * stride_db + head_q * stride_dh + offs_m * stride_dm

    lse = tl.load(Lse_ptr + lse_offset, mask=mask_m, other=0.0)
    delta = tl.load(Delta_ptr + delta_offset, mask=mask_m, other=0.0)

    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    k_base = batch_id * stride_kb + head_kv * stride_kh
    v_base = batch_id * stride_vb + head_kv * stride_vh

    diag_shift = S_k - S_q
    if IS_CAUSAL:
        max_k_pos = (pid_m + 1) * BLOCK_M - 1 + diag_shift
        n_end = tl.minimum(S_k, max_k_pos + 1)
    else:
        n_end = S_k

    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_k

        # Load K, V: (BLOCK_N, BLOCK_D)
        k_offset = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_offset = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(K_ptr + k_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(V_ptr + v_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # qk: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if IS_CAUSAL:
            causal_allowed = offs_n[None, :] <= (offs_m[:, None] + diag_shift)
            valid = mask_m[:, None] & mask_n[None, :] & causal_allowed
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        # Softmax probability P = exp(qk - lse)
        p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)

        # dp = dO @ V^T
        dp = tl.dot(do.to(v.dtype), tl.trans(v))

        # ds = P * (dp - delta) * sm_scale
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq += ds @ K
        dq += tl.dot(ds.to(k.dtype), k)

    # Store dQ: (BLOCK_M, BLOCK_D)
    dq_offset = (
        batch_id * stride_dqb
        + head_q * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(
        dQ_ptr + dq_offset,
        dq.to(dQ_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _flash_bwd_dkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    dK_ptr,
    dV_ptr,
    Lse_ptr,
    Delta_ptr,
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
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lseb,
    stride_lseh,
    stride_lsem,
    stride_db,
    stride_dh,
    stride_dm,
    B,
    H_q,
    H_kv,
    S_q,
    S_k,
    group_size,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACTUAL_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b_hkv = tl.program_id(1)

    batch_id = pid_b_hkv // H_kv
    head_kv = pid_b_hkv % H_kv

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < S_k
    mask_d = offs_d < ACTUAL_D

    # Load K, V blocks: (BLOCK_N, BLOCK_D)
    k_offset = (
        batch_id * stride_kb
        + head_kv * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_offset = (
        batch_id * stride_vb
        + head_kv * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )

    k = tl.load(K_ptr + k_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(V_ptr + v_offset, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    diag_shift = S_k - S_q

    # For GQA/MQA, loop through all Q heads that share this KV head
    head_q_start = head_kv * group_size
    head_q_end = head_q_start + group_size

    for head_q in range(head_q_start, head_q_end):
        q_base = batch_id * stride_qb + head_q * stride_qh
        do_base = batch_id * stride_dob + head_q * stride_doh
        lse_base = batch_id * stride_lseb + head_q * stride_lseh
        delta_base = batch_id * stride_db + head_q * stride_dh

        # In causal mode, this K block only attends to queries where:
        # offs_m >= offs_n - diag_shift
        if IS_CAUSAL:
            min_m_pos = pid_n * BLOCK_N - diag_shift
            m_start_val = tl.maximum(0, min_m_pos)
        else:
            m_start_val = 0

        for m_start in range(m_start_val, S_q, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S_q

            # Load Q and dO: (BLOCK_M, BLOCK_D)
            q_offset = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            do_offset = do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod

            q = tl.load(Q_ptr + q_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            do = tl.load(dO_ptr + do_offset, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            lse = tl.load(Lse_ptr + lse_base + offs_m * stride_lsem, mask=mask_m, other=0.0)
            delta = tl.load(Delta_ptr + delta_base + offs_m * stride_dm, mask=mask_m, other=0.0)

            # qk: (BLOCK_M, BLOCK_N)
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            if IS_CAUSAL:
                causal_allowed = offs_n[None, :] <= (offs_m[:, None] + diag_shift)
                valid = mask_m[:, None] & mask_n[None, :] & causal_allowed
            else:
                valid = mask_m[:, None] & mask_n[None, :]

            # p: (BLOCK_M, BLOCK_N)
            p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)

            # dV += P^T @ dO: (BLOCK_N, BLOCK_M) x (BLOCK_M, BLOCK_D) -> (BLOCK_N, BLOCK_D)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

            # dp = dO @ V^T: (BLOCK_M, BLOCK_N)
            dp = tl.dot(do.to(v.dtype), tl.trans(v))

            # ds = P * (dp - delta) * sm_scale
            ds = p * (dp - delta[:, None]) * sm_scale

            # dK += ds^T @ Q: (BLOCK_N, BLOCK_M) x (BLOCK_M, BLOCK_D) -> (BLOCK_N, BLOCK_D)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    # Store dK and dV: (BLOCK_N, BLOCK_D)
    dk_offset = (
        batch_id * stride_dkb
        + head_kv * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd
    )
    dv_offset = (
        batch_id * stride_dvb
        + head_kv * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd
    )

    tl.store(dK_ptr + dk_offset, dk.to(dK_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dV_ptr + dv_offset, dv.to(dV_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])


def flash_attn_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton Flash Attention backward kernels.

    Returns:
        tuple (dq, dk, dv)
    """
    B, H_q, H_kv, S_q, S_k, D = validate_attention_layout(q, k, v)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    group_size = H_q // H_kv

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    delta = allocate_delta_buffer(B, H_q, S_q, q.device)

    BLOCK_D = get_padded_head_dim(D)
    BLOCK_M = 64 if S_q >= 64 else triton.next_power_of_2(S_q)
    BLOCK_M = max(min(BLOCK_M, 64), 16)
    BLOCK_N = 64 if S_k >= 64 else triton.next_power_of_2(S_k)
    BLOCK_N = max(min(BLOCK_N, 64), 16)

    # Step 1: Preprocess Delta = sum(dO * O, dim=-1)
    grid_prep = (triton.cdiv(S_q, BLOCK_M), B * H_q)
    _bwd_preprocess_kernel[grid_prep](
        out,
        do,
        delta,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        S_q,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    # Step 2: Compute dQ
    num_warps = 4 if D <= 64 else 8
    grid_dq = (triton.cdiv(S_q, BLOCK_M), B * H_q)
    _flash_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        do,
        dq,
        lse,
        delta,
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
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        B,
        H_q,
        H_kv,
        S_q,
        S_k,
        group_size,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        ACTUAL_D=D,
        num_warps=num_warps,
    )

    # Step 3: Compute dK and dV
    grid_dkv = (triton.cdiv(S_k, BLOCK_N), B * H_kv)
    _flash_bwd_dkv_kernel[grid_dkv](
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse,
        delta,
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
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        B,
        H_q,
        H_kv,
        S_q,
        S_k,
        group_size,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        ACTUAL_D=D,
        num_warps=num_warps,
    )

    return dq, dk, dv


__all__ = ["flash_attn_bwd", "_flash_bwd_dq_kernel", "_flash_bwd_dkv_kernel"]
