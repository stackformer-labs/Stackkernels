"""Prefill forward kernel: MHA/GQA/MQA, causal (bottom-right aligned) or non-causal, saves LSE."""

import math
import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps
from stackkernels._opt.memory.layout import get_padded_head_dim, validate_attention_layout
from stackkernels._opt.memory.workspace import allocate_lse_buffer


@triton.jit
def _flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Lse_ptr,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lseb,
    stride_lseh,
    stride_lsem,
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

    # Row offsets for Q block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Base pointers
    q_offset = (
        batch_id * stride_qb
        + head_q * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    k_base = batch_id * stride_kb + head_kv * stride_kh
    v_base = batch_id * stride_vb + head_kv * stride_vh

    mask_m = offs_m < S_q
    mask_d = offs_d < ACTUAL_D

    # Load Q block: (BLOCK_M, BLOCK_D)
    q = tl.load(
        Q_ptr + q_offset,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    # Initialize online softmax stats
    m_val = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Causal sequence offset difference for bottom-right alignment: S_k - S_q
    diag_shift = S_k - S_q

    # Determine maximum key position for this query block
    if IS_CAUSAL:
        max_k_pos = (pid_m + 1) * BLOCK_M - 1 + diag_shift
        n_end = tl.minimum(S_k, max_k_pos + 1)
    else:
        n_end = S_k

    # Loop over KV sequence blocks
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_k

        # Load K block: (BLOCK_N, BLOCK_D)
        k_offset = (
            k_base
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(
            K_ptr + k_offset,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        # Compute scores: (BLOCK_M, BLOCK_D) x (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # Using transposed K
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply causal mask (bottom-right aligned) and bounds check
        if IS_CAUSAL:
            causal_allowed = offs_n[None, :] <= (offs_m[:, None] + diag_shift)
            valid = mask_m[:, None] & mask_n[None, :] & causal_allowed
        else:
            valid = mask_m[:, None] & mask_n[None, :]

        qk = tl.where(valid, qk, -float("inf"))

        # Online softmax update
        local_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_val, local_max)

        scale = tl.exp(m_val - new_m)
        p = tl.exp(qk - new_m[:, None])

        l_val = l_val * scale + tl.sum(p, axis=1)
        acc = acc * scale[:, None]

        # Load V block: (BLOCK_N, BLOCK_D)
        v_offset = (
            v_base
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(
            V_ptr + v_offset,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        acc += tl.dot(p.to(v.dtype), v)
        m_val = new_m

    # Final normalization
    l_safe = tl.where(l_val == 0.0, 1.0, l_val)
    out = acc / l_safe[:, None]
    lse = m_val + tl.log(l_safe)

    # Store output: (BLOCK_M, BLOCK_D)
    o_offset = (
        batch_id * stride_ob
        + head_q * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(
        O_ptr + o_offset,
        out.to(O_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )

    # Store LSE: (BLOCK_M,)
    lse_offset = (
        batch_id * stride_lseb
        + head_q * stride_lseh
        + offs_m * stride_lsem
    )
    tl.store(Lse_ptr + lse_offset, lse, mask=mask_m)


def flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Triton Flash Attention forward kernel.

    Supports MHA, GQA, MQA, arbitrary head_dim up to 256, and bottom-right causal masking.

    Args:
        q: Query tensor of shape (B, H_q, S_q, D).
        k: Key tensor of shape (B, H_kv, S_k, D).
        v: Value tensor of shape (B, H_kv, S_k, D).
        is_causal: Whether to apply bottom-right causal mask.
        sm_scale: Scaling factor (default: 1 / sqrt(D)).

    Returns:
        tuple (output of shape (B, H_q, S_q, D), lse of shape (B, H_q, S_q))
    """
    B, H_q, H_kv, S_q, S_k, D = validate_attention_layout(q, k, v)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    group_size = H_q // H_kv

    # Allocate output and LSE buffers
    out = torch.empty_like(q)
    lse = allocate_lse_buffer(B, H_q, S_q, q.device)

    # Determine tile sizes
    BLOCK_D = get_padded_head_dim(D)
    BLOCK_M = 64 if S_q >= 64 else triton.next_power_of_2(S_q)
    BLOCK_M = max(min(BLOCK_M, 128), 16)
    BLOCK_N = 64 if S_k >= 64 else triton.next_power_of_2(S_k)
    BLOCK_N = max(min(BLOCK_N, 64), 16)

    grid = (triton.cdiv(S_q, BLOCK_M), B * H_q)
    num_warps = 4 if D <= 64 else 8

    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
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

    return out, lse


__all__ = ["flash_attn_fwd", "_flash_fwd_kernel"]
