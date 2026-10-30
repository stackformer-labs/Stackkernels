"""Cross entropy loss forward and single-pass backward kernels in Triton."""

KERNEL_REV = 1

from typing import Literal

import torch
import triton
import triton.language as tl

from stackkernels._opt.heuristics._util import determine_num_warps


@triton.jit
def _cross_entropy_fwd_kernel(
    Logits_ptr,
    Target_ptr,
    Loss_ptr,
    Lse_ptr,
    stride_logits_m,
    stride_logits_v,
    M,
    V,
    ignore_index,
    label_smoothing,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(Target_ptr + row_idx)

    # 1. Compute online softmax: max and sum of exp
    m_val = -float("inf")
    d_val = 0.0
    sum_z = 0.0

    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V

        z = tl.load(
            Logits_ptr + row_idx * stride_logits_m + v_offsets * stride_logits_v,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        local_max = tl.max(z, axis=0)
        new_m = tl.maximum(m_val, local_max)

        scale = tl.exp(m_val - new_m)
        d_val = d_val * scale + tl.sum(tl.exp(z - new_m), axis=0)
        m_val = new_m

        if label_smoothing > 0.0:
            sum_z += tl.sum(tl.where(mask, z, 0.0), axis=0)

    lse = m_val + tl.log(d_val)
    tl.store(Lse_ptr + row_idx, lse)

    # 2. Compute loss for this row
    if target == ignore_index:
        tl.store(Loss_ptr + row_idx, 0.0)
    else:
        # Load the logit for the target class
        z_target = tl.load(
            Logits_ptr + row_idx * stride_logits_m + target * stride_logits_v
        ).to(tl.float32)

        nll = lse - z_target
        if label_smoothing > 0.0:
            smooth_loss = lse - (sum_z / V)
            loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
        else:
            loss = nll

        tl.store(Loss_ptr + row_idx, loss)


@triton.jit
def _cross_entropy_bwd_kernel(
    dLoss_ptr,
    Logits_ptr,
    Target_ptr,
    Lse_ptr,
    dLogits_ptr,
    stride_logits_m,
    stride_logits_v,
    stride_dlogits_m,
    stride_dlogits_v,
    M,
    V,
    ignore_index,
    label_smoothing,
    scale_grad,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(Target_ptr + row_idx)

    if target == ignore_index:
        # Zero gradients for ignored targets
        for v_start in range(0, V, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < V
            tl.store(
                dLogits_ptr + row_idx * stride_dlogits_m + v_offsets * stride_dlogits_v,
                0.0,
                mask=mask,
            )
        return

    lse = tl.load(Lse_ptr + row_idx).to(tl.float32)
    dy = tl.load(dLoss_ptr + row_idx).to(tl.float32) * scale_grad
    smooth_prob = label_smoothing / V

    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V

        z = tl.load(
            Logits_ptr + row_idx * stride_logits_m + v_offsets * stride_logits_v,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        # Softmax probability p = exp(z - lse)
        p = tl.exp(z - lse)

        # True target probability q:
        # With smoothing: q = (1 - eps) * I(j == target) + (eps / V)
        # Without smoothing: q = I(j == target)
        is_target = v_offsets == target
        if label_smoothing > 0.0:
            q = tl.where(is_target, 1.0 - label_smoothing + smooth_prob, smooth_prob)
        else:
            q = tl.where(is_target, 1.0, 0.0)

        dz = dy * (p - q)

        tl.store(
            dLogits_ptr + row_idx * stride_dlogits_m + v_offsets * stride_dlogits_v,
            dz,
            mask=mask,
        )


def cross_entropy_fwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch Triton forward kernel for cross entropy loss.

    Returns:
        tuple (loss_per_token, lse)
    """
    M, V = logits.shape
    loss = torch.empty((M,), dtype=torch.float32, device=logits.device)
    lse = torch.empty((M,), dtype=torch.float32, device=logits.device)

    BLOCK_V = min(triton.next_power_of_2(V), 2048)
    num_warps = determine_num_warps(BLOCK_V)
    grid = (M,)

    _cross_entropy_fwd_kernel[grid](
        logits,
        target,
        loss,
        lse,
        logits.stride(0),
        logits.stride(1),
        M,
        V,
        ignore_index,
        label_smoothing,
        BLOCK_V=BLOCK_V,
        num_warps=num_warps,
    )
    return loss, lse


def cross_entropy_bwd(
    dloss: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    scale_grad: float = 1.0,
) -> torch.Tensor:
    """Launch single-pass Triton backward kernel for cross entropy loss."""
    M, V = logits.shape
    dlogits = torch.empty_like(logits)

    BLOCK_V = min(triton.next_power_of_2(V), 2048)
    num_warps = determine_num_warps(BLOCK_V)
    grid = (M,)

    _cross_entropy_bwd_kernel[grid](
        dloss,
        logits,
        target,
        lse,
        dlogits,
        logits.stride(0),
        logits.stride(1),
        dlogits.stride(0),
        dlogits.stride(1),
        M,
        V,
        ignore_index,
        label_smoothing,
        scale_grad,
        BLOCK_V=BLOCK_V,
        num_warps=num_warps,
    )
    return dlogits


__all__ = ["cross_entropy_fwd", "cross_entropy_bwd"]
