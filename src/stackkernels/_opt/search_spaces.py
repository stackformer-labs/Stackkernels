"""Candidate configuration search space generation filtered by hardware constraints."""

from typing import Any

from stackkernels._opt.classes import decode, elementwise, row, tiled
from stackkernels._opt.constraints import check
from stackkernels._opt.hardware import DeviceProfile, get_device_profile

OP_TO_CLASS: dict[str, str] = {
    "rms_norm": "row",
    "rmsnorm": "row",
    "layer_norm": "row",
    "layernorm": "row",
    "fused_add_norm": "row",
    "cross_entropy": "row",
    "rope": "row",
    "topk_topp": "row",
    "sampling": "row",
    "elementwise": "elementwise",
    "gelu": "elementwise",
    "silu": "elementwise",
    "relu": "elementwise",
    "leaky_relu": "elementwise",
    "sigmoid": "elementwise",
    "glu": "elementwise",
    "swiglu": "elementwise",
    "geglu": "elementwise",
    "reglu": "elementwise",
    "dropout": "elementwise",
    "fused_dropout_add": "elementwise",
    "kv_update": "elementwise",
    "flash_fwd": "tiled",
    "flash_bwd": "tiled",
    "flash_attn": "tiled",
    "flash_attention": "tiled",
    "fused_linear_ce": "tiled",
    "fused_linear_cross_entropy": "tiled",
    "decode": "decode",
    "flash_decode": "decode",
}


def get_kernel_class(op: str) -> str:
    """Return the kernel class ('row', 'elementwise', 'tiled', 'decode') for an op."""
    return OP_TO_CLASS.get(op.strip().lower(), "row")


def get_search_space(
    op: str,
    shape: Any = None,
    profile: DeviceProfile | None = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate up to 12 valid candidate configurations around seed rules."""
    prof = profile or get_device_profile()
    kclass = get_kernel_class(op)

    # 1. Get initial seeds from class module
    if kclass == "row":
        seeds = row.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
    elif kclass == "elementwise":
        seeds = elementwise.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
    elif kclass == "tiled":
        seeds = tiled.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
    elif kclass == "decode":
        seeds = decode.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
    else:
        seeds = [{"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 1}]

    # 2. Generate neighboring variations around the seeds
    raw_candidates: list[dict[str, Any]] = []

    for seed in seeds:
        if seed not in raw_candidates:
            raw_candidates.append(dict(seed))

        if kclass in ("row", "elementwise"):
            bs = seed.get("BLOCK_SIZE", 1024)
            w = seed.get("num_warps", 4)
            for new_bs in (bs // 2, bs * 2):
                if 32 <= new_bs <= 8192:
                    cand = {"BLOCK_SIZE": new_bs, "num_warps": w, "num_stages": 1}
                    if cand not in raw_candidates:
                        raw_candidates.append(cand)
            for new_w in (w // 2, w * 2):
                if 1 <= new_w <= 16:
                    cand = {"BLOCK_SIZE": bs, "num_warps": new_w, "num_stages": 1}
                    if cand not in raw_candidates:
                        raw_candidates.append(cand)

        elif kclass == "tiled":
            bm = seed.get("BLOCK_M", 64)
            bn = seed.get("BLOCK_N", 64)
            w = seed.get("num_warps", 4)
            st = seed.get("num_stages", 2)
            variant = seed.get("kernel_variant", "flash_fwd")

            for new_bm, new_bn in ((bm // 2, bn), (bm, bn // 2), (bm * 2, bn)):
                if 16 <= new_bm <= 128 and 16 <= new_bn <= 128:
                    cand = {
                        "BLOCK_M": new_bm,
                        "BLOCK_N": new_bn,
                        "num_warps": w,
                        "num_stages": st,
                        "kernel_variant": variant,
                    }
                    if cand not in raw_candidates:
                        raw_candidates.append(cand)

        elif kclass == "decode":
            split = seed.get("SPLIT_KV", 4)
            bn = seed.get("BLOCK_N", 64)
            w = seed.get("num_warps", 4)
            for new_s in (split // 2, split * 2):
                if 1 <= new_s <= 32:
                    cand = {
                        "SPLIT_KV": new_s,
                        "BLOCK_M": 16,
                        "BLOCK_N": bn,
                        "num_warps": w,
                        "num_stages": 1,
                        "kernel_variant": "decode",
                    }
                    if cand not in raw_candidates:
                        raw_candidates.append(cand)

    # 3. Filter candidates by static constraints
    valid_candidates: list[dict[str, Any]] = []
    for cand in raw_candidates:
        rejections = check(cand, prof, kernel_class=kclass, dtype=dtype, problem=shape)
        if len(rejections) == 0:
            if cand not in valid_candidates:
                valid_candidates.append(cand)

    # 4. Guarantee at least the primary seed is present if no candidates survived
    if not valid_candidates and seeds:
        valid_candidates.append(seeds[0])

    # 5. Cap at 12 candidates
    return valid_candidates[:12]


__all__ = ["get_search_space", "get_kernel_class", "OP_TO_CLASS"]
