"""Op name to implementation registry per backend and capability query."""

from typing import Any, Callable

SUPPORTED_OPS: dict[str, dict[str, str]] = {
    "rms_norm": {
        "torch": "stackkernels.torch.norm.rms_norm",
        "jax": "stackkernels.jax.norm.rms_norm",
    },
    "layer_norm": {
        "torch": "stackkernels.torch.norm.layer_norm",
        "jax": "stackkernels.jax.norm.layer_norm",
    },
    "fused_add_norm": {
        "torch": "stackkernels.torch.norm.fused_add_norm",
        "jax": "stackkernels.jax.norm.fused_add_norm",
    },
    "swiglu": {
        "torch": "stackkernels.torch.activation.swiglu",
        "jax": "stackkernels.jax.activation.swiglu",
    },
    "geglu": {
        "torch": "stackkernels.torch.activation.geglu",
        "jax": "stackkernels.jax.activation.geglu",
    },
    "reglu": {
        "torch": "stackkernels.torch.activation.reglu",
        "jax": "stackkernels.jax.activation.reglu",
    },
    "glu": {
        "torch": "stackkernels.torch.activation.glu",
        "jax": "stackkernels.jax.activation.glu",
    },
    "gelu": {
        "torch": "stackkernels.torch.activation.gelu",
        "jax": "stackkernels.jax.activation.gelu",
    },
    "silu": {
        "torch": "stackkernels.torch.activation.silu",
        "jax": "stackkernels.jax.activation.silu",
    },
    "relu": {
        "torch": "stackkernels.torch.activation.relu",
        "jax": "stackkernels.jax.activation.relu",
    },
    "leaky_relu": {
        "torch": "stackkernels.torch.activation.leaky_relu",
        "jax": "stackkernels.jax.activation.leaky_relu",
    },
    "sigmoid": {
        "torch": "stackkernels.torch.activation.sigmoid",
        "jax": "stackkernels.jax.activation.sigmoid",
    },
    "rope": {
        "torch": "stackkernels.torch.embedding.apply_rope",
        "jax": "stackkernels.jax.embedding.apply_rope",
    },
    "apply_rope": {
        "torch": "stackkernels.torch.embedding.apply_rope",
        "jax": "stackkernels.jax.embedding.apply_rope",
    },
    "cross_entropy": {
        "torch": "stackkernels.torch.loss.cross_entropy",
        "jax": "stackkernels.jax.loss.cross_entropy",
    },
    "fused_linear_cross_entropy": {
        "torch": "stackkernels.torch.loss.fused_linear_cross_entropy",
        "jax": "stackkernels.jax.loss.fused_linear_cross_entropy",
    },
    "flash_attention": {
        "torch": "stackkernels.torch.attention.flash_attention",
        "jax": "stackkernels.jax.attention.flash_attention",
    },
    "flash_decode": {
        "torch": "stackkernels.torch.attention.flash_decode",
        "jax": "stackkernels.jax.attention.flash_decode",
    },
}

SUPPORTED_HEAD_DIMS: tuple[int, ...] = (64, 80, 96, 128, 256)


def get_op(name: str, backend: str = "torch") -> Callable[..., Any]:
    """Retrieve the functional operator callable for the given backend."""
    import importlib

    norm_name = str(name).strip().lower()
    if norm_name not in SUPPORTED_OPS:
        raise ValueError(
            f"Unknown op '{name}'. Supported ops: {', '.join(sorted(SUPPORTED_OPS.keys()))}"
        )

    backend_norm = str(backend).strip().lower()
    if backend_norm not in ("torch", "jax"):
        raise ValueError(f"Unknown backend '{backend}'. Must be 'torch' or 'jax'.")

    mod_path, fn_name = SUPPORTED_OPS[norm_name][backend_norm].rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)


def is_supported(
    op: str,
    dtype: str = "float16",
    head_dim: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Check whether an operator configuration is supported on active or target hardware."""
    norm_op = str(op).strip().lower()
    if norm_op not in SUPPORTED_OPS:
        return False

    norm_dtype = str(dtype).strip().lower().replace("torch.", "")
    if norm_dtype == "bfloat16" and compute_capability is not None:
        if compute_capability < (8, 0):
            return False

    if head_dim is not None and norm_op in ("flash_attention", "flash_decode"):
        if head_dim not in SUPPORTED_HEAD_DIMS:
            return False

    return True


__all__ = ["get_op", "is_supported", "SUPPORTED_OPS", "SUPPORTED_HEAD_DIMS"]
