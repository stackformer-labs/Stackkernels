"""skip if no CUDA, per-dtype tolerances, backend fixture"""

import os
import sys
from pathlib import Path
from warnings import filterwarnings

import torch

# Ensure repository root and src/ are on sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Enable TRITON_INTERPRET ONLY on CPU machines; remove it when CUDA GPU is present
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    if "TRITON_INTERPRET" in os.environ:
        del os.environ["TRITON_INTERPRET"]
else:
    filterwarnings("ignore", category=UserWarning, module="triton")
    filterwarnings("ignore", message=".*Failed to find CUDA.*")
    os.environ["TRITON_INTERPRET"] = "1"

import pytest


@pytest.fixture
def cuda_available() -> bool:
    """Fixture indicating whether CUDA is available and functional."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


@pytest.fixture
def skip_if_no_cuda(cuda_available: bool) -> None:
    """Skip test if running on a machine without CUDA."""
    if not cuda_available:
        pytest.skip("Test requires a functional CUDA device")


@pytest.fixture
def dtype_tolerances():
    """Return appropriate (atol, rtol) tolerances by dtype (tight fp32, loose fp16/bf16)."""
    def _get_tolerances(dtype: torch.dtype) -> dict[str, float]:
        if dtype == torch.float32:
            return {"atol": 1e-4, "rtol": 1e-4}
        elif dtype == torch.float16:
            return {"atol": 5e-3, "rtol": 5e-3}
        elif dtype == torch.bfloat16:
            return {"atol": 1e-2, "rtol": 1e-2}
        else:
            return {"atol": 5e-3, "rtol": 5e-3}

    return _get_tolerances


@pytest.fixture(params=["torch", "jax"])
def backend_name(request) -> str:
    """Parametrized backend fixture over torch and jax."""
    if request.param == "jax":
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX is not installed in the current environment")
    return request.param
