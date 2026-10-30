"""importing stackkernels works without torch or jax"""

import sys
from unittest.mock import patch

import pytest


def test_import_with_blocked_torch_and_jax():
    """Verify stackkernels imports cleanly when torch and jax are completely unavailable."""
    blocked_modules = {
        "torch": None,
        "torch.cuda": None,
        "torch.nn": None,
        "jax": None,
        "jax.numpy": None,
    }

    # Evict stackkernels from sys.modules to simulate fresh import
    loaded_stackkernels = {
        k: sys.modules.pop(k)
        for k in list(sys.modules.keys())
        if k == "stackkernels" or k.startswith("stackkernels.")
    }

    try:
        with patch.dict(sys.modules, blocked_modules):
            import stackkernels

            # Basic attributes must be accessible
            assert hasattr(stackkernels, "__version__")
            assert stackkernels.__version__ == "0.1.0"

            # Options must work
            assert hasattr(stackkernels, "options")
            assert hasattr(stackkernels, "get_backend")
            assert hasattr(stackkernels, "set_backend")
            assert stackkernels.get_backend() in ("auto", "triton", "reference")

            # Frameworks must not have been imported
            assert "torch" not in sys.modules or sys.modules["torch"] is None
            assert "jax" not in sys.modules or sys.modules["jax"] is None
    finally:
        # Restore sys.modules
        sys.modules.update(loaded_stackkernels)


def test_options_configuration():
    """Verify backend options getter and setter."""
    import stackkernels

    original_backend = stackkernels.get_backend()
    try:
        stackkernels.set_backend("reference")
        assert stackkernels.get_backend() == "reference"
        assert stackkernels.options().backend == "reference"

        stackkernels.set_backend("triton")
        assert stackkernels.get_backend() == "triton"
        assert stackkernels.options().backend == "triton"

        stackkernels.options(backend="auto")
        assert stackkernels.get_backend() == "auto"

        with pytest.raises(ValueError, match="Invalid backend"):
            stackkernels.set_backend("cuda_native")
    finally:
        stackkernels.set_backend(original_backend)
