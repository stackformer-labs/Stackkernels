"""global switches: backend = auto | triton | reference"""

import os
from typing import Literal

BackendType = Literal["auto", "triton", "reference"]
VALID_BACKENDS: tuple[str, ...] = ("auto", "triton", "reference")


class Options:
    """Global configuration options for stackkernels."""

    def __init__(self, backend: str | None = None) -> None:
        if backend is None:
            backend = os.environ.get("STACKKERNELS_BACKEND", "auto")
        self._backend: str = "auto"
        self.backend = backend

    @property
    def backend(self) -> str:
        """Get the active backend ('auto', 'triton', or 'reference')."""
        return self._backend

    @backend.setter
    def backend(self, val: str) -> None:
        """Set the active backend with validation."""
        normalized = str(val).strip().lower()
        if normalized not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{val}'. Must be one of: {', '.join(VALID_BACKENDS)}"
            )
        self._backend = normalized

    def __repr__(self) -> str:
        return f"Options(backend='{self._backend}')"


_GLOBAL_OPTIONS = Options()


def get_backend() -> str:
    """Return the current backend setting."""
    return _GLOBAL_OPTIONS.backend


def set_backend(backend: str) -> None:
    """Set the active backend ('auto', 'triton', or 'reference')."""
    _GLOBAL_OPTIONS.backend = backend


def options(backend: str | None = None) -> Options:
    """Get the global options object, optionally setting the backend."""
    if backend is not None:
        set_backend(backend)
    return _GLOBAL_OPTIONS
