"""lazy package root; must never import torch or jax"""

import importlib
from typing import Any

from stackkernels._options import get_backend, options, set_backend

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "options",
    "get_backend",
    "set_backend",
    "torch",
    "jax",
    "_core",
    "_opt",
    "_reference",
    "hub",
]

_LAZY_MODULES: dict[str, str] = {
    "torch": "stackkernels.torch",
    "jax": "stackkernels.jax",
    "_core": "stackkernels._core",
    "_opt": "stackkernels._opt",
    "_reference": "stackkernels._reference",
    "hub": "stackkernels.hub",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        return importlib.import_module(_LAZY_MODULES[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))
