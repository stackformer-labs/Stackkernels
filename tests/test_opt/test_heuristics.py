"""Test math utilities, dynamic hardware detection, and heuristic interfaces on CPU."""

import os
import triton

from stackkernels._core._common import cdiv, next_pow2
from stackkernels._opt.hardware import (
    get_gpu_arch,
    get_hardware_info,
    is_gpu_available,
)
from stackkernels._opt.heuristics import get_heuristic
from stackkernels._opt.heuristics._util import determine_num_warps


def test_next_pow2_uses_triton():
    """Verify next_pow2 matches triton.next_power_of_2 directly."""
    for n in (1, 2, 3, 5, 7, 9, 15, 64, 127, 256, 1000, 4096):
        assert next_pow2(n) == triton.next_power_of_2(n)


def test_cdiv_uses_triton():
    """Verify cdiv matches triton.cdiv directly."""
    for a, b in ((1, 1), (9, 3), (10, 3), (11, 3), (12, 3), (100, 7)):
        assert cdiv(a, b) == triton.cdiv(a, b)


def test_determine_num_warps_scaling():
    """Verify warp count scales dynamically with block size."""
    warps_64 = determine_num_warps(64)
    warps_512 = determine_num_warps(512)
    warps_2048 = determine_num_warps(2048)

    assert warps_64 >= 1
    assert warps_512 >= warps_64
    assert warps_2048 >= warps_512


def test_heuristics_contract():
    """Verify heuristics return valid config dictionaries without hardcoded value checks."""
    for op_name in ("rms_norm", "layer_norm", "elementwise", "glu", "rope", "cross_entropy"):
        fn = get_heuristic(op_name)
        assert callable(fn), f"Heuristic for {op_name} must be callable"

        cfg = fn(shape=1024)
        assert isinstance(cfg, dict)
        assert "BLOCK_SIZE" in cfg or "BLOCK_M" in cfg


def test_hardware_dynamic_detection():
    """Verify hardware detection runs dynamically without hardcoded tables."""
    available = is_gpu_available()
    assert isinstance(available, bool)

    arch = get_gpu_arch()
    assert isinstance(arch, str)

    info = get_hardware_info()
    assert info is not None
    assert isinstance(info.name, str)
    assert isinstance(info.arch, str)

    # In CPU environment, sm_count reflects host cpu cores dynamically
    if not available:
        assert info.has_gpu is False
        assert info.sm_count == (os.cpu_count() or 1)
