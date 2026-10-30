"""Unit tests for the device-aware candidate tuning pipeline."""

import pytest
from stackkernels._opt.hardware import DeviceProfile
from stackkernels._opt.tune import get_kernel_rev, get_triton_version, tune
from tests.fixtures.profiles import A100_PROFILE, CPU_PROFILE, T4_PROFILE


def test_get_kernel_rev():
    """Verify KERNEL_REV integer is retrieved from _core kernel modules."""
    assert get_kernel_rev("rms_norm") == 1
    assert get_kernel_rev("layer_norm") == 1
    assert get_kernel_rev("flash_fwd") == 1
    assert get_kernel_rev("nonexistent_op") == 1


def test_get_triton_version():
    """Verify Triton version retrieval."""
    ver = get_triton_version()
    assert isinstance(ver, str)
    assert len(ver) > 0


def test_tune_cpu_fallback():
    """Verify tune on CPU returns the candidate seed without throwing."""
    cfg = tune("rms_norm", shape=(2, 512), dtype="float16", profile=CPU_PROFILE, write_to_disk=False)
    assert isinstance(cfg, dict)
    assert "BLOCK_SIZE" in cfg
    assert "num_warps" in cfg
    assert cfg["BLOCK_SIZE"] == 512


def test_tune_with_custom_candidates():
    """Verify passing explicit candidate configs returns the chosen config."""
    custom_cands = [
        {"BLOCK_SIZE": 256, "num_warps": 2, "num_stages": 1},
        {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 1},
    ]
    cfg = tune(
        "rms_norm",
        shape=(2, 1024),
        candidate_configs=custom_cands,
        profile=CPU_PROFILE,
        write_to_disk=False,
    )
    assert cfg["BLOCK_SIZE"] == 256
    assert cfg["num_warps"] == 2


def test_tune_tiled_profile():
    """Verify tune with T4 profile produces a valid tiled configuration."""
    cfg = tune(
        "flash_fwd",
        shape=(2, 8, 128, 64),
        dtype="float16",
        profile=T4_PROFILE,
        write_to_disk=False,
    )
    assert "BLOCK_M" in cfg
    assert "BLOCK_N" in cfg
    assert "num_warps" in cfg
    assert "num_stages" in cfg
    # On sm75 (T4 without cp.async), num_stages is 1
    assert cfg["num_stages"] == 1

    # On sm80 (A100 with cp.async), num_stages is 2
    cfg_a100 = tune(
        "flash_fwd",
        shape=(2, 8, 128, 64),
        dtype="float16",
        profile=A100_PROFILE,
        write_to_disk=False,
    )
    assert cfg_a100["num_stages"] == 2
