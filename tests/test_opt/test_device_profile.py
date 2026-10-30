"""Tests for DeviceProfile and hardware query pipeline."""

import pytest
from stackkernels._opt.hardware import (
    DeviceProfile,
    get_device_profile,
    get_hardware_info,
)
from tests.fixtures.profiles import (
    A100_PROFILE,
    CPU_PROFILE,
    H100_PROFILE,
    T4_PROFILE,
)


def test_device_profile_fields():
    assert T4_PROFILE.name == "Tesla T4"
    assert T4_PROFILE.compute_capability == (7, 5)
    assert T4_PROFILE.has_cp_async is False
    assert T4_PROFILE.has_bf16 is False
    assert T4_PROFILE.warp_size == 32

    assert A100_PROFILE.compute_capability == (8, 0)
    assert A100_PROFILE.has_cp_async is True
    assert A100_PROFILE.has_bf16 is True

    assert H100_PROFILE.compute_capability == (9, 0)
    assert H100_PROFILE.has_cp_async is True
    assert H100_PROFILE.has_bf16 is True


def test_device_profile_fingerprint():
    fp_t4 = T4_PROFILE.fingerprint()
    assert "tesla-t4" in fp_t4
    assert "sm75" in fp_t4
    assert "40sm" in fp_t4

    fp_a100 = A100_PROFILE.fingerprint()
    assert "sm80" in fp_a100
    assert "108sm" in fp_a100

    fp_cpu = CPU_PROFILE.fingerprint()
    assert fp_cpu == "cpu"


def test_get_device_profile_cpu():
    profile = get_device_profile("cpu")
    assert profile.compute_capability == (0, 0)
    assert profile.is_cuda is False
    assert profile.fingerprint() == "cpu"


def test_legacy_get_hardware_info():
    hw = get_hardware_info()
    assert hasattr(hw, "name")
    assert hasattr(hw, "arch")
    assert hasattr(hw, "sm_count")
