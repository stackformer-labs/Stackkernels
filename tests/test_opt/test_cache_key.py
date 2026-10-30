"""Unit tests for 6-tuple cache key invalidation and lookup behavior."""

import pytest
from stackkernels._opt.cache import TuningCache, make_cache_key, serialize_key, deserialize_key


def test_cache_6tuple_key_roundtrip():
    """Verify storing and retrieving with explicit 6-tuple key."""
    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_SIZE": 1024, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        config=config,
        triton_version="3.7.1",
        kernel_rev=1,
    )

    retrieved = cache.get(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        triton_version="3.7.1",
        kernel_rev=1,
    )
    assert retrieved == config


def test_cache_invalidation_triton_version():
    """Verify any change in Triton version cleanly misses cache."""
    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_SIZE": 1024, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        config=config,
        triton_version="3.7.1",
        kernel_rev=1,
    )

    # Different Triton version must miss
    retrieved = cache.get(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        triton_version="3.2.0",
        kernel_rev=1,
    )
    assert retrieved is None


def test_cache_invalidation_kernel_rev():
    """Verify any change in kernel revision cleanly misses cache."""
    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_SIZE": 1024, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        config=config,
        triton_version="3.7.1",
        kernel_rev=1,
    )

    # Newer kernel revision must miss
    retrieved = cache.get(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        triton_version="3.7.1",
        kernel_rev=2,
    )
    assert retrieved is None


def test_cache_invalidation_device_fingerprint():
    """Verify different device fingerprint cleanly misses cache."""
    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_SIZE": 1024, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        config=config,
        triton_version="3.7.1",
        kernel_rev=1,
    )

    # Lookup with T4 fingerprint must miss
    retrieved = cache.get(
        op="rms_norm",
        arch="sm_75_sm40_m16",
        dtype="float16",
        bucket=2048,
        triton_version="3.7.1",
        kernel_rev=1,
    )
    assert retrieved is None


def test_serialization_6tuple():
    """Verify serialization and deserialization of 6-tuple and legacy 4-tuple keys."""
    key = make_cache_key(
        op="rms_norm",
        arch="sm_80_sm108_m40",
        dtype="float16",
        bucket=2048,
        triton_version="3.7.1",
        kernel_rev=1,
    )
    raw = serialize_key(key)
    assert raw == "rms_norm:sm_80_sm108_m40:3.7.1:1:float16:2048"

    deser = deserialize_key(raw)
    assert deser == key

    # Test legacy 4-tuple compatibility
    legacy_raw = "rms_norm:sm_80:float16:2048"
    legacy_deser = deserialize_key(legacy_raw)
    assert legacy_deser == ("rms_norm", "sm_80", "unknown", 1, "float16", "2048")

