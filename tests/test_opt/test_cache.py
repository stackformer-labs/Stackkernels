"""Test tuning cache operations (round trip, serialization, bucketing)."""

import json

from stackkernels._opt.cache import TuningCache, make_cache_key, normalize_bucket


def test_cache_round_trip():
    """Verify storing and retrieving configurations."""
    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_SIZE": 1024, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_80",
        dtype="float16",
        bucket=2048,
        config=config,
    )

    retrieved = cache.get(
        op="rms_norm",
        arch="sm_80",
        dtype="float16",
        bucket=2048,
    )
    assert retrieved == config


def test_cache_miss():
    """Verify cache miss returns None."""
    cache = TuningCache(load_shipped=False)
    assert cache.get(op="rms_norm", arch="sm_80", dtype="float16", bucket=9999) is None


def test_bucket_normalization():
    """Verify different representations of the same bucket resolve identically."""
    assert normalize_bucket(4096) == "4096"
    assert normalize_bucket("4096") == "4096"
    assert normalize_bucket((1024, 4096)) == "1024x4096"
    assert normalize_bucket([1024, 4096]) == "1024x4096"

    cache = TuningCache(load_shipped=False)
    config = {"BLOCK_M": 64, "BLOCK_N": 64}

    cache.set(
        op="flash_attn",
        arch="sm_90",
        dtype="bfloat16",
        bucket=(1024, 4096),
        config=config,
    )

    # Lookup with list instead of tuple should hit
    assert cache.get(
        op="flash_attn",
        arch="sm_90",
        dtype="bfloat16",
        bucket=[1024, 4096],
    ) == config

    # Lookup with pre-formatted string should hit
    assert cache.get(
        op="flash_attn",
        arch="sm_90",
        dtype="bfloat16",
        bucket="1024x4096",
    ) == config


def test_cache_file_round_trip(tmp_path):
    """Verify saving to JSON and loading from JSON file."""
    cache = TuningCache(cache_dir=tmp_path, load_shipped=False)
    config = {"BLOCK_SIZE": 512, "num_warps": 4}

    cache.set(
        op="rms_norm",
        arch="sm_89",
        dtype="float16",
        bucket=1024,
        config=config,
    )

    saved_path = cache.save_user_cache(tmp_path / "custom_cache.json")
    assert saved_path.exists()

    # Load into a fresh cache instance
    new_cache = TuningCache(load_shipped=False)
    new_cache.load_file(saved_path)

    loaded_config = new_cache.get(
        op="rms_norm",
        arch="sm_89",
        dtype="float16",
        bucket=1024,
    )
    assert loaded_config == config


def test_clear_cache():
    """Verify clearing cache resets stored configurations."""
    cache = TuningCache(load_shipped=False)
    cache.set(
        op="rms_norm",
        arch="sm_80",
        dtype="float16",
        bucket=1024,
        config={"BLOCK_SIZE": 1024},
    )
    assert cache.get(op="rms_norm", arch="sm_80", dtype="float16", bucket=1024) is not None

    cache.clear()
    assert cache.get(op="rms_norm", arch="sm_80", dtype="float16", bucket=1024) is None
