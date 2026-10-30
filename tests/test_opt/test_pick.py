"""Test resolution order and config picking (Override > Cache > Heuristic)."""

from stackkernels._opt.cache import TuningCache
from stackkernels._opt.pick import pick


def test_override_takes_precedence():
    """Verify user override takes highest priority over cache and heuristics."""
    cache = TuningCache(load_shipped=False)
    cache.set(
        op="rms_norm",
        arch="sm_80",
        dtype="float16",
        bucket=4096,
        config={"BLOCK_SIZE": 2048, "num_warps": 8, "source": "cache"},
    )

    override_config = {"BLOCK_SIZE": 512, "num_warps": 2, "source": "override"}

    result = pick(
        op="rms_norm",
        shape=4096,
        dtype="float16",
        arch="sm_80",
        override=override_config,
        cache=cache,
    )
    assert result == override_config


def test_cache_takes_precedence_over_heuristic():
    """Verify cache is used when no override is provided."""
    cache = TuningCache(load_shipped=False)
    cached_config = {"BLOCK_SIZE": 1024, "num_warps": 4, "source": "cache"}
    cache.set(
        op="rms_norm",
        arch="sm_80",
        dtype="float16",
        bucket=4096,
        config=cached_config,
    )

    result = pick(
        op="rms_norm",
        shape=4096,
        dtype="float16",
        arch="sm_80",
        override=None,
        cache=cache,
    )
    assert result == cached_config


def test_heuristic_fallback_on_cache_miss():
    """Verify static heuristic is used when cache misses and no override is given."""
    cache = TuningCache(load_shipped=False)

    result = pick(
        op="rms_norm",
        shape=4096,
        dtype="float16",
        arch="sm_80",
        override=None,
        cache=cache,
    )
    assert isinstance(result, dict)
    assert "BLOCK_SIZE" in result or "BLOCK_M" in result


def test_heuristic_lookup_by_op_name():
    """Verify heuristic lookup works by op name with case/alias tolerance."""
    res_lower = pick(op="rms_norm", shape=1024, arch="sm_80")
    res_upper = pick(op="RMS_NORM", shape=1024, arch="sm_80")
    res_alias = pick(op="rmsnorm", shape=1024, arch="sm_80")

    assert res_lower == res_upper == res_alias


def test_unknown_op_fallback():
    """Verify fallback defaults are returned for unregistered ops."""
    result = pick(op="unknown_unregistered_op", shape=1024)
    assert isinstance(result, dict)
    assert len(result) > 0
