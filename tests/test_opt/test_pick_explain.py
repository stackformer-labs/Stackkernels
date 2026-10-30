"""Unit tests for pick(..., explain=True) documenting decision source, hardware facts, and occupancy."""

import pytest
from stackkernels._opt.cache import TuningCache
from stackkernels._opt.pick import pick
from tests.fixtures.profiles import A100_PROFILE, CPU_PROFILE, H100_PROFILE, T4_PROFILE


def test_pick_explain_heuristic():
    """Verify explain=True returns config and structured explanation from heuristic."""
    config, explanation = pick(
        "rms_norm",
        shape=(2, 512),
        dtype="float16",
        profile=T4_PROFILE,
        explain=True,
    )
    assert isinstance(config, dict)
    assert "BLOCK_SIZE" in config
    assert isinstance(explanation, str)
    assert "Decision Source: heuristic" in explanation
    assert "SM Count: 40" in explanation
    assert "Occupancy & Wave Math:" in explanation
    assert "Estimated Occupancy:" in explanation
    assert "Selected Configuration:" in explanation


def test_pick_explain_override():
    """Verify explain=True reflects user override source."""
    override = {"BLOCK_SIZE": 256, "num_warps": 2}
    config, explanation = pick(
        "rms_norm",
        shape=(2, 1024),
        dtype="float16",
        override=override,
        profile=A100_PROFILE,
        explain=True,
    )
    assert config == override
    assert "Decision Source: user_override" in explanation
    assert "SM Count: 108" in explanation


def test_pick_explain_cache():
    """Verify explain=True reflects user cache source."""
    cache = TuningCache(load_shipped=False)
    cached_cfg = {"BLOCK_SIZE": 512, "num_warps": 4}
    cache.set(
        op="rms_norm",
        arch=T4_PROFILE.fingerprint(),
        dtype="float16",
        bucket=512,
        config=cached_cfg,
    )

    config, explanation = pick(
        "rms_norm",
        shape=512,
        dtype="float16",
        cache=cache,
        profile=T4_PROFILE,
        explain=True,
    )
    assert config == cached_cfg
    assert "Decision Source: user_cache" in explanation


def test_pick_explain_tiled_across_profiles():
    """Verify explanation works across different hardware architectures (T4, A100, H100)."""
    for prof in (T4_PROFILE, A100_PROFILE, H100_PROFILE):
        config, explanation = pick(
            "flash_fwd",
            shape=(2, 8, 256, 64),
            dtype="float16",
            profile=prof,
            explain=True,
        )
        assert "BLOCK_M" in config
        assert f"Device Profile: {prof.name}" in explanation
        assert f"SM Count: {prof.sm_count}" in explanation
        assert "Waves:" in explanation
