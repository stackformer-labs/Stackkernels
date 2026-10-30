"""Unit tests for RMSNorm persistent grid-stride variant selection."""

import pytest
from stackkernels._opt.pick import pick
from tests.fixtures.profiles import A100_PROFILE, T4_PROFILE


def test_pick_selects_persistent_variant():
    """Verify persistent variant is chosen when rows > 4 * SM count and cols <= 2048."""
    # T4 has 40 SMs -> 4 * 40 = 160 threshold
    cfg = pick("rms_norm", shape=(200, 512), dtype="float16", profile=T4_PROFILE)
    assert cfg.get("variant") == "persistent"


def test_pick_selects_default_variant_when_cols_large():
    """Verify default variant is chosen when cols > 2048 even if row count is large."""
    # 200 rows > 160, but cols 4096 > 2048
    cfg = pick("rms_norm", shape=(200, 4096), dtype="float16", profile=T4_PROFILE)
    assert cfg.get("variant", "default") == "default"


def test_pick_selects_default_variant_when_rows_small():
    """Verify default variant is chosen when rows <= 4 * SM count."""
    # 50 rows <= 160 on T4
    cfg = pick("rms_norm", shape=(50, 512), dtype="float16", profile=T4_PROFILE)
    assert cfg.get("variant", "default") == "default"


def test_persistent_variant_across_profiles():
    """Verify persistent variant threshold adapts dynamically to different SM counts."""
    # A100 has 108 SMs -> threshold is 4 * 108 = 432
    # 300 rows is <= 432 on A100 -> default
    cfg_a100_small = pick("rms_norm", shape=(300, 1024), dtype="float16", profile=A100_PROFILE)
    assert cfg_a100_small.get("variant", "default") == "default"

    # 500 rows is > 432 on A100 -> persistent
    cfg_a100_large = pick("rms_norm", shape=(500, 1024), dtype="float16", profile=A100_PROFILE)
    assert cfg_a100_large.get("variant") == "persistent"
