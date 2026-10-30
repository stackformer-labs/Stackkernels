"""Tests for search space generation and constraint filtering."""

from stackkernels._opt.constraints import check
from stackkernels._opt.search_spaces import get_kernel_class, get_search_space
from tests.fixtures.profiles import A100_PROFILE, T4_PROFILE


def test_kernel_class_mapping():
    assert get_kernel_class("rms_norm") == "row"
    assert get_kernel_class("swiglu") == "elementwise"
    assert get_kernel_class("flash_fwd") == "tiled"
    assert get_kernel_class("flash_decode") == "decode"


def test_search_space_capped_at_12():
    candidates = get_search_space("rms_norm", shape=1024, profile=T4_PROFILE, dtype="float16")
    assert 1 <= len(candidates) <= 12

    # Every candidate in the search space must pass static constraints
    for cand in candidates:
        rej = check(cand, T4_PROFILE, kernel_class="row", dtype="float16", problem=1024)
        assert rej == []


def test_tiled_search_space_filtered():
    candidates = get_search_space("flash_attn", shape=(2, 8, 512, 64), profile=T4_PROFILE, dtype="float16")
    assert 1 <= len(candidates) <= 12
    for cand in candidates:
        rej = check(cand, T4_PROFILE, kernel_class="tiled", dtype="float16", problem={"head_dim": 64})
        assert rej == []

