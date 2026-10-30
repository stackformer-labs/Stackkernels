"""Tests for kernel class seed configuration generators."""

from stackkernels._opt.classes import decode, elementwise, row, tiled
from stackkernels._opt.constraints import check
from tests.fixtures.profiles import A100_PROFILE, H100_PROFILE, T4_PROFILE


def test_row_seeds():
    for prof in (T4_PROFILE, A100_PROFILE):
        seeds = row.seed_configs(prof, problem=1024, dtype="float16")
        assert len(seeds) >= 3
        # First seed should be valid under constraints
        rejections = check(seeds[0], prof, kernel_class="row", dtype="float16", problem=1024)
        assert rejections == []

    # Large row test (chunked)
    seeds_large = row.seed_configs(T4_PROFILE, problem=16384, dtype="float16")
    assert seeds_large[0]["BLOCK_SIZE"] <= 4096


def test_elementwise_seeds():
    for prof in (T4_PROFILE, A100_PROFILE):
        seeds = elementwise.seed_configs(prof, problem=1024 * 1024, dtype="float16")
        assert len(seeds) >= 3
        rejections = check(seeds[0], prof, kernel_class="elementwise", dtype="float16")
        assert rejections == []


def test_tiled_seeds():
    for prof in (T4_PROFILE, A100_PROFILE, H100_PROFILE):
        for d in (64, 128, 256):
            seeds = tiled.seed_configs(prof, problem={"head_dim": d}, dtype="float16")
            assert len(seeds) >= 3
            rejections = check(seeds[0], prof, kernel_class="tiled", dtype="float16", problem={"head_dim": d})
            assert rejections == []


def test_decode_seeds():
    for prof in (T4_PROFILE, A100_PROFILE):
        # Short sequence: split=1
        seeds_short = decode.seed_configs(prof, kv_len=64)
        assert seeds_short[0]["SPLIT_KV"] == 1

        # Long sequence: split > 1
        seeds_long = decode.seed_configs(prof, kv_len=4096)
        assert seeds_long[0]["SPLIT_KV"] >= 4
