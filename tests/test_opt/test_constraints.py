"""Tests for candidate configuration static constraints checking."""

from stackkernels._opt.constraints import check
from tests.fixtures.profiles import A100_PROFILE, T4_PROFILE


def test_valid_config_passes():
    cfg = {"BLOCK_SIZE": 1024, "num_warps": 4}
    rejections = check(cfg, T4_PROFILE, kernel_class="row", dtype="float16", problem=1024)
    assert rejections == []


def test_bf16_rejected_on_t4_allowed_on_a100():
    cfg = {"BLOCK_SIZE": 1024, "num_warps": 4}
    rej_t4 = check(cfg, T4_PROFILE, kernel_class="row", dtype="bfloat16")
    assert any("bfloat16" in r for r in rej_t4)

    rej_a100 = check(cfg, A100_PROFILE, kernel_class="row", dtype="bfloat16")
    assert rej_a100 == []


def test_stages_rejected_on_t4_without_cp_async():
    cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
    rej_t4 = check(cfg, T4_PROFILE, kernel_class="tiled", dtype="float16")
    assert any("cp.async" in r for r in rej_t4)

    rej_a100 = check(cfg, A100_PROFILE, kernel_class="tiled", dtype="float16")
    assert rej_a100 == []


def test_thread_limit_rejected():
    cfg = {"BLOCK_SIZE": 1024, "num_warps": 64}  # 64 warps = 2048 threads
    rejections = check(cfg, T4_PROFILE, kernel_class="row")
    assert any("Thread count" in r for r in rejections)


def test_oversized_shared_memory_rejected():
    # 256 x 256 tiles with head_dim 256 = stages * (256*256 + 2*256*256) * 2 bytes = 2 * (65536 + 131072) * 2 = 786 KB
    # T4 max smem is 48KB
    cfg = {"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 8, "num_stages": 2}
    rejections = check(cfg, T4_PROFILE, kernel_class="tiled", dtype="float16", problem={"head_dim": 256})
    assert any("shared memory" in r for r in rejections)


def test_absurdly_large_tile_vs_problem_rejected():
    cfg = {"BLOCK_SIZE": 4096, "num_warps": 4}
    # problem row size is 64, 4096 is 64x larger (>8x)
    rejections = check(cfg, T4_PROFILE, kernel_class="row", problem=64)
    assert any("absurdly larger" in r for r in rejections)


def test_accumulator_register_budget_rejected():
    # 128x128 tile with 2 warps = 16384 / 64 = 256 regs/thread > 114 max budget
    cfg = {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 2, "num_stages": 2}
    rejections = check(cfg, A100_PROFILE, kernel_class="tiled", dtype="float16")
    assert any("Accumulator registers" in r for r in rejections)


def test_stages_depth_on_sm80_rejected():
    # num_stages > 4 on Ampere (sm80) is rejected
    cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 5}
    rejections = check(cfg, A100_PROFILE, kernel_class="tiled", dtype="float16")
    assert any("pipeline depth limit" in r for r in rejections)


def test_max_row_limit_rejected():
    # BLOCK_SIZE > 16384 is rejected
    cfg = {"BLOCK_SIZE": 32768, "num_warps": 4}
    rejections = check(cfg, T4_PROFILE, kernel_class="row")
    assert any("exceeds maximum row limit" in r for r in rejections)
