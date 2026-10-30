"""Tests for pure occupancy math and compiler probe."""

from stackkernels._opt.occupancy import (
    accumulator_regs_per_thread,
    probe,
    programs_per_sm,
    waves,
)
from tests.fixtures.profiles import A100_PROFILE, CPU_PROFILE, T4_PROFILE


def test_programs_per_sm_t4():
    # T4: 1024 max threads per SM, 65536 max regs per SM, 65536 max smem per SM, 16 block limit
    # 4 warps = 128 threads. Max programs from threads = 1024 // 128 = 8.
    # 32 regs per thread -> 128 * 32 = 4096 regs per program. Limit = 65536 // 4096 = 16.
    # 0 smem. Block limit = 16.
    # Expected: min(8, 16, 16) = 8
    occ = programs_per_sm(T4_PROFILE, n_regs=32, smem_bytes=0, num_warps=4)
    assert occ == 8

    # High register usage: 128 regs per thread -> 16384 regs per prog -> 4 programs
    occ_high_regs = programs_per_sm(T4_PROFILE, n_regs=128, smem_bytes=0, num_warps=4)
    assert occ_high_regs == 4

    # High shared memory: 32KB per program -> 65536 // 32768 = 2 programs
    occ_smem = programs_per_sm(T4_PROFILE, n_regs=32, smem_bytes=32768, num_warps=4)
    assert occ_smem == 2


def test_programs_per_sm_a100_block_limit():
    # A100 has 32 block limit
    # 1 warp = 32 threads -> 2048 // 32 = 64 thread limit, 16 regs -> 65536 // 512 = 128 reg limit
    # Block limit is 32
    occ = programs_per_sm(A100_PROFILE, n_regs=16, smem_bytes=0, num_warps=1)
    assert occ == 32


def test_programs_per_sm_cpu():
    occ = programs_per_sm(CPU_PROFILE, n_regs=32, smem_bytes=0, num_warps=4)
    assert occ == 1


def test_waves_calculation():
    # 40 SMs, 2 programs per SM = 80 capacity
    assert waves(80, T4_PROFILE, programs_per_sm=2) == 1
    assert waves(81, T4_PROFILE, programs_per_sm=2) == 2
    assert waves((10, 8), T4_PROFILE, programs_per_sm=2) == 1
    assert waves((10, 9), T4_PROFILE, programs_per_sm=2) == 2


def test_accumulator_regs_per_thread():
    # 64 x 64 = 4096 / (32 * 4) = 4096 / 128 = 32 regs
    assert accumulator_regs_per_thread(64, 64, num_warps=4) == 32
    # 128 x 64 = 8192 / (32 * 4) = 64 regs
    assert accumulator_regs_per_thread(128, 64, num_warps=4) == 64


def test_probe_none_kernel():
    res = probe(None)
    assert res is None
