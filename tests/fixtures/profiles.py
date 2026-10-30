"""DeviceProfile test fixtures for T4, A100, H100 and CPU."""

from stackkernels._opt.hardware import DeviceProfile

T4_PROFILE = DeviceProfile(
    name="Tesla T4",
    compute_capability=(7, 5),
    sm_count=40,
    warp_size=32,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_threads_per_sm=1024,
    max_smem_per_block=49152,
    max_smem_per_sm=65536,
    l2_bytes=4 * 1024 * 1024,
    total_memory_bytes=16 * 1024 * 1024 * 1024,
    peak_bandwidth_gbps=320.0,
    peak_tflops={"fp16": 65.0, "fp32": 8.1},
    has_cp_async=False,
    has_bf16=False,
)

A100_PROFILE = DeviceProfile(
    name="NVIDIA A100-SXM4-40GB",
    compute_capability=(8, 0),
    sm_count=108,
    warp_size=32,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_threads_per_sm=2048,
    max_smem_per_block=167936,
    max_smem_per_sm=167936,
    l2_bytes=40 * 1024 * 1024,
    total_memory_bytes=40 * 1024 * 1024 * 1024,
    peak_bandwidth_gbps=1555.0,
    peak_tflops={"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    has_cp_async=True,
    has_bf16=True,
)

A100_80GB_PROFILE = DeviceProfile(
    name="NVIDIA A100-SXM4-80GB",
    compute_capability=(8, 0),
    sm_count=108,
    warp_size=32,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_threads_per_sm=2048,
    max_smem_per_block=167936,
    max_smem_per_sm=167936,
    l2_bytes=40 * 1024 * 1024,
    total_memory_bytes=80 * 1024 * 1024 * 1024,
    peak_bandwidth_gbps=2039.0,
    peak_tflops={"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    has_cp_async=True,
    has_bf16=True,
)

H100_PROFILE = DeviceProfile(
    name="NVIDIA H100 80GB HBM3",
    compute_capability=(9, 0),
    sm_count=132,
    warp_size=32,
    max_regs_per_sm=65536,
    max_regs_per_thread=255,
    max_threads_per_sm=2048,
    max_smem_per_block=232448,
    max_smem_per_sm=232448,
    l2_bytes=50 * 1024 * 1024,
    total_memory_bytes=80 * 1024 * 1024 * 1024,
    peak_bandwidth_gbps=3350.0,
    peak_tflops={"fp16": 989.0, "bf16": 989.0, "fp32": 67.0},
    has_cp_async=True,
    has_bf16=True,
)

CPU_PROFILE = DeviceProfile(
    name="cpu",
    compute_capability=(0, 0),
    sm_count=8,
    warp_size=32,
    max_regs_per_sm=0,
    max_regs_per_thread=0,
    max_threads_per_sm=1,
    max_smem_per_block=0,
    max_smem_per_sm=0,
    l2_bytes=0,
    total_memory_bytes=0,
    peak_bandwidth_gbps=None,
    peak_tflops=None,
    has_cp_async=False,
    has_bf16=False,
)

__all__ = ["T4_PROFILE", "A100_PROFILE", "A100_80GB_PROFILE", "H100_PROFILE", "CPU_PROFILE"]
