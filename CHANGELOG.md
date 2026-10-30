# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Comprehensive documentation site built with Material for MkDocs and `mkdocstrings`.
- User guides for Normalization, Activations, RoPE, Loss functions, Attention, and JAX.
- Runnable integration examples in `examples/stackformer_integration.py`, `examples/pytorch_usage.py`, and `examples/jax_usage.py`.
- Documentation workflow `.github/workflows/docs.yml` checking `mkdocs build --strict`.
- Documentation example test suite `tests/test_docs_examples.py`.
- Google-style docstrings across all public functions and classes in `stackkernels.torch` and `stackkernels.jax`.

---

## [0.2.0]

### Added
- **Device-Aware Optimization Layer (`stackkernels._opt`)**:
    - Multi-tier device profiling (`DeviceProfile`) querying active CUDA SM count, shared memory, and register files, with fallback tables for offline and CPU execution.
    - Stable hardware fingerprinting (`fingerprint()`) incorporating architecture, compute capability, SM count, and VRAM.
    - Device calibration module (`calibrate.py`) measuring achieved memory bandwidth and arithmetic throughput.
    - Static constraint checker (`constraints.py`) evaluating thread limits, shared memory safety margins (10% buffer), accumulator register budgets, and architecture capabilities (BF16, cp.async).
    - Kernel taxonomy classes (`row`, `elementwise`, `tiled`, `decode`) providing pure Python configuration seeds.
    - Neighbor candidate generator (`search_spaces.py`) producing constraint-pruned candidate sets capped at 12 configurations.
    - Tuning pipeline (`tune.py`) combining seeds, constraint pruning, compiler register/spill probing, and timing.
    - Persistent tuning cache (`TuningCache`) keyed by 6-tuple `(op, fingerprint, triton_ver, kernel_rev, dtype, bucket)` with support for `$STACKKERNELS_CACHE_DIR`.
    - Resolution order in `pick()`: User Override > User Cache > Shipped Tuned Configuration > Class Heuristic.
    - `pick(..., explain=True)` reporting detailed resolution rationale and candidate rejections.
    - Persistent grid-stride forward variant for RMSNorm on high-SM devices.
- **CLI Commands**:
    - `python -m stackkernels info`: Displays system environment, GPU profile, and calibrated peaks.
    - `python -m stackkernels calibrate`: Microbenchmarks hardware bandwidth and throughput.
    - `python -m stackkernels tune`: Autotunes an operator and persists winning configurations to disk cache.

### Fixed
- Isolated `_util.py` to be pure Python without module-level `triton` imports.
- Corrected single-pass row tile bounds in `_opt/classes/row.py` up to $N=8192$.
- Made `TRITON_INTERPRET` conditional in test configurations to avoid disabling CUDA execution on GPU environments.
- Fixed test discovery by configuring `pythonpath` in `pyproject.toml` and adding package markers.
