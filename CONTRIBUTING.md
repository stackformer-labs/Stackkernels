# Contributing to StackKernels

Thank you for contributing to StackKernels. This document outlines development practices, testing workflows, architecture boundaries, and submission requirements.

---

## 1. Development Setup

StackKernels requires Python 3.9 or newer.

1. Clone the repository and install editable dependencies:
   ```bash
   git clone https://github.com/stackformer-labs/stackkernels.git
   cd stackkernels
   pip install -e ".[dev,docs,torch]"
   ```
2. Verify package installation:
   ```bash
   python -m stackkernels info
   ```

---

## 2. Running Tests

### Running on CPU (Default / CI)

When developing locally on CPU or inside continuous integration environments without a CUDA device, run CPU-safe tests:

```bash
pytest tests/ -m "not gpu"
```

To run all operator tests using the reference backend:

```bash
STACKKERNELS_BACKEND=reference pytest tests/ops/
```

### Running on GPU

On a machine equipped with CUDA and OpenAI Triton:

```bash
pytest tests/
```

---

## 3. Architecture Rules and Layer Isolation

StackKernels maintains strict separation of concerns across layers:

1. **`_core/` (Layer 1)**: Contains pure Triton kernels. Modules in `_core` must **never** import `_opt`, `torch`, or `jax`. All tuning parameters must be passed as explicit kernel arguments. Never use `@triton.autotune`.
2. **`_opt/` (Layer 2)**: Optimization logic. Modules in `_opt/heuristics/` and `_opt/classes/` must be pure Python: they receive a `DeviceProfile` and return plain configuration dictionaries without framework imports at module level.
3. **`torch/` and `jax/` (Layer 3)**: Framework bindings. Handle tensor layout, autograd integration, and call `_opt.pick()` to configure kernel execution.
4. **No Hardcoded Constants**: Do not hardcode device numbers in heuristic logic. GPU hardware constants belong exclusively in `_opt/hardware.py` fallback tables.

---

## 4. Checklist for Adding a New Op

Follow this checklist from `docs/adding_an_op.md`:

1. **Reference Implementations**:
   - Add PyTorch reference in `src/stackkernels/_reference/torch/<category>.py`.
   - Add JAX reference in `src/stackkernels/_reference/jax/<category>.py`.
   - Add numerical consistency unit tests in `tests/ops/test_<op>.py`.
2. **Core Triton Kernel**:
   - Implement kernel in `src/stackkernels/_core/<category>/<op>.py`.
   - Define module-level `KERNEL_REV = 1`.
   - Keep tuning knobs (`BLOCK_SIZE`, `num_warps`, `num_stages`) as explicit arguments.
3. **Optimization Registration**:
   - Map op to its kernel class in `src/stackkernels/_opt/search_spaces.py` (`OP_TO_CLASS`).
   - Register module path in `src/stackkernels/_opt/tune.py` (`OP_TO_MODULE`).
   - Create class heuristic wrapper in `src/stackkernels/_opt/heuristics/<op>.py`.
4. **Framework Wrappers**:
   - Implement `torch.autograd.Function` and functional entry point in `src/stackkernels/torch/<category>.py`.
   - Implement JAX functional entry point with `custom_vjp` in `src/stackkernels/jax/<category>.py`.
   - Export public functions in `torch/__init__.py` and `jax/__init__.py`.
5. **Documentation & Validation**:
   - Add Google-style docstrings with Args, Returns, and minimal examples.
   - Add per-op guide in `docs/user_guide/<op>.md`.
   - Update `mkdocs.yml` if new pages are added.

---

## 5. Commit and Pull Request Rules

- **Commit Messages**: Use concise, imperative commit messages describing the change (e.g. `feat(opt): add device-aware tile pruning`, `docs: update attention variant matrix`).
- **Strict Documentation Build**: All documentation changes must pass `mkdocs build --strict` with zero warnings.
- **Accuracy First**: Never claim performance figures or hardware validations without linking to a reproducible benchmark artifact.

