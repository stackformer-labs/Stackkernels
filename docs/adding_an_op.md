# Adding a New Operation to StackKernels

Follow this step-by-step checklist when introducing a new kernel operation to StackKernels.

---

## 1. Implement Reference Implementations
- Create PyTorch reference in `src/stackkernels/_reference/torch/<category>.py` (using fp32 accumulation).
- Create JAX reference in `src/stackkernels/_reference/jax/<category>.py`.
- Add numerical consistency unit tests in `tests/ops/test_<op>.py`.

---

## 2. Implement Core Triton Kernel
- Place kernel in `src/stackkernels/_core/<category>/<op>.py`.
- Define module-level integer:
  ```python
  KERNEL_REV = 1
  ```
- **Rule**: Do **NOT** import `_opt`, `torch`, or `jax` inside `_core/`.
- Keep tuning knobs (`BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) as explicit arguments.

---

## 3. Register Op in Optimization Layer
1. **Assign Kernel Class**:
   Add op mapping to `src/stackkernels/_opt/search_spaces.py`:
   ```python
   OP_TO_CLASS = {
       ...
       "<new_op>": "row",  # or "elementwise", "tiled", "decode"
   }
   ```
2. **Register Module for Revision Tracking**:
   Add op module path to `src/stackkernels/_opt/tune.py`:
   ```python
   OP_TO_MODULE = {
       ...
       "<new_op>": "stackkernels._core.<category>.<op>",
   }
   ```
3. **Add Heuristic Wrapper**:
   Create `src/stackkernels/_opt/heuristics/<op>.py`:
   ```python
   from stackkernels._opt.classes import row
   from stackkernels._opt.hardware import DeviceProfile, get_device_profile

   def get_<op>_config(shape=None, dtype="float16", arch=None, profile=None, **kwargs):
       prof = profile or get_device_profile()
       seeds = row.seed_configs(prof, problem=shape, dtype=dtype, **kwargs)
       return seeds[0]
   ```
   Export the function in `src/stackkernels/_opt/heuristics/__init__.py`.

---

## 4. Implement Framework Wrappers
- **PyTorch** (`src/stackkernels/torch/<category>.py`):
  - Call `_opt.pick("<new_op>", shape=..., dtype=...)` to resolve optimal block sizes and warp counts.
  - Implement forward and backward using `torch.autograd.Function`.
  - Provide fallback to `_reference` when CUDA is unavailable.
- **JAX** (`src/stackkernels/jax/<category>.py`):
  - Bind Triton kernel via `jax.triton_call` and `custom_vjp`.

---

## 5. Verify and Test
1. **Optimization Unit Tests**:
   - Verify `pick("<new_op>", shape=...)` returns a valid config.
   - Verify `pick("<new_op>", shape=..., explain=True)` outputs human-readable explanation.
   - Verify search space generation in `tests/test_opt/test_search_spaces.py`.
2. **Framework Unit Tests**:
   - Run CPU unit tests:
     ```bash
     pytest tests/test_opt tests/ops/test_<op>.py
     ```
   - Ensure clean CUDA skip when running on CPU-only machines.
