# CUDA backend readiness TODO

## Current status

The project is **not fully ready** for CUDA yet. It is close architecturally, but needs a backend/build preparation pass before implementing CUDA kernels.

Validated during audit:

- `make BUILD_DIR=/tmp/gradients-linux-audit UNAME_S=Linux build` succeeds.
- The Linux build currently compiles only the null backend; `gd_backend_create_default()` returns `GD_ERR_UNSUPPORTED` there, so real tensor execution/tests cannot run on Linux yet.
- The old `make check` / `make docs-check` path has been removed; use `make test` for the core test suite.
- `README.md` points at stale `docs/rules/add_new_op.md`; the actual op guide is `docs/guides/register_op.md`.

## Build system readiness

The current `Makefile` is effectively split into Darwin/Metal vs non-Darwin/null:

- Darwin:
  - defines `GD_ENABLE_METAL=1`
  - compiles Objective-C `.m` sources
  - builds `gradients.metallib`
  - links `Foundation` and `Metal`
- non-Darwin:
  - defines `GD_ENABLE_METAL=0`
  - excludes `src/backends/metal/*`
  - builds `src/backends/null/*`

### Missing for CUDA

- Add backend selection, e.g. `BACKEND=auto|metal|cuda|null`.
- Add `GD_ENABLE_CUDA=1` when CUDA is selected.
- Add `NVCC`, `CUDA_HOME`, and `CUDA_ARCH` handling.
- Add `.cu` compilation rules.
- Add CUDA link flags for tests/probes, likely:
  - `-lcudart`
  - `-lcublas`
  - possibly `-lcublasLt`
- Update source filtering so only one backend implementation defines `gd_backend_*` symbols.
- Handle C/C++ linkage carefully: `.cu` files compile as C++, while internal backend headers are plain C.

## Library/backend design readiness

The design is **partially ready**.

### What is already good

- Core tensor/autograd code already dispatches through backend entry points in `src/core/backend.h`.
- Tensor descriptors and `gd_backend_tensor_view` / matrix view structs are mostly backend-neutral.
- Ops are capsule-based under `src/ops/<op>/`, which CUDA can mirror with per-op CUDA files.
- The scope/fence model maps reasonably to CUDA streams/events.

### Main blockers

1. **Only one concrete backend kind exists**

   `gd_backend_kind` currently only exposes `GD_BACKEND_METAL`. Add `GD_BACKEND_CUDA` and decide how backend creation selects CUDA.

2. **Backend API is global-symbol based, not a vtable**

   This is acceptable for one selected backend per build, but it is not ready for Metal + CUDA in the same binary or runtime backend switching.

3. **Memory model assumes host-visible GPU buffers**

   `src/core/memory.c` rejects backend buffers that are not host-visible. Every arena span stores `host_ptr`, and the dataloader directly copies into `tensor.storage.host_ptr`.

   CUDA can initially satisfy this with managed/mapped memory, but a performance-oriented CUDA backend probably needs device memory plus explicit upload/download/staging support.

4. **Generated backend glue is Metal/null only**

   `tools/gen_ops.c` generates:

   - `src/backends/null/backend_generated.c`
   - `src/backends/metal/metal_ops_generated.inc`

   There is no CUDA-generated op registration/stub path yet.

5. **Tests are not backend-parametrized**

   Linux/null builds compile, but normal tensor tests need a real backend. CUDA bring-up should start with selective backend smoke tests.

## Recommended implementation plan

### Phase 1: backend/build skeleton

- Add `BACKEND ?= auto` to `Makefile`.
- Support explicit `BACKEND=metal`, `BACKEND=cuda`, and `BACKEND=null`.
- Add CUDA source discovery for `src/backends/cuda`, `src/ops/**/cuda_*`, and possibly shared CUDA kernels.
- Add `.cu` build rules and CUDA link flags.
- Add `GD_BACKEND_CUDA` to `gd_backend_kind`.
- Add a CUDA backend skeleton under `src/backends/cuda/` implementing:
  - `gd_backend_create_default`
  - `gd_backend_destroy`
  - buffer create/destroy
  - upload/download
  - fill
  - scope begin/flush
  - record/wait/destroy fence via CUDA events

### Phase 2: initial memory strategy

Choose one:

- Quick bring-up: use `cudaMallocManaged` or mapped host-visible allocations so current arena assumptions hold.
- Performance path: refactor core memory to allow non-host-visible buffers and make dataloader/upload paths use explicit backend transfer/staging APIs.

Recommended: start with managed memory for first correctness pass, then refactor toward device memory/staging once kernels are working.

### Phase 3: first CUDA op

Start with a small op:

- `add` or `relu` for elementwise dispatch and validation.
- Then `matmul` via cuBLAS/cuBLASLt.

Mirror the Metal capsule convention:

```text
src/ops/<op>/cuda_<op>.cu
src/ops/<op>/cuda_<op>_types.h   # if an op-local ABI struct is needed
```

### Phase 4: generated CUDA glue

Extend `tools/gen_ops.c` to generate CUDA backend registration/stub glue if needed, analogous to Metal/null outputs.

### Phase 5: tests/probes

- Add a CUDA backend smoke probe.
- Add a minimal CUDA tensor test: context create, allocation, upload/download, fill.
- Add first op correctness test under CUDA.
- Add CI/local commands documenting `BACKEND=cuda` usage.

## Bottom line

The existing architecture is promising for adding CUDA, but the project is not CUDA-ready yet. Do a small backend abstraction/build refactor and CUDA skeleton first, then start porting kernels.