# CUDA backend readiness TODO

## Current status

The project is **not fully ready** for CUDA yet. It is close architecturally, but needs a backend/build preparation pass before implementing CUDA kernels.

Validated during audit:

- `make BUILD_DIR=/tmp/gradients-linux-audit UNAME_S=Linux build` succeeds.
- The Linux build currently compiles only the null backend; `gd_backend_create_default()` returns `GD_ERR_UNSUPPORTED` there, so real tensor execution/tests cannot run on Linux yet.
- `make docs-check` currently fails because `Makefile` expects `docs/design_spec.md`, which is missing.
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

## CUDA memory management plan

The current design is close to supporting proper CUDA memory, but the implementation still assumes M-series-style host-visible GPU buffers in several places. This is acceptable for a first CUDA correctness pass with `cudaMallocManaged`, but not for a performance-oriented CUDA backend.

### What already works architecturally

- Tensor descriptors already reference backend storage by `buffer + offset`, not by raw host pointers only.
- Backend ops already receive backend-neutral views such as `gd_backend_tensor_view`.
- Public transfer APIs already exist and should become the required host/device boundary:
  - `gd_span_upload`
  - `gd_span_download`
  - `gd_tensor_write`
  - `gd_tensor_read`
  - `gd_upload`
  - `gd_download`
  - `gd_tensor_from_f32`
- Arena suballocation can work over CUDA device buffers as long as offsets remain valid.

### Current blocker

Core memory currently requires every backend arena buffer to be host-visible:

- `gd_arena_init()` rejects buffers where `gd_backend_buffer_is_host_visible()` is false.
- `gd_span` stores `host_ptr` and arena allocation fills it unconditionally from `arena->base + offset`.
- Dataloader writes directly into tensor storage with `memcpy(field->tensor.storage.host_ptr, ...)`.
- Some tests/probes expect `span.host_ptr != NULL`.

That is incompatible with normal CUDA `cudaMalloc` device-local allocations.

### Required refactor

1. Make `host_ptr` optional.
   - CUDA device buffers should be allowed to return `NULL` from `gd_backend_buffer_host_ptr()`.
   - `gd_backend_buffer_is_host_visible()` should be false for normal CUDA device-local buffers.

2. Allow arenas backed by non-host-visible buffers.
   - Remove the hard rejection in `gd_arena_init()`.
   - Store `arena->base = NULL` for device-local buffers.
   - In `gd_arena_alloc()`, set:

   ```c
   out->host_ptr = arena->base != NULL ? arena->base + off : NULL;
   ```

3. Keep span validation compatible with optional host pointers.
   - If `arena->base == NULL`, validate only arena kind/slot/generation/buffer/offset/range.
   - If `arena->base != NULL`, continue checking `span->host_ptr == arena->base + span->offset`.

4. Replace dataloader direct writes with explicit backend upload.
   - Replace host-pointer `memcpy` with:

   ```c
   gd_span_upload(ctx, &field->tensor.storage, 0U, field->host_data, field->nbytes);
   ```

5. Update tests/probes.
   - Stop requiring `span.host_ptr != NULL` for all backends.
   - Only assert host pointer availability when the selected backend guarantees host-visible storage.
   - Add CUDA-specific tests for explicit upload/download.

6. Add CUDA allocation modes.
   - Bring-up/debug mode: `cudaMallocManaged`.
   - Production default: `cudaMalloc` device-local arena buffers.
   - Optional staging path: `cudaHostAlloc` / `cudaMallocHost` pinned host buffers for faster async transfers.

### Recommended order

1. Keep managed memory while bringing up initial CUDA ops.
2. Refactor core memory so `host_ptr` is optional.
3. Switch CUDA backend buffers from `cudaMallocManaged` to `cudaMalloc`.
4. Make dataloader and all CPU interactions use upload/download paths.
5. Add pinned staging and asynchronous transfer optimization after correctness is stable.

### Bottom line for CUDA memory

There is no fundamental design problem: tensors and ops are already backend-buffer oriented. The implementation needs a focused memory-model refactor to stop assuming host-visible storage everywhere. After that, `gd_tensor_empty(ctx, ...)` under `BACKEND=cuda` can behave like a PyTorch CUDA tensor allocation, while `gd_tensor_write/read` and dataloader upload paths handle host/device transfer explicitly.
