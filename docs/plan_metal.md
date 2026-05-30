# gradients.c — Metal Backend Plan

Status: draft v0.1

Goal: bring up a **correct Metal backend** for Apple Silicon that executes the
existing graph IR op-by-op on the GPU, proven equal to CPU_REF through the
parity harness. Performance work (fusion, layout planning, autotune) is
explicitly deferred. This plan turns "Metal is a registered backend" from a seam
into a working second execution backend.

Reference: [`docs/design_spec.md`](design_spec.md) §6–7 (backend roles), §8–9
(storage/memory), §13 (graph execution), §17 (private backend API), §20 (parity
debugging). Prerequisite seams: [`docs/plan_metal_prereqs.md`](plan_metal_prereqs.md)
(P0–P8 landed: vtable, opaque storage, transfer routing, executor dispatch,
synchronization, fallback policy, device-aware memory kind, `gd_graph_compare`).

Guiding invariant: **Metal adds one vtable registration + a kernel library. No
core/graph/autograd/op source changes for execution.** The only core touch
allowed is auto-registering the backend on macOS. Every op lands with a
`gd_graph_compare(CPU, METAL)` parity test before the next op starts.

Target hardware (v1): Apple M-series, unified memory. `MTLBuffer` with
`storageModeShared` is CPU-addressable, so host↔device transfer is `memcpy` to
`buffer.contents`. No staging buffers, no blit encoders in v1.

Non-goal: kernel fusion, `lower_graph`, physical layout selection, workspace
planning, buffer reuse, kernel autotuning/cache, multi-GPU, fp16/bf16/quant
kernels. Tracked separately. v1 is F32, one compute kernel per IR node.

---

## Definition of done

1. `make check` passes on macOS with the Metal backend registered; CPU-only and
   non-macOS builds stay green.
2. Metal backend is selected by device, not special-cased: `gd_graph_compile(g,
   {GD_DEVICE_METAL,0})` runs the graph on the GPU through the standard vtable.
3. Every implemented op proves CPU↔Metal parity via `gd_graph_compare` within
   F32 tolerance (atol/rtol 1e-4 default; tightened per op where exact).
4. A full MLP training step (forward + backward + AdamW) runs on Metal and
   matches CPU_REF forward tensors and gradients node-by-node.
5. Unsupported ops (none, once M3 lands) fall back per policy: `GD_FALLBACK_NONE`
   → `GD_ERR_UNSUPPORTED`; `GD_FALLBACK_CPU_REF` → whole-graph CPU.
6. No GPU work observed before `synchronize`/blocking download; memory-safe under
   the existing leak checks (no `MTLBuffer` leaks across context lifetime).
7. The backend is one file tree under `src/backends/metal/` plus `.metal`
   shaders; removing its registration reverts to CPU-only with no other edits.

---

## Phase status checklist

- [x] M0 — Backend skeleton: device/queue, registration, shared-buffer storage, transfers, synchronize, metallib load, pipeline cache, one kernel (`add`) end-to-end
- [ ] M1 — Elementwise + unary kernels (mul, scale, relu, silu, copy, cast)
- [ ] M2 — Matmul/linear + reductions (matmul, linear, sum, mean, softmax, rms_norm, cross_entropy)
- [ ] M3 — Backward + optimizer (*_bwd, reduce_to, step_inc, adamw_step, assert_*)
- [ ] M4 — Full MLP training parity CPU↔Metal (forward tensors + gradients)

M0 is the keystone; M1–M3 are independent op batches once M0 lands. M4 depends on
M1–M3. Each op is "done" only with a passing parity test.

---

## Backend architecture

### Metal API facts (verified against Apple's Metal Programming Guide)
Local copies in `data/metal/` (cmd_submission, buffers, functions_libraries,
compute_encoder). The design below is pinned to these:
- **Coherency boundary is the command buffer.** A shared `MTLBuffer` is
  CPU-addressable, but the GPU only observes CPU writes made *before* `commit`,
  and the CPU only observes GPU writes *after* the command buffer reaches
  `MTLCommandBufferStatusCompleted`. Unified memory does **not** remove the need
  to synchronize — it makes the P4 blocking-read contract mandatory for
  correctness, not just ordering.
- **Long-lived objects** (build once at `init`, reuse): `MTLDevice`,
  `MTLCommandQueue`, `MTLLibrary`, `MTLComputePipelineState`. **Transient,
  single-use**: `MTLCommandBuffer`, command encoders (cheap, autoreleased).
- **Serial encoder ordering.** Commands in one command buffer execute in encode
  order; a compute command's results are visible to commands encoded after it.
  So a single `MTLComputeCommandEncoder` with one `dispatchThreads` per node is
  correct under the default serial dispatch — no manual `memoryBarrier`.
- **Execution status.** After `waitUntilCompleted`, `commandBuffer.status ==
  MTLCommandBufferStatusError` signals failure; `commandBuffer.error` carries the
  reason. Map to `GD_ERR_BACKEND`.
- **Library load.** `newLibraryWithURL:error:` on the prebuilt `.metallib`;
  `newFunctionWithName:` returns `nil` if a kernel is absent.
- **Dispatch sizing.** Use pipeline `maxTotalThreadsPerThreadgroup` and
  `threadExecutionWidth` to pick threadgroup size; `dispatchThreads:` (non-
  uniform) avoids manual grid rounding on Apple GPUs.

### File layout
```
src/backends/metal/metal_backend.h     vtable decl, _gd_metal_backend_register
src/backends/metal/metal_backend.m      vtable impl (ARC), device/queue/library/pipeline cache
src/backends/metal/metal_exec.m         compile/execute: per-value buffer plan + encode loop
src/backends/metal/kernels.metal        F32 compute kernels (one per op)
src/backends/metal/metal_probe.{h,m}    (existing toolchain probe; remove once real backend lands)
```
`gradients.metallib` is already produced by the Makefile (P7). Loaded at runtime
via `newLibraryWithURL:`.

### Memory model (unified)
- caps: `host_visible = true`, `supports_cpu_ref = false`,
  `default_memory = GD_MEM_UNIFIED`.
- `storage_alloc`: `[device newBufferWithLength:nbytes options:MTLResourceStorageModeShared]`,
  zero-filled. Handle = retained `id<MTLBuffer>` (`(__bridge_retained void*)`).
  Validate `desc.device.type == GD_DEVICE_METAL`; accept `GD_MEM_UNIFIED` and
  `GD_MEM_HOST` (shared buffer satisfies both); reject `GD_MEM_DEVICE`-only asks
  that forbid host access (none in v1).
- `storage_free`: `CFBridgingRelease` the buffer (ARC drops it). Spec §9: free
  must be async-safe — v1 `execute` blocks at `synchronize` before any free path
  that matters, and graph executables are freed only after the command buffer
  completes; document the ordering, revisit when async lands.
- `storage_host_ptr`: return `buffer.contents` (valid because shared).
- `upload`/`download`: `memcpy` to/from `buffer.contents + offset`. `download`
  calls `synchronize` first (P4 blocking contract).

### Backend state (`impl`)
The `.m` is ARC, so wrap backend-private state in a small Obj-C object holding
strong references, stored in `backend->impl` via `(__bridge_retained void*)` and
released (`__bridge_transfer`) in `shutdown`:
```
@interface GDMetalState : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;          // long-lived, created once
@property id<MTLLibrary> library;             // loaded from .metallib at init
@property NSArray<id<MTLComputePipelineState>> *pipelines; // indexed by op kind
@property id<MTLCommandBuffer> inFlight;      // last committed buffer, or nil
@end
```
`queue`/`library`/`pipelines` are built once. `inFlight` tracks the committed
buffer so `synchronize` can wait and `execute` can avoid the re-run coherency
hazard (below).

### Execution model (one kernel per node — the GPU_SAFE rung)
- `compile`: allocate a Metal `gd_storage` for **every** graph value (leaves
  included, unlike CPU_REF which borrows leaf host storage). Mirror
  `cpu_compile`'s table. Record which values are external leaves so `execute`
  can stage their host bytes into the shared buffer before dispatch.
  - leaf staging: read leaf CPU storage host ptr, `memcpy` into the value's
    Metal buffer. Must happen **before** `commit` (coherency rule).
- `execute`:
  1. If `inFlight != nil`, `waitUntilCompleted` on it first — we are about to
     mutate input buffers via CPU writes, which the docs require to precede the
     next `commit`. This keeps single-run async behavior while making re-runs
     correct. Clear `inFlight`.
  2. Stage all external-leaf host bytes into their Metal buffers.
  3. One `[queue commandBuffer]`; one `[cmdbuf computeCommandEncoder]`; for each
     node: `setComputePipelineState` for the op, `setBuffer:offset:atIndex:` for
     inputs/outputs, `setBytes:` a `params` struct, `dispatchThreads:`;
     `endEncoding` after the loop.
  4. `commit` (no wait). Store the buffer as `inFlight`.
- `execute_until(node_id)`: same loop bounded by `node_id`, then commit +
  `waitUntilCompleted` (debug partial execution stays parity-comparable).
- `synchronize`: if `inFlight`, `waitUntilCompleted`; check `status`/`error`
  (→ `GD_ERR_BACKEND`); clear `inFlight`.
- `download`: `synchronize` first, then `memcpy` from `buffer.contents+off`
  (blocking, per P4).
- `value_storage`: return the per-value Metal `gd_storage` + offset. Existing
  `gd_tensor_materialize` / `gd_graph_compare` download through it.
- `executable_free`: `synchronize` (ensure no in-flight buffer references the
  buffers), then release owned buffers.

### Kernel dispatch conventions
- elementwise/unary/copy/cast: 1D grid = numel, `dispatchThreads:` with
  `threadsPerThreadgroup` clamped to `maxTotalThreadsPerThreadgroup`. Broadcast
  handled by passing both operand shapes/strides in `params` and indexing like
  the CPU kernel.
- matmul/linear: 2D grid over (M, N); naive inner-product per thread in v1
  (tiling deferred). Honor `trans_a`/`trans_b`/bias via params.
- reductions (sum/mean/softmax/rms_norm/cross_entropy): one threadgroup per
  reduced row; threadgroup memory + parallel reduction. Match CPU numerics
  closely enough for 1e-4 (document any reduction-order tolerance).

### Pipeline cache
- At `init`: build `MTLComputePipelineState` for each kernel function name once,
  store in a fixed table keyed by `_gd_op_kind` (+ variant). `supports_node`
  returns true iff the op's pipeline is present. Missing → fallback policy.

### `params` struct
- A single C struct shared (by layout) between `.m` and `.metal`, carrying
  ndim, sizes[GD_MAX_DIMS], strides, and op attrs (scale/dim/eps/trans/...).
  Defined once in a header included by both; keep it POD and 16-byte friendly.

---

## M0 — Backend skeleton + first kernel

- [x] Phase complete

Note (landed): `src/backends/metal/{metal_backend.m,kernels.metal,
metal_kernel_types.h}`. Single TU for M0. `GDMetalState` (ARC) holds
device/queue/library/pipelines/inFlight; built once at init from
`gradients.metallib` (path via `GRADIENTS_METALLIB` or `build/gradients.metallib`).
Unified shared buffers; per-value buffer plan with leaf staging before commit;
one serial compute encoder, one `dispatchThreads` per node; blocking
download/synchronize with command-buffer status→`GD_ERR_BACKEND`. Auto-registered
in `gd_context_create` under `-DGD_ENABLE_METAL=1` (best-effort; CPU-only if no
GPU/metallib). Backend handle exposed to core via `_gd_storage_handle`. Tests:
`tests/test_metal.c` (direct add, CPU↔Metal parity incl. broadcast, fallback
NONE/CPU_REF). Verified clean under ASan (CPU and Metal paths).

**Two latent core bugs found via the Metal heap-churn + ASan and fixed** (both
pre-existing, masked by the default allocator zeroing realloc'd slots):
1. `add_value` left `_gd_value.name` uninitialized → `free()` of garbage in
   `_gd_graph_clear`. Fixed: initialize `name = NULL`.
2. `_gd_graph_emit` held a caller `out_desc` that can alias
   `graph->values[].desc`; `import_tensor`/`add_value` realloc that array →
   use-after-free. Fixed: copy `out_desc` on entry before any growth.
Several CPU-era tests used `GD_DEVICE_METAL` as the "absent backend" stand-in;
retargeted to the never-registered `GD_DEVICE_VULKAN` (and the P8/fallback stub
backends moved to free device slots).

### Intent
Stand up the whole pipeline with exactly one op (`add`) so every mechanism is
exercised: device acquisition, registration, shared-buffer storage, leaf
staging, encode/commit, synchronize, download, parity. Everything after M0 is
"add another kernel".

### Changes
- `metal_backend.m`: `MTLCreateSystemDefaultDevice`, `newCommandQueue`,
  `newLibraryWithURL:` (resolve `gradients.metallib` next to the binary; allow
  `GRADIENTS_METALLIB` env override for tests). Build pipeline for `add`.
- Storage alloc/free/host_ptr, upload/download, synchronize as above.
- `metal_exec.m`: compile (per-value shared buffers + leaf staging marks),
  execute (encode `add` for `_GD_OP_ADD`, error for others), value_storage,
  executable_free, execute_until.
- `supports_node`: true only for `_GD_OP_ADD` in M0.
- Registration: `_gd_metal_backend_register(ctx)`; auto-register in
  `gd_context_create` on macOS when `MTLCreateSystemDefaultDevice` succeeds.
  Default device stays CPU. If no GPU, skip silently (CPU-only).
- Remove `metal_probe.*` once the real `init` compiles and links.

### Acceptance
- `gd_graph_compare(g, {CPU,0}, {METAL,0}, NULL)` on `c = a + b` reports no
  mismatch.
- With `GD_FALLBACK_NONE`, a graph containing a non-`add` op compiled to METAL
  returns `GD_ERR_UNSUPPORTED` naming the op; with `GD_FALLBACK_CPU_REF` it runs
  whole-graph on CPU.
- `make check` green on macOS; Metal frameworks linked; no buffer leaks.

---

## M1 — Elementwise and unary kernels

- [ ] Phase complete

### Ops
`mul` (elementwise add/mul share a kernel template), `scale`, `relu`, `silu`,
`copy`, `cast` (F32↔F32 first; integer casts as needed by tests).

### Changes
- Kernels in `kernels.metal`; pipelines added to the cache; `supports_node`
  extended. Broadcast indexing for `mul`/`add` matches `_gd_cpu_k_elementwise`.

### Acceptance
- Per-op parity test (CPU↔METAL) for each, including a broadcast case for the
  binary ops and a negative-input case for relu/silu.

---

## M2 — Matmul, linear, reductions

- [ ] Phase complete

### Ops
`matmul` (+ `trans_a`/`trans_b`), `linear` (+ `trans_b`/bias), `sum`, `mean`,
`softmax`, `rms_norm`, `cross_entropy`.

### Changes
- Naive matmul/linear (2D grid, per-thread dot product). Reductions via
  threadgroup parallel reduction; softmax uses max-subtract for stability to
  match CPU. cross_entropy reduces over the class dim.

### Acceptance
- Parity tests per op; matmul covers transpose flags and a non-square shape;
  reductions cover `keepdim` and a non-last `dim`. Tolerance 1e-4; tighten where
  the GPU result is bit-stable.

---

## M3 — Backward and optimizer

- [ ] Phase complete

### Ops
`relu_bwd`, `silu_bwd`, `softmax_bwd`, `sum_bwd`, `mean_bwd`,
`cross_entropy_bwd`, `reduce_to`, `step_inc`, `adamw_step`, and (optional, debug)
`assert_finite` / `assert_close`.

### Changes
- Mirror the CPU backward kernels. `adamw_step` and `step_inc` mutate parameter/
  state buffers in place (in-place nodes, `n_outputs == 0`); encode them as
  compute dispatches over the param buffers.
- `assert_*`: either implement as a kernel writing a flag buffer that `execute`
  checks post-sync, or leave unsupported (forces CPU fallback for debug graphs).
  Decision recorded at implementation; not required for M4.

### Acceptance
- Parity tests per backward op. `adamw_step` parity: run N steps on CPU and
  Metal from identical init, compare params each step (note: `gd_graph_compare`
  v1 runs a graph twice, so test optimizer parity by two separate single-backend
  runs and compare materialized params, not via the double-run harness).

---

## M4 — Full MLP training parity

- [ ] Phase complete

### Intent
The headline proof: the MLP from [`plan_mlp.md`] trains on Metal and agrees with
CPU_REF on forward activations and gradients.

### Changes
- A test/example building the MLP forward+backward+step graph, compiled and run
  on METAL, with forward-tensor and gradient comparison against CPU_REF.
- Use `gd_graph_compare` for the forward subgraph; for gradients/optimizer use
  separate single-backend runs comparing materialized leaf grads/params (avoids
  the double-execution side-effect limitation of the v1 harness).

### Acceptance
- Forward activations match within 1e-4 node-by-node.
- Leaf gradients match within 1e-4.
- A few AdamW steps keep parameters in agreement within tolerance.

---

## Build system

Already wired by prereqs P7: `.metal → .air → .metallib` via `xcrun metal` /
`metallib`, staged to `build/gradients.metallib`; `.m` compiled with
`-fobjc-arc`; `-framework Metal -framework Foundation -framework QuartzCore`
linked. Confirmed working with the installed Metal toolchain on this machine.

Add: the backend must locate the metallib at runtime. Default to a path relative
to the executable / a compiled-in build path; allow `GRADIENTS_METALLIB`
override so tests run from the repo. If the metallib is absent, `init` fails
gracefully and the backend is not registered (CPU-only), keeping `make check`
green on machines without the shader toolchain.

---

## Out of scope (explicitly deferred)

- Fusion / `lower_graph` / pattern matching (FlashAttention-style lowering).
- Physical layout selection, channels-last, packed/blocked layouts.
- Workspace/arena planning, buffer reuse/liveness (v1 = per-value buffer).
- Kernel cache keyed by shape/dtype/layout; autotuning.
- fp16/bf16/fp8 and quantized kernels; mixed-precision compute policy on GPU.
- Async multi-command-buffer pipelining, events/fences, multi-GPU.
- Tiled/SIMD-group-optimized matmul; MPS/MPSGraph integration.

## Risks

- **Reduction numerics drift.** GPU parallel reductions reorder sums vs CPU.
  Mitigation: stable algorithms (max-subtract softmax, Welford-free simple
  passes), 1e-4 tolerance, parity tests catch regressions; tighten per op.
- **Leaf staging vs borrowing mismatch.** CPU_REF borrows leaf host storage;
  Metal must copy leaves into device buffers. Mitigation: explicit staging step
  in `execute`, covered by M0 parity (add of two leaves).
- **Async-free correctness assumptions.** Per the docs, GPU sees CPU writes only
  before `commit`, CPU sees GPU writes only after `Completed`. Mitigation: leaf
  staging happens before `commit`; `download`/`synchronize` block on the
  in-flight buffer; `execute` waits on any prior in-flight buffer before
  re-staging inputs. The async window (execute return → read) is preserved for
  the common single-run case.
- **metallib discovery at runtime.** Path resolution is environment-sensitive.
  Mitigation: `GRADIENTS_METALLIB` override + graceful non-registration when
  absent; tests set the env to `build/gradients.metallib`.
- **Storage free during in-flight work.** Spec §9 wants async-safe free.
  Mitigation: `executable_free` synchronizes before releasing owned buffers; a
  deferred-free queue keyed on command-buffer completion is deferred until true
  async execution exists.
- **Command-buffer failure is silent without a status check.** Mitigation:
  after every `waitUntilCompleted`, inspect `status`/`error` and surface
  `GD_ERR_BACKEND` with the localized description.
- **Parity harness double-execution.** `gd_graph_compare` runs the graph twice,
  unsafe for in-place training graphs. Mitigation: compare forward subgraphs via
  the harness; compare gradients/params via separate single-backend runs (M3/M4
  acceptance reflects this).
