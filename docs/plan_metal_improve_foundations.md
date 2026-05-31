# gradients.c — Metal GPU_SAFE Performance Foundations

Status: draft v0.1

Goal: improve performance of the current **GPU_SAFE** Metal backend without
requiring graph fusion, layout-planning modes, quantization, or FlashAttention.
This plan keeps the existing correctness-first execution model — one IR node maps
to one backend operation — but removes avoidable host/device churn, forced sync,
copy kernels, and CPU-side encode overhead.

This is deliberately a foundations plan. It should make `GD_DEVICE=metal make gpt`
faster on the tiny GPT example while also building reusable profiling and runtime
mechanisms for all future backends.

Reference docs:
- [`docs/design_spec.md`](design_spec.md) §8–9 (storage/memory), §13 (graph
  execution), §17–18 (backend/compiler), §20 (debugging)
- [`docs/plan_metal_prereqs.md`](plan_metal_prereqs.md) backend seam,
  synchronization, fallback, parity harness
- [`docs/plan_metal.md`](plan_metal.md) current Metal GPU_SAFE backend
- [`docs/plan_gpt.md`](plan_gpt.md) GPT graph and SDPA plan

Official Metal sources captured under `data/metal/`:
- `programming_guide_intro.md` from Apple's archived Metal Programming Guide
  introduction.
- `metal_reference.md` from the current Apple Developer Metal documentation.
- Existing local chapters: `cmd_submission.md`, `buffers.md`,
  `compute_encoder.md`, `functions_libraries.md`.

Metal doc implications for this plan:
- **Command queues are ordered.** Command buffers submitted to one queue execute
  in enqueue order. This supports P6 queued repeated graph runs on a serial queue
  when no CPU staging/readback is required.
- **Command buffers/encoders are transient single-use; queues, buffers,
  libraries, and pipeline states are long-lived.** This supports P8 precomputed
  executable plans and reinforces that PSOs/libraries must stay cached.
- **Command buffer is the coherency boundary.** GPU observes CPU writes only if
  they occur before command-buffer commit; CPU observes GPU writes only after the
  command buffer reaches completed status. This is the core rule for P4/P5/P6.
- **Completed handlers exist but must stay lightweight.** Useful later for
  deferred frees / dirty-state flips, not heavy writeback work.
- **Compute commands in one encoder execute in encode order; prior compute
  results are visible to later commands in that command buffer.** This validates
  the current one-encoder GPU_SAFE node sequence without extra barriers for
  simple buffer dependencies.
- **Threadgroup sizing should use `threadExecutionWidth` and
  `maxTotalThreadsPerThreadgroup`.** This directly guides P9/P10 kernel upgrades.
- **Buffer `contents` is CPU-accessible for appropriate storage modes, but access
  still obeys the command-buffer visibility rules.** Shared/unified memory does
  not remove sync requirements.
- **Metal Performance Shaders is an official optimized compute library.** P11 can
  evaluate MPS for matmul/linear while staying within GPU_SAFE node granularity.
- **Current docs list GPU counters/counter sample buffers.** P1 may start with
  host timing, but future device-side profiling should use official counter
  sampling where available.

Current observed baseline on this machine/config:

```text
make gpt                 # CPU
  ~2.87s / 600 steps  ~= 4.8 ms/step

GD_DEVICE=metal make gpt # Metal GPU_SAFE
  ~13.07s / 600 steps ~= 21.8 ms/step
```

Tiny GPT train graph snapshot:

```text
nodes: 173
values: 221
leaf values: 85
leaf staging/writeback footprint: ~307 KiB each direction per step

op counts:
  copy             36
  matmul           31
  adamw_step       20
  reduce_to        15
  add              15
  linear           14
  rms_norm family  15
  sdpa family       4
```

Main issue: GPU_SAFE is currently correct but over-synchronized, over-copied,
and uses naive kernels. For this tiny workload, dispatch/sync/copy overhead
beats useful GPU math.

---

## Reporting rule for every completed task

When marking any task below as done, update this document with:

1. **Before/after numbers** for relevant workloads.
   - At minimum: `make gpt` CPU and `GD_DEVICE=metal make gpt` Metal timings.
   - Include per-step timing when possible.
   - Include profiling counters when task touches profiling/runtime.
2. **Non-trivial choices made.**
   - API shape, runtime semantics, synchronization decisions, backend behavior.
3. **Learnings / surprises.**
   - Anything useful for future backend, CUDA/Vulkan, fusion, or model work.
4. **Validation.**
   - Tests run, parity checks, ASan/UBSan if relevant.

Rationale: perf work is easy to make local and tribal. Numbers and decisions
must be shared in the docs so other agents can build on them.

---

## Constraints / non-goals

In scope:
- current Metal backend (`GD_DEVICE_METAL`)
- current GPU_SAFE semantics: one graph node -> one backend op/dispatch/call
- profiling across all backends through common `GD_PROFILE` surface
- fewer unnecessary copies/syncs
- faster individual kernels
- better executable planning

Out of scope for this plan:
- `GPU_LAYOUT` mode as separate selectable mode
- `GPU_FUSED` mode / graph fusion
- FlashAttention G3 as production fused SDPA
- quantized / fp16 / bf16 kernels
- multi-device / streams API exposed publicly
- CUDA/Vulkan implementation

Important distinction: some changes below are also prerequisites for future
layout/fusion, but each task must improve or clarify GPU_SAFE independently.

---

## Phase status checklist

- [x] P0 — Common `GD_PROFILE` infrastructure for all backends
- [x] P1 — Metal profiling counters and per-op timing/cost report
- [x] P2 — Honest benchmark/release build target
- [x] P3 — Device-resident parameters and optimizer state
- [ ] P4 — Lazy writeback: download only on explicit CPU read/sync need
- [ ] P5 — Dirty-tracked staging: upload only changed host leaves
- [x] P6 — Async run boundary: remove unconditional post-run wait where safe
- [x] P7 — Zero-copy reshape/copy aliasing for metadata-only copies
- [ ] P8 — Precomputed Metal executable encode plan
- [ ] P9 — GPU_SAFE kernel upgrades batch 1: reductions/norm/CE/reduce_to
- [ ] P10 — GPU_SAFE kernel upgrades batch 2: matmul/linear
- [ ] P11 — Optional MPS-backed matmul/linear path
- [ ] P12 — GPT example/device ergonomics and regression benchmark

---

## P0 — Common `GD_PROFILE` infrastructure

- [x] Phase complete

### Intent
Add one backend-agnostic profiling switch rather than Metal-only ad hoc logging.

Environment variable:

```text
GD_PROFILE=0|1|summary|json|trace
```

Suggested meanings:
- unset / `0`: disabled
- `1` or `summary`: human-readable summary at process exit or context destroy
- `json`: machine-readable JSON report
- `trace`: verbose per-run/per-node events (expensive; debug only)

Optional filters:

```text
GD_PROFILE_BACKEND=cpu,metal
GD_PROFILE_PATH=build/profile.json
```

### Common counters
Track at context/backend/executable level:

```text
backend name/device
number of graph compiles
number of graph runs
nodes executed
op counts
wall time in compile/run/synchronize/download/upload
bytes uploaded/downloaded
number of explicit synchronizations
number of blocking downloads
number of backend allocations/frees
allocated bytes / peak live bytes when cheap to know
```

### Design notes
- Profiling must be low-overhead when disabled.
- Avoid public ABI churn if possible: internal profiler in `gd_context` is fine.
- Use monotonic host timer first. Backend device timestamps can come later.
- CPU_REF should also report op counts and CPU run time, so Metal speedups have a baseline.

### Acceptance
- `GD_PROFILE=summary make gpt` prints CPU profile.
- `GD_PROFILE=summary GD_DEVICE=metal make gpt` prints Metal profile.
- `make check` unchanged when profiling disabled.
- Document sample output and baseline numbers here.

### Completion notes
Completed in first implementation pass.

Numbers (`GD_PROFILE=summary`):
- Debug CPU `make gpt`: `runs=601`, `nodes=103843`, `run_ms≈2707.7`.
- Release CPU `make bench-gpt`: `runs=601`, `nodes=103843`, `run_ms≈565.1`.
- Release Metal `GD_DEVICE=metal make bench-gpt`: `runs=601`,
  `nodes=103843`, `run_ms≈12841.4`.

Choices:
- `GD_PROFILE=summary|json|trace` implemented internally on `gd_context`; no
  public ABI change.
- `GD_PROFILE_PATH` writes summary/json to a file; default summary goes to
  `stderr` at `gd_context_destroy`.
- `GD_PROFILE_BACKEND=cpu_ref,metal` filters which backend stats are collected
  and printed (matches backend vtable name or device type name).
- Common counters are recorded in generic call sites: graph compile/run,
  `gd_synchronize`, storage upload/download/allocation/free.
- Op counts are counted on graph runs, not compiles.
- Host wall-clock uses `gettimeofday` for portability across the current C/Obj-C
  build; device counters remain future work.

Learnings:
- CPU_REF shows zero graph syncs for GPT because the example never explicitly
  synchronizes CPU and CPU downloads are immediate.
- Metal staging/writeback through CPU storage also appears as CPU_REF transfer
  counters, which is useful: it reveals current staging uses CPU backend copy
  paths heavily.

Validation:
- `GD_PROFILE=summary make gpt`
- `GD_PROFILE=json GD_PROFILE_PATH=/tmp/gd_profile.json make gpt` + JSON parsed
  with `python3 -m json.tool`
- `make check`

---

## P1 — Metal profiling counters and per-op timing/cost report

- [x] Phase complete

### Intent
Make current Metal bottlenecks visible before changing runtime behavior.

### Metrics
At minimum report:

```text
per graph run:
  stage_leaves time + bytes + count
  encode time
  command buffer wait time
  writeback_externals time + bytes + count
  total run time

per op kind:
  node count
  dispatch count
  total encoded dispatches
  approximate logical bytes/elements when easy

per executable:
  number of values
  number of leaf/external values
  number of produced values
  total allocated Metal bytes
  total leaf bytes staged/writeback-eligible
```

If device timing is simple enough, add Metal command buffer GPU time using
available timestamps. If not, host-side timing is sufficient for this phase.

### Acceptance
- Profile identifies how much of `GD_DEVICE=metal make gpt` is stage, encode,
  GPU wait, and writeback.
- Profile prints top op counts for GPT train graph.
- Numbers added to this doc.

### Completion notes
Completed in first implementation pass.

Release Metal profile (`GD_PROFILE=summary GD_DEVICE=metal make bench-gpt`):

```text
metal: compiles=2 compile_ms≈0.751 runs=601 run_ms≈12841.4 nodes=103843
metal: downloads=8 download_bytes=2076
metal: allocs=286 allocated_bytes=864332 peak_live_bytes=691532
metal event stage_leaves: count=601 items=51022 bytes=188512192 ms≈14.9
metal event metal_encode: count=601 items=103843 ms≈63.9
metal event metal_wait: count=601 items=601 ms≈12740.7
metal event writeback_externals: count=601 items=51022 bytes=188512192 ms≈20.1
```

Top op counts across train+eval:

```text
copy=21608 matmul=18601 adamw_step=12000 add=9004 reduce_to=9000
linear=8414 mul=3602 rms_norm family=9005 sdpa family=2402 rope family=4804
```

Choices:
- Metal-specific runtime events are reported through the common profiler as named
  backend events: `stage_leaves`, `metal_encode`, `metal_wait`,
  `writeback_externals`.
- Host-side event timing used first. Official Metal counter sample buffers remain
  future work because current bottleneck is already obvious.

Learnings:
- Actual stage/writeback memcpy time is small (~35 ms combined over 600 steps),
  but it performs ~51k leaf copies each direction and forces the current sync
  model. P4/P5 still matter for async/device-resident work.
- `metal_wait` dominates. Under current synchronous GPU_SAFE this includes all
  GPU execution time for 173 tiny dispatches per train step.
- Release CPU improves sharply; Metal does not because GPU kernels and dispatch
  count dominate, not host compiler optimization.

Validation:
- `GD_PROFILE=summary GD_DEVICE=metal make bench-gpt`
- `make check`

---

## P2 — Honest benchmark/release build target

- [x] Phase complete

### Intent
Current Makefile defaults to `-O0 -g3`. Good for dev, bad for perf claims.
Add a repeatable benchmark command without changing correctness/debug defaults.

### Proposed commands

```text
make bench-gpt
make bench-gpt GD_DEVICE=metal
```

or:

```text
make bench CFLAGS='-std=c11 -O2 -DNDEBUG ...'
```

Preferred: dedicated target that builds release objects under separate build dir
(e.g. `build-release/`) so debug and release artifacts do not mix.

### Acceptance
- CPU and Metal GPT timings can be collected in debug and release modes.
- This doc records both, so future perf work does not compare debug CPU vs debug
  Metal accidentally.

### Completion notes
Completed in first implementation pass.

Commands added:

```text
make bench-gpt
GD_DEVICE=metal make bench-gpt
```

Implementation:
- `BENCH_CFLAGS` defaults to `-O2 -DNDEBUG` plus the same warnings.
- Target builds under `build-release/` by recursively invoking `make` with
  `BUILD_DIR=build-release`, leaving debug `build/` untouched.

Numbers including make target overhead after release build existed:

```text
make bench-gpt                 real≈0.77s  ~= 1.3 ms/step wall
GD_DEVICE=metal make bench-gpt real≈13.09s ~= 21.8 ms/step wall
```

Profiler run time inside graph:

```text
CPU release:   run_ms≈565.1    ~= 0.94 ms/step over 601 runs
Metal release: run_ms≈12841.4  ~= 21.37 ms/run over 601 runs
```

Choices:
- Kept debug defaults unchanged.
- Used a separate build dir instead of requiring users to manually clean/rebuild
  with different `CFLAGS`.

Learnings:
- CPU release speedup is large versus debug (`~2.87s` -> `~0.77s` wall).
- Metal remains roughly unchanged versus debug because current cost is GPU
  dispatch/kernel execution and synchronization, not C host optimization.

Validation:
- `make bench-gpt`
- `GD_DEVICE=metal make bench-gpt`
- `make check`

---

## P3 — Device-resident parameters and optimizer state

- [x] Phase complete

### Intent
Stop treating Metal params/state as CPU leaves copied into shadow Metal buffers
every step. Parameters and optimizer state should live on target device when the
model is created for that device.

### Current gap
`src/nn/nn.c:create_param()` hardcodes:

```c
gd_device cpu = {GD_DEVICE_CPU, 0};
```

Metal compile currently allocates one Metal buffer for every graph value,
including external leaves, then stages CPU leaf bytes before each run and writes
all leaf bytes back after completion.

### Proposed design
- Add device selection to GPT/model creation, or honor context default device.
- Create parameter tensors on target device.
- Optimizer state should follow parameter device by default.
- For Metal external values already backed by Metal storage, executable value
  should alias that storage instead of allocating a shadow buffer.
- CPU initialization remains through `gd_tensor_copy_from_cpu`, which uploads
  once to device storage.

Potential API options:

```c
gd_context_set_default_device(ctx, target);
gd_gpt_create(ctx, &cfg, seed, &model); /* uses default device */
```

or extend config:

```c
cfg.device = target;
```

Prefer context default if it aligns with existing design.

### Acceptance
- GPT params can be created on Metal.
- AdamW state tensors are Metal when params are Metal.
- Metal executable aliases Metal leaves instead of staging CPU leaves for params/state.
- CPU read of params still works through explicit blocking download.
- GPT train parity still passes.
- Profile shows reduced stage/writeback bytes/counts.

### Completion notes
Completed.

Numbers (`GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make bench-gpt`):

```text
before P3/P6:
  uploads=0 downloads=8
  allocs=286 allocated_bytes=864332
  stage_leaves items=51022 bytes=188512192 ms≈14.9
  writeback_externals items=51022 bytes=188512192 ms≈20.1
  metal_wait count=601 ms≈12740.7
  wall warm release≈13.09s

after P3/P6:
  uploads=24 upload_bytes=78660   # one-time CPU -> Metal initialization
  downloads=8 download_bytes=2076 # loss/logits reads only
  allocs=264 allocated_bytes=785740
  stage_leaves items=0 bytes=0 ms≈0.0
  writeback_externals items=0 bytes=0 ms≈0.0
  metal_wait count=8 ms≈7950.9
  metal_encode ms≈4450.8
  wall warm release≈12.58s
```

Choices:
- `gd_gpt_create` now honors `gd_context_default_device(ctx)` for parameter
  creation. No public GPT config change.
- `examples/gpt/gpt.c` sets the context default device after resolving
  `GD_DEVICE`, and creates token/position/target tensors on the same target.
- AdamW state already followed parameter device through `make_state_like`; the
  optimizer step scalar follows context default device.
- Metal compile aliases external leaf storage when the external tensor is already
  Metal-backed, contiguous, same device index, and has zero storage offset.
- CPU-backed external leaves retain old shadow-buffer behavior for safety.

Learnings:
- Device-resident leaves remove all per-step staging/writeback in GPT.
- Allocation count drops because Metal no longer creates shadow buffers for
  Metal-backed leaves (`286 -> 264` allocations in profile).
- Wall-time win is modest because remaining cost is still tiny-kernel execution
  and command-buffer waits.

Validation:
- `GD_DEVICE=metal make gpt`
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make bench-gpt`
- `make check`

---

## P4 — Lazy writeback: download only on explicit CPU read/sync need

- [ ] Phase complete

### Intent
Remove unconditional `writeback_externals()` after every Metal graph run.

### Current behavior
`metal_execute_range()` commits work, then immediately calls `metal_synchronize()`.
`metal_synchronize()` calls `writeback_externals()` for every external leaf.
This makes params, grads, and optimizer state visible to CPU after each run, but
it destroys GPU training throughput.

### Proposed behavior
- Metal graph run marks relevant external tensors/storage as device-dirty.
- Do not copy external values back immediately.
- `gd_tensor_copy_to_cpu`, `gd_storage_copy_to_cpu`, materialization, debug dump,
  and explicit `gd_synchronize` become the readback boundaries.
- Readback should only download requested storage/range when possible.

### Semantics decision needed
`gd_synchronize(ctx, metal)` currently means backend work complete. Should it
also mean all dirty external leaves are copied to CPU? Options:

1. **Synchronize only waits.** CPU visibility requires explicit copy/read.
2. **Synchronize waits + flushes dirty host mirrors.** More conservative, more expensive.

Recommendation: option 1 for performance, but document clearly. Existing public
copy APIs already promise CPU-visible bytes when they return.

### Acceptance
- `gd_graph_run(train)` no longer downloads all params/state each step.
- `gd_tensor_copy_to_cpu(ctx, loss, ...)` still returns correct loss.
- Reading a param after training returns updated value.
- Existing tests updated to use explicit reads where needed.
- Profile shows writeback cost removed or sharply reduced.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P5 — Dirty-tracked staging: upload only changed host leaves

- [ ] Phase complete

### Intent
Stop uploading all external leaves before every Metal run.

### Current behavior
`stage_leaves()` copies every external leaf into its Metal buffer every run.
For GPT, tokens/positions/targets are static, params are updated by GPU, and
optimizer state is updated by GPU — most leaves should not need host upload each
step.

### Proposed design
Add storage/tensor version or dirty flags:

```text
host_dirty
device_dirty[backend/device]
last_uploaded_version per executable value
```

Rules:
- `gd_tensor_copy_from_cpu` marks host source/current bytes dirty for device mirrors.
- Metal kernels writing a value mark Metal bytes current and host mirror stale.
- `stage_leaves()` uploads only leaves whose host version changed since last upload.
- If a leaf is device-resident and current on Metal, no upload.

### Acceptance
- Static GPT inputs upload once, not every step.
- GPU-updated params/state are not re-uploaded from stale CPU mirrors.
- Correctness preserved under repeated training runs and input changes.
- Profile reports staged leaf count/bytes near zero after warmup except changed inputs.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P6 — Async run boundary: remove unconditional post-run wait where safe

- [x] Phase complete

### Intent
Let `gd_graph_run()` enqueue Metal work and return when no immediate CPU-visible
side effect is required. Serial command queue preserves order across repeated
runs.

### Current behavior
Metal execute is effectively synchronous because it waits immediately after
commit.

### Proposed design
- Keep pre-run synchronization only when CPU staging needs to mutate buffers that
  might still be in use.
- If no staging is needed, allow command buffers to queue on same serial queue.
- Blocking operations (`gd_tensor_copy_to_cpu`, `gd_synchronize`, executable
  free, graph reset/destroy where needed) wait.
- Ensure buffer lifetime remains safe until command completion.

### Acceptance
- Multiple `gd_graph_run(train)` calls can queue without immediate wait when no
  CPU read occurs.
- Loss printing every 100 steps syncs only at print step.
- Tests using immediate reads still pass because reads synchronize.
- Profile shows lower per-step host wait in no-read loops.

### Completion notes
Completed for safe/device-resident executables.

Numbers after P3/P6 (`GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make bench-gpt`):

```text
metal_wait count: 601 -> 8
metal_wait ms:    ~12740.7 -> ~7950.9
metal_encode ms:  ~63.9 -> ~4450.8
wall warm release: ~13.09s -> ~12.58s
```

Choices:
- Metal executables now track whether any CPU-backed external leaves require
  staging/writeback.
- If staging is needed, old conservative behavior remains: wait before CPU
  shadow-buffer mutation and wait/writeback after run.
- If staging/writeback is not needed (device-resident GPT path), `gd_graph_run`
  commits work and returns; blocking downloads and `gd_synchronize` wait on the
  latest queued command buffer.
- Waiting on the latest command buffer is sufficient for one serial Metal queue:
  earlier command buffers are ordered before it by Metal queue semantics.

Learnings:
- Async boundary moved wait time from every `gd_graph_run` to the 8 blocking
  downloads in GPT (loss prints + final logits), as intended.
- Total wall-time gain is modest because the GPU still executes the same many
  tiny kernels. Host encode time rises because the CPU can now run ahead until
  Metal/driver backpressure appears.

Validation:
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make bench-gpt`
- `make check`

---

## P7 — Zero-copy reshape/copy aliasing for metadata-only copies

- [x] Phase complete

### Intent
Remove copy dispatches for reshape/view operations that do not require physical
movement.

### Current behavior
GPT graph has 36 `_GD_OP_COPY` nodes, many from reshaping q/k/v and attention
outputs. Metal allocates output buffers and dispatches copy kernels.

### Proposed design
At graph/compile time, detect copy nodes that are pure contiguous reshape:
- same dtype
- same numel
- contiguous-compatible input/output
- no required data reordering

Then executable value for output aliases input storage/buffer with new logical
desc and same offset. No allocation, no dispatch.

This stays GPU_SAFE because op semantics are unchanged; backend just chooses a
zero-copy implementation for an identity data movement.

### Acceptance
- GPT copy dispatch count decreases significantly (target: remove reshape-only copies).
- Output values retain correct shape/desc.
- Backward and parity tests pass.
- Profile shows fewer dispatches and lower allocated bytes.

### Completion notes
Completed in first implementation pass.

Numbers:
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make gpt` reports
  `event=copy_alias_skip count=9608 items=9608` across 601 runs.
- Wall time stayed roughly flat/noisy (`GD_DEVICE=metal make gpt` around
  `13.00s` after change vs `13.07s` baseline). This confirms copy dispatches are
  not the dominant bottleneck for this tiny graph; GPU wait across remaining
  kernels still dominates.

Choices:
- Implemented in Metal compile, not graph IR: eligible `_GD_OP_COPY` output
  values retain/alias the input Metal storage and skip dispatch at encode time.
- Eligibility is conservative: one input, one output, output is not external,
  same dtype/device/quant, zero storage offsets, both logical layouts contiguous,
  and equal byte size.
- Added `copy_alias_skip` profiler event to make skipped dispatches visible.

Learnings:
- GPT has `21608` copy op executions across train+eval; `9608` were safe
  metadata-only aliases with current conservative checks.
- Remaining copy ops likely include backward/support copies or cases where a real
  materialization target is required; do not broaden without node-by-node parity.

Validation:
- `./build/tests/test_metal`
- `./build/tests/test_metal_gpt`
- `./build/tests/test_metal_gpt_train`
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal GD_DEVICE=metal make gpt`

---

## P8 — Precomputed Metal executable encode plan

- [ ] Phase complete

### Intent
Reduce CPU overhead in the encode loop while keeping one dispatch per node.

### Current behavior
Execute-time code switches on node op, looks up pipelines, builds params, binds
buffers, computes dispatch sizes.

### Proposed design
During `metal_compile`, build per-node encode records:

```text
op
pipeline pointer
extra pipeline pointer when needed (e.g. sdpa_bwd dkv)
input value ids
output value ids
prebuilt params blob
params size
thread grid
threadgroup size
flags/variant
```

At execute time:
- refresh dynamic pointers only when storage aliasing changed
- bind buffers
- set prebuilt params
- dispatch

### Acceptance
- Same graph output/parity.
- Lower encode time in `GD_PROFILE` for GPT.
- Code remains maintainable; complex ops may keep custom encoder with cached params.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P9 — GPU_SAFE kernel upgrades batch 1: reductions/norm/CE/reduce_to

- [ ] Phase complete

### Intent
Improve obvious slow scalar/reference kernels while keeping one kernel per op.

### Targets
- `sum` / `mean`: threadgroup reductions instead of one thread doing whole row.
- `softmax`: threadgroup row reduction for max/sum.
- `rms_norm`: threadgroup reduction over last dim for larger rows.
- `rms_norm_bwd` / `rms_norm_wbwd`: parallelize reductions.
- `cross_entropy`: parallel reduction instead of single-thread scalar loss.
- `reduce_to`: avoid single-thread scatter-add where possible.
- `embedding_bwd`: consider atomics or grouped accumulation instead of serial path.

### Acceptance
- Per-op parity tests pass at existing tolerances.
- GPT training parity remains green.
- Profile shows reduced GPU wait time for affected ops.
- Any numerical tolerance change documented here.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P10 — GPU_SAFE kernel upgrades batch 2: matmul/linear

- [ ] Phase complete

### Intent
Replace naive scalar matmul/linear kernels with tiled GPU kernels while still
mapping one IR node to one kernel dispatch.

### Current behavior
`gd_matmul` and `gd_linear` use one thread per output element and scalar loop over K.
This is correct but poor on Apple GPUs.

### Proposed work
- Implement threadgroup-tiled F32 GEMM for common contiguous cases.
- Keep current generic kernel as fallback for batched/broadcast/transposed edge cases.
- Specialize `linear` as matrix multiply over rows x out_features.
- Add simple shape-based dispatch variant selection in Metal backend.

### Acceptance
- Matmul/linear Metal parity tests pass.
- GPT train parity passes.
- Profile shows lower GPU wait in matmul/linear-heavy graphs.
- Document tile sizes, Apple GPU observations, and fallback conditions.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P11 — Optional MPS-backed matmul/linear path

- [ ] Phase complete

### Intent
Evaluate using Metal Performance Shaders for matmul/linear while preserving
GPU_SAFE node granularity.

### Design
One IR matmul/linear node may encode one MPS matrix multiplication operation
instead of custom compute kernel. This is still GPU_SAFE: no graph fusion, no
semantic change.

### Questions
- Does MPS overhead pay off for tiny GPT shapes?
- Does it handle needed transposes/batches cleanly?
- Can it share existing `MTLBuffer` storage without extra copies?
- How does it affect command encoder structure?

### Acceptance
- Experimental path behind env/config flag, e.g. `GD_METAL_MPS=1`.
- Benchmarks compare custom naive, custom tiled (if present), and MPS.
- Keep only if numbers justify complexity.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## P12 — GPT example/device ergonomics and regression benchmark

- [ ] Phase complete

### Intent
Make GPT benchmark and device selection representative and repeatable.

### Work
- Let GPT model params be created on requested device.
- Avoid using `gd_synchronize(ctx, metal)` as Metal availability probe if that
  becomes expensive or semantically narrower.
- Add fixed benchmark output:

```text
device
steps
compile time
train time
time/step
loss first/last
accuracy
profile summary path when enabled
```

- Consider env knobs:

```text
GD_GPT_STEPS=600
GD_GPT_BATCH=1
GD_GPT_SEQ=16
GD_GPT_DMODEL=32
```

Keep default tiny overfit workload stable for regression.

### Acceptance
- `make bench-gpt` and `GD_DEVICE=metal make bench-gpt` produce comparable output.
- This doc records benchmark table across phases.

### Completion notes
_To fill when done: numbers, choices, learnings, validation._

---

## Suggested first sprint

Do these first; they should give highest signal and biggest immediate win:

1. P0 — common `GD_PROFILE`
2. P1 — Metal profile counters
3. P2 — release/bench target
4. P7 — zero-copy reshape/copy aliasing
5. P4/P5 — lazy writeback + dirty staging
6. P3 — device-resident params/state

Reasoning:
- P0–P2 prevent blind perf work.
- P7 is small and removes many dispatches in GPT.
- P4/P5 attack current sync/copy tax.
- P3 is highest-impact runtime semantic fix but touches model/device behavior,
  optimizer state, and external leaf aliasing, so it benefits from profiling first.

---

## Benchmark table

Fill this table as phases land.

| Phase | Build | Command | Time | ms/step | Notes |
|---|---|---|---:|---:|---|
| baseline | debug `-O0` | `make gpt` | ~2.87s | ~4.8 | CPU |
| baseline | debug `-O0` | `GD_DEVICE=metal make gpt` | ~13.07s | ~21.8 | Metal GPU_SAFE |
| P2 | release `-O2` | `make bench-gpt` | ~0.78s | ~1.3 | CPU warm build |
| P2 | release `-O2` | `GD_DEVICE=metal make bench-gpt` | ~13.09s | ~21.8 | before P3/P6 |
| P3+P6 | release `-O2` | `GD_DEVICE=metal make bench-gpt` | ~12.58s | ~21.0 | device-resident leaves, async run boundary |

---

## Risks

- **Stale host mirrors.** Lazy writeback/dirty tracking can make CPU reads wrong
  if copy paths do not synchronize/download correctly. Mitigate with tests that
  train on Metal then read params/loss/grads explicitly.
- **Changed sync semantics.** Users may assume `gd_graph_run` means CPU-visible
  side effects. Existing docs allow async-ready design, but behavior changes must
  be documented.
- **Aliasing bugs.** Zero-copy reshape can corrupt results if a copy was actually
  needed. Mitigate with strict eligibility checks and parity tests.
- **Profiler overhead.** Profiling must not distort tiny workload too much; keep
  disabled path near-zero overhead.
- **Numerics drift.** Faster reductions/tiled GEMM may change accumulation order.
  Keep parity gates and document tolerance decisions.
- **MPS complexity.** MPS may help large GEMMs but hurt tiny shapes; keep optional
  until numbers prove value.
