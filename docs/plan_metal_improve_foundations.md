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
- [x] P4 — Lazy writeback: download only on explicit CPU read/sync need (CPU-backed path; device-resident already lazy)
- [x] P5 — Dirty-tracked staging: upload only changed host leaves
- [x] P6 — Async run boundary: remove unconditional post-run wait where safe
- [x] P7 — Zero-copy reshape/copy aliasing for metadata-only copies
- [x] P8 — Precomputed Metal executable encode plan (pipeline resolution cached; full ICB deferred)
- [x] P9 — GPU_SAFE kernel upgrades batch 1: reductions/norm/CE/reduce_to (hot offenders; per-row polish + sdpa_bwd deferred)
- [x] P10 — GPU_SAFE kernel upgrades batch 2: matmul/linear
- [ ] P11 — Optional MPS-backed matmul/linear path
- [ ] P12 — GPT example/device ergonomics and regression benchmark
- [x] P13 — GPU_SAFE kernel upgrades batch 3: per-row threadgroup reductions (RMSNorm family)

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

- [x] Phase complete

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
Scope: the **device-resident** path was already lazy after P3/P6 (params/state
are Metal storage; reads go through `metal_download` -> `synchronize`). The
remaining work was the **CPU-backed Metal path**, where a param/grad lives in the
Metal *shadow* buffer but the user reads the leaf's *CPU* storage, which has no
link to the executable. P4 adds the missing read->flush bridge so that path can
run async too.

Mechanism (lazy writeback markers):
- `gd_storage` gains an optional pending-flush marker `(pending_backend,
  pending_cookie)`. A backend that computed newer bytes but deferred the copy
  registers itself; the next host read resolves it.
- New backend vtable hook `flush_pending(self, cookie)` (nullable; CPU_REF leaves
  it NULL).
- `gd_storage_copy_to_cpu` and `gd_storage_data_cpu` call
  `storage_resolve_pending()` before reading; `gd_storage_copy_from_cpu` clears
  the marker (a host write supersedes a deferred device write).
- Metal: `metal_execute_range` no longer waits+writes-back at the end of a
  CPU-backed run. It marks each mutated CPU-backed leaf pending and adds the
  executable to a pending set. `metal_flush_pending` / `gd_synchronize` /
  `metal_executable_free` drain the set: wait the (serial) in-flight command
  buffer once, then `writeback_externals` for every pending executable.
- Correctness rests on P5: "skip staging unchanged host leaves" keeps the shadow
  buffers authoritative across runs, so deferring writeback never lets a later
  run re-stage stale CPU bytes over GPU-updated params.

Numbers (CPU-backed params, Metal graph, 100 train steps, read loss once at end;
`GD_PROFILE=summary`):

```text
before P4 (eager): metal_wait ~100, writeback_externals ~100 (one per run)
after  P4 (lazy):  metal_wait = 1, writeback_externals = 1,
                   stage_leaves items = 85 total (only the first run stages)
                   run_ms ~112 over 100 runs (~1.12 ms/run, async return)
final loss read once = 0.00302 (matches per-step-read reference)
```

The CPU-backed path is now as async as the device-resident path: per-run
wait+writeback collapsed from O(steps) to 1 (at the read).

Semantics decision: chose **option 1** — `gd_synchronize` (and any blocking
read) flushes pending writeback, but `gd_graph_run` does not. Public copy APIs
still guarantee CPU-visible bytes when they return, because they resolve pending
first.

Choices / safety:
- Pending markers are unretained backend+cookie pointers; the only producer
  (Metal) drains and clears them in `synchronize`/`flush_pending`, and
  `metal_executable_free` synchronizes before freeing, so a cookie can never
  dangle past its executable.
- A pending *set* of executables (not a single one) handles consecutive async
  runs and multiple graphs without losing any writeback (serial queue => one
  wait completes all).
- `writeback_externals` copies shadow->CPU via `gd_storage_copy_from_cpu`, which
  also clears the leaf's pending marker, keeping markers and the pending set
  consistent.

Learnings:
- The subtle hazard was re-staging: without P5, a deferred-writeback run followed
  by another run would upload stale CPU params over the GPU's in-place update.
  P5's version-gated staging is precisely what makes P4 safe — the two features
  are co-dependent for the CPU-backed path.
- Device-residency (P3) remains the recommended path; P4 mainly closes the gap
  for mixed CPU-param / Metal-compute and debug workflows.

Validation:
- `make check` — including `test_metal_gpt_train` (CPU params + Metal graph,
  multi-step, reads params after) and a new `test_metal_lazy_writeback` (25
  in-place AdamW steps on Metal with no intermediate reads, single read-back
  matches the double-precision reference).
- `GD_PROFILE=summary` wait/writeback/stage counts before vs after.

---

## P5 — Dirty-tracked staging: upload only changed host leaves

- [x] Phase complete

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
Completed for CPU-backed Metal shadow buffers.

Numbers:
- Device-resident GPT path remains at `stage_leaves items=0 bytes=0`, as in P3.
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal ./build/tests/test_metal` reports
  small staged footprint across the Metal test suite: `stage_leaves count=16
  items=53 bytes=1348 ms≈0.007`, with repeated CPU-backed graph reuse covered by
  a test update.

Choices:
- Added a per-storage version counter that increments on successful
  `gd_storage_copy_from_cpu` (host/API writes).
- Metal executable values remember `staged_version`; CPU-backed leaves are
  uploaded only when their source storage version changed or they have never
  been staged.
- Metal now identifies CPU-backed external values that actually need writeback:
  node outputs bound with `emit_to`, plus known in-place mutators
  (`step_inc`, `adamw_step` param/m/v). Read-only CPU leaves are no longer
  written back just because they are external.
- If no CPU leaf changed, Metal skips pre-run synchronization and queues the next
  command buffer, preserving Metal's command-buffer coherency rule because no CPU
  mutation of shadow buffers occurs.

Learnings:
- Dirty staging matters mainly for CPU-backed graphs and mixed debug workflows;
  the optimized GPT path is now fully device-resident, so it naturally stages
  nothing per step.
- Precise writeback targeting is required before version-based staging is useful;
  writing back read-only leaves would bump their versions and force useless
  re-uploads.

Validation:
- Added `test_metal_add_direct` coverage for reusing a compiled Metal graph after
  changing a CPU-backed input tensor; output reflects the new host data.
- `GD_PROFILE=summary GD_PROFILE_BACKEND=metal ./build/tests/test_metal`
- `make check`

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

- [x] Phase complete (pipeline resolution cached at compile; full prerecord/ICB deferred)

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
Measured the host encode cost first to scope the work honestly:

```text
6-layer 12.39M, T=64 (per run): encode ~0.26 ms vs GPU step ~345 ms
tiny 19.6K, T=16  (per run):     encode ~0.10 ms vs GPU step ~3.3 ms (metal_wait ~3.5 ms)
```

Host encode is a tiny fraction even on the tiny high-iteration model, so this is
a host-side micro-opt / architecture cleanup, not a wall-time lever on current
workloads.

Implemented: per-node pipeline resolution is now done **once at
`metal_compile`** (stored on the executable as unretained `__bridge` pointers in
`node_pso` / `node_pso2`; the `GDMetalState` retains the pipelines for the
context lifetime). The per-run encode loop (and the trace loop) pass the cached
PSOs to `encode_node`, so the hot path no longer does an `NSDictionary` lookup
(NSNumber boxing + hash) per node per run. `sdpa_bwd`'s second pipeline
(`gd_sdpa_bwd_dkv`) is cached in `node_pso2`.

Numbers: encode/run for the 6-layer model went ~0.26 ms -> ~0.24-0.31 ms (within
run-to-run noise). End-to-end step unchanged (GPU-bound).

Choices:
- Cached only pipeline state, not a full prerecorded dispatch/param plan. The
  remaining encode cost is the unavoidable `setBuffer`/`setBytes`/`dispatchThreads`
  Obj-C calls, which a params/dispatch cache would not remove; the existing
  per-op encoders stay (the spec explicitly allows complex ops to keep custom
  encoders).
- Plan pointers are unretained `__bridge` (no refcount churn) because pipeline
  lifetime strictly outlives executables; the arrays are freed in
  `metal_executable_free`.

Learnings:
- The per-node dictionary lookup was a minor slice of encode; the ObjC encoding
  calls dominate. Eliminating those needs **indirect command buffers (ICB)** to
  prerecord the whole dispatch sequence once and re-execute it, which is a much
  larger change and only pays off if a workload becomes host-encode-bound (e.g.
  many tiny kernels with async submission and no per-iter sync). Deferred until
  measurements justify it.
- For current GPT workloads the bottleneck is GPU execution (`sdpa_bwd`), not
  host encode, so further host-side work is not warranted now.

Validation:
- `make check` (all parity + GPT train parity green)
- `GD_PROFILE=summary` encode timings before/after on 6-layer and tiny configs

---

## P9 — GPU_SAFE kernel upgrades batch 1: reductions/norm/CE/reduce_to

- [x] Phase complete (hot offenders; per-row reduction polish + sdpa_bwd deferred)

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
Data-driven via a new per-op GPU-time attribution path (see below). On a 1-layer
4.2M model at T=64 (`GD_PROFILE=trace`, one command buffer per node, host-timed),
the single-thread kernels dominated a training step at **3940 ms total**:

```text
reduce_to     3593.4 ms   (single thread!)
embedding_bwd  142.3 ms   (single thread: zero 2.56M + scatter)
cross_entropy  100.7 ms   (single thread scalar reduction)
sdpa_bwd        49.8 ms
... everything else < 10 ms (matmul 9.1, etc.)
```

Fixed the three single-thread offenders:
- **`reduce_to`**: rewritten to one thread per *target* element that sums only
  the `go` positions broadcasting to it (iterating the reduced dims via strides).
  Total work O(go_numel) across `target_numel` threads instead of one serial
  scan. `3593 ms -> ~4 ms`.
- **`embedding_bwd`**: one thread per `dtable` element `(v,c)` sums gradient rows
  whose id == v. No atomics, no serial zero+scatter. `142 ms -> ~1.4 ms`.
- **`cross_entropy`**: single threadgroup, each thread accumulates a strided set
  of positions, then a threadgroup reduction yields the mean. `100 ms -> ~3.4 ms`.

After these, the 1-layer trace step is **121 ms** (32x), and the real workload
(full 12.39M params, 6 layers, T=64, non-trace train step) dropped from
**~11,400 ms/iter to ~372 ms/iter (~30x)** — Metal is now faster than the CPU
reference (~2,660 ms/iter) instead of 4.3x slower.

Profiler addition (part of P1, landed here because it was needed to localize the
cost): `GD_PROFILE=trace` makes the Metal backend encode one command buffer per
node and wait on each, accumulating host-measured time per op kind. The summary
then prints `op=<name> count=N gpu_ms=...`. This serial mode is for profiling
only; normal execution batches all nodes into one command buffer.

Choices:
- Per-target `reduce_to` iterates reduced dims only (O(go_numel) total), not the
  O(target*go) per-target full scan that an earlier naive attempt used.
- `embedding_bwd` uses the scatter-free "gather by id" formulation to stay
  deterministic and atomic-free; cost grows with n_ids but is fully parallel.
- `cross_entropy` threadgroup is capped at 256 (`GD_CE_TG`) to match the
  threadgroup `partial[]` array; positions are handled grid-stride within the
  single group.

Learnings:
- A single-thread GPU kernel is ~50-100x slower than the same loop on a CPU core,
  so any `gid != 0` kernel is a latent cliff. Trace attribution turned an opaque
  "11s, GPU slower than CPU" into a one-line diagnosis.
- The tied-embedding LM head produces a `[B,V,d] -> [V,d]` grad reduction; with
  `V=8000, d=320` that single-thread `reduce_to` alone was 91% of the step.

Deferred (not hot on current workloads, tracked as their own phases):
- Per-row threadgroup reductions for `sum`/`mean`/`softmax`/`rms_norm*`
  (`rms_norm_wbwd` is now the largest of these at ~9 ms; acceptable for now).
  **Tracked as P13.**
- `sdpa_bwd` is now the largest single op (~55 ms in trace); it is the O(T^2)
  reference recompute and belongs to the deferred FlashAttention-2 backward
  (plan_gpt G3+), not this GPU_SAFE batch.

### 2026-05-31 revisit: CE occupancy fix (GPU_SAFE)

The first CE upgrade removed the worst single-thread cliff but left two GPT-scale
occupancy problems:
- `cross_entropy` forward was still **one threadgroup total** (≤256 threads)
  looping over all positions and classes.
- `cross_entropy_bwd` was **one thread per token**, each serially scanning
  `V=8000` three times (max, sum, write). The measured 32 ms was mostly serial
  softmax work, not the `dlogits` write bandwidth.

Implemented safer kernel parallelism before attempting cut-CE fusion:
- `gd_cross_entropy_bwd`: one threadgroup per position; threads cooperate over
  classes for row max/sum reductions, then each thread writes a strided class
  slice. Same math, one op/dispatch, CPU_REF remains oracle.
- `gd_cross_entropy`: pass 1 one threadgroup per position writes per-position
  losses to per-node scratch; pass 2 (`gd_cross_entropy_reduce`) reduces scratch
  to the scalar mean. Same op semantics, still GPU_SAFE at the IR level.

Numbers (M1 Pro, GPT bench, B=4,T=256,6L, fresh metallib + rebuilt bench):
```text
before CE revisit (post F1): cross_entropy 17.5 ms, cross_entropy_bwd ~32 ms,
                             step ~408 ms, ~2509 tok/s
after bwd only:              cross_entropy 17.5 ms, cross_entropy_bwd 1.56 ms,
                             step 376.6 ms, 2719 tok/s
after fwd+bwd (initial):     cross_entropy 0.54 ms, cross_entropy_bwd 1.39 ms,
                             step 358.7 ms, 2855 tok/s
after fwd+bwd (clean release, BENCH_CFLAGS=-O3 -ffast-math):
                             cross_entropy 0.56 ms, cross_entropy_bwd 1.44 ms,
                             step 361.3 ms, 2834 tok/s
```
User workload (B=8,T=512,6L): best step **1905 ms -> 1731 ms**, tokens/s
**2150 -> 2366** (~10.0% faster; earlier no-parallel run saw 1693 ms / 2420 tok/s).
Clean release trace after CE: `sdpa_bwd` 642.6 ms, `matmul` 245.7, `sdpa`
224.1, `linear` 104.6, `embedding_bwd` 48.4, `rms_norm_wbwd` 37.8,
`reduce_to` 20.4, `adamw_step` 16.5, CE fwd+bwd 6.6.

Validation: `make check` green, GPT train parity green, `test_metal_gpt` ASan
clean. Learning: the CE tail was under-parallelization/occupancy, not primarily
materialization bandwidth; this makes cut-cross-entropy fusion lower priority
until profiling shows CE is still hot after the safe kernel fix.

### 2026-05-31 tail triage after CE

Clean release profile (BENCH_CFLAGS=`-O3 -ffast-math`) made the post-CE shape
clear:
- T=256: `sdpa_bwd` 129.6 ms, `matmul` 83.1, `sdpa` 53.2, `linear` 37.4,
  tail (`add`/`copy`/`adamw_step`/`reduce_to`) ~65.8, CE fwd+bwd 2.0.
- T=512,B=8: `sdpa_bwd` 642.6 ms, `matmul` 245.7, `sdpa` 224.1, `linear`
  104.6, tail ~82.1, CE fwd+bwd 6.6.

Quick wins checked:
- `reduce_to`: all GPT instances are leading-batch reductions (`[B,...] -> [...]`,
  B=4/8). Added a fast path: `out[i] = sum_b go[b*N+i]`, bypassing generic
  coordinate/divmod logic. Trace: T=256 `16.6 -> 14.6 ms`; T=512 `20.4 -> 15.0
  ms`. End-to-end is small/noisy (`~361 -> ~359 ms` at T=256; `~1731 -> ~1728
  ms` at T=512) but the kernel is simpler and safe.
- `adamw_step`: tried hoisting bias-correction `pow(beta,t)` to once per
  threadgroup. No measurable win (barrier/threadgroup state cancelled it, or the
  compiler/runtime already made pow cheap enough). Reverted; do not ship.
- `copy`: 48 reshape copies already alias-skip; the remaining 56 are external
  gradient-slot writes before AdamW. Avoiding them changes public gradient
  side-effect semantics, so not a quick GPU_SAFE win.

Conclusion: tail has no large low-risk win left at this size. Active next target
should be attention backward (`sdpa_bwd`), especially for the user workload where
`sdpa`+`sdpa_bwd` is ~50% of step. Immediate follow-up retuned attention split-K
(`GD_METAL_SDPA_SPLIT_MIN 256 -> 128`): T=256 `sdpa_bwd 129.6 -> 113.5 ms`, step
`361.3 -> 328.4 ms`; T=512 user workload `sdpa_bwd 642.6 -> 610.5 ms`, step
`1731 -> 1697 ms`. Then parallelized split-K `dq/dkv` reductions over channels:
T=256 `sdpa_bwd 113.5 -> 111.9 ms`, step `328.4 -> 318.9 ms`; T=512 `sdpa_bwd
610.5 -> 584.9 ms`, step `1697 -> 1620 ms`. Then fused the split stats+dq scan:
T=256 `sdpa_bwd 111.9 -> 99.7 ms`, step `318.9 -> 309.2 ms`; T=512 `sdpa_bwd
584.9 -> 519.8 ms`, step `1620 -> 1596 ms` (tracked in
`plan_block_sparse_sdpa_metal.md` §6.4–§6.6).

Validation:
- `make check` (all CPU<->Metal parity + GPT train parity green at 1e-4)
- `GD_PROFILE=trace ... make gpt-bench` per-op attribution before/after
- 12.39M GPT train step: ~11.4 s -> ~0.372 s/iter on Metal

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
Completed.

Standalone GEMM microbenchmark (512x512x512, F32, 50 iters, release):

```text
naive  gd_matmul        metal  ~1.589 ms/iter  ~169 GFLOP/s
tiled  gd_matmul_tiled  metal  ~0.929 ms/iter  ~289 GFLOP/s   (~1.7x)
cpu_ref                  cpu    ~466.6 ms/iter  ~0.6 GFLOP/s
```

Tiny GPT (`GD_DEVICE=metal make bench-gpt`) wall time: `~12.58s` (P3/P6) ->
`~12.42s` with tiled kernels. The toy config (M,N,K <= 64, T=16) is dispatch and
command-buffer-wait bound, so tiled GEMM barely moves it; the win shows on real
GEMM sizes as above.

Choices:
- Added `gd_matmul_tiled` and `gd_linear_tiled` using `GD_METAL_GEMM_TILE` (16)
  threadgroup-memory tiles, streaming K in tiles so each operand element is read
  from device memory once per tile instead of once per output element.
- `gd_matmul_tiled` reproduces the reference kernel's batch-broadcast and
  transpose addressing (grid z = batch index), so it is a drop-in for every shape
  the naive kernel accepted; the naive kernels are retained in the metallib but
  no longer routed.
- `gd_linear_tiled` folds bias add into the store and selects the weight layout
  by `trans_w`.
- Added `dispatch_gemm_tiles` using fixed-size threadgroups (required for
  threadgroup-memory tiles) with grid rounded up; removed the now-unused
  `dispatch_2d` helper.
- Tile size 16 chosen as a safe default (256 threads/group, well under Apple
  `maxTotalThreadsPerThreadgroup`); SIMD-group matrix intrinsics deferred.

Learnings:
- Threadgroup tiling gives ~1.7x on a mid GEMM at 16x16 tiles with no SIMD-group
  matmul; larger tiles / `simdgroup_matrix` would push further but add
  complexity and per-GPU tuning.
- Accumulation order changes (tile-summed) vs CPU double accumulation; parity
  still holds at 1e-4 for all existing tests including GPT train parity.
- For the tiny-GPT regression workload, kernel throughput is not the bottleneck;
  command-buffer wait dominates. A larger benchmark config (P12 knobs) is needed
  to show end-to-end model speedups from kernel work.

Validation:
- `./build/tests/test_metal` (matmul/linear parity incl. trans_b, batched)
- `./build/tests/test_metal_gpt`, `test_metal_gpt_train`, `test_metal_mlp`
- `make check`
- GEMM microbenchmark above (naive vs tiled, parity-equivalent results).

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

## P13 — GPU_SAFE kernel upgrades batch 3: per-row threadgroup reductions

- [x] Phase complete (RMSNorm family; standalone softmax/sum/mean not exercised by GPT)

### Intent
Finish the GPU_SAFE kernel parallelization started in P9 for the remaining
reduction-style kernels that still use one-thread-per-row (or fewer) and leave
most GPU lanes idle. These were not hot on the current 12.39M / T=64 workload
(so P9 shipped without them), but they become the next ceiling as the hot
offenders are gone and as rows/seq grow.

Still GPU_SAFE: one IR node -> one kernel -> one dispatch, same math; only the
intra-kernel thread strategy changes (threadgroup memory + barrier reductions).
No fusion, no layout planning. (Confirmed: per-row threadgroup reduction is the
same category as the P9 `cross_entropy` reduction and the P10 tiled GEMM.)

### Targets
Current per-op trace cost on the 1-layer 4.2M / T=64 step (post-P9) for context;
re-profile on a larger row/seq config before optimizing:
- `rms_norm_wbwd` (~9 ms, largest remaining reduction): reduces over rows per
  channel and recomputes the per-row RMS inside the loop. Parallelize the
  per-row RMS and the cross-row accumulation.
- `rms_norm` / `rms_norm_bwd`: threadgroup reduction over the last dim instead of
  one thread per row, for large `d_model`.
- `softmax`: threadgroup row reduction for max/sum (one thread per
  `(outer,inner)` today).
- `sum` / `mean` (`gd_reduce`): threadgroup reduction over the reduced axis for
  large reduced dims.

### Approach
- One threadgroup per row (or per reduced slice); threads cooperatively load and
  reduce via threadgroup memory + `threadgroup_barrier`, matching the
  `gd_cross_entropy` pattern landed in P9.
- Cap threadgroup size by `maxTotalThreadsPerThreadgroup` and size the shared
  array to that cap.
- Keep the existing one-thread-per-row kernels as the correctness reference for
  parity diffing during development.

### Acceptance
- Per-op CPU<->Metal parity holds at 1e-4 (document any reduction-order tolerance).
- GPT train parity stays green.
- `GD_PROFILE=trace` shows reduced `gpu_ms` for each upgraded op on a config with
  large rows / sequence (e.g. bigger `d_model`, longer `T`).

### Out of scope
- `sdpa_bwd` (O(T^2) reference recompute, ~55 ms in trace): tracked as the
  deferred FlashAttention-2 backward in `plan_gpt.md` G3+, not a per-row
  reduction; it is a different (fused/tiled) rewrite, not GPU_SAFE batch work.

### Completion notes
Grounded on a fresh 6-layer trace baseline (12.39M params, T=64) taken before this
phase. That confirmed `sdpa_bwd` dominates (298.9 ms of a 506 ms trace step) and
is G3/out-of-scope; the largest GPU_SAFE op was `rms_norm_wbwd` (29.5 ms). GPT
never emits standalone `softmax`/`sum`/`mean` (softmax is internal to SDPA), so
those were left as-is.

Fixed the RMSNorm family (one IR node -> one kernel -> one dispatch preserved):
- **`rms_norm`**: one threadgroup per row; threads cooperatively reduce sum(x^2)
  via threadgroup memory, then write the normalized row. `5.4 -> 3.4 ms`.
- **`rms_norm_bwd`**: one threadgroup per row; single-pass cooperative reduction
  of both sum(x^2) and `A = sum_c go*weight*x`, then write dx. `5.5 -> 3.0 ms`.
- **`rms_norm_wbwd`**: root cause was recomputing each row's RMS *per channel*
  (`last`x redundant). New kernel uses one threadgroup per channel tile and
  computes the per-row rms_inv cooperatively **once per row tile**, cached in
  threadgroup memory, then each channel thread sums over rows. `29.5 -> 3.6 ms`
  (~8x).

Whole-step effect (6-layer, T=64): trace step `506 -> 480 ms`; real non-trace
train step `~372 -> ~345 ms`. The remaining time is almost entirely `sdpa_bwd`
(~302 ms in trace, ~87%); every non-SDPA op is now < 4 ms.

Choices:
- Shared threadgroup size `GD_RMS_TG = 256` (kernels) / `GD_METAL_RMS_TG = 256`
  (host), power of two for an exact tree reduction.
- `rms_norm`/`rms_norm_bwd` dispatch one threadgroup per row; `rms_norm_wbwd`
  dispatches `ceil(last / 256)` threadgroups (one per channel tile) and tiles the
  row dimension by 256 so threadgroup memory is bounded regardless of rows.
- 1D `threadgroup_position_in_grid` declared as scalar `uint` (MSL rejects mixing
  a `uint3` grid attribute with scalar `threads_per_threadgroup`).

Learnings:
- The dominant `rms_norm_wbwd` cost was redundant work, not under-parallelism:
  caching rms_inv per row tile removed an O(last) redundancy factor and gave the
  8x even though the kernel still only spans a couple of threadgroups.
- With P9+P13 done, the GPU_SAFE backend has no remaining single-thread or
  redundant-reduction cliffs on the GPT workload; the next real lever is SDPA
  (G3), which is explicitly outside GPU_SAFE.

Validation:
- `make check` (CPU<->Metal parity for rms_norm fwd/bwd/wbwd + GPT train parity,
  all at 1e-4; overfit sanity green)
- `GD_PROFILE=trace ... GD_BENCH_LAYERS=6` per-op attribution before/after

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
| P10 | release `-O2` | `GD_DEVICE=metal make bench-gpt` | ~12.42s | ~20.7 | tiled GEMM (tiny GPT still wait-bound) |
| P10 | release `-O2` | metal 512³ GEMM microbench | ~0.93ms/iter | — | ~289 GFLOP/s vs ~169 naive |
| P9 | release `-O2` | metal 12.39M GPT train, T=64 (gpt-bench) | ~0.372s/iter | — | ~30x vs pre-P9 ~11.4s; CPU ref ~2.66s |
| P13 | release `-O2` | metal 12.39M GPT train, T=64 (gpt-bench) | ~0.345s/iter | — | RMSNorm family parallelized; rest now bounded by sdpa_bwd (G3) |
| P4 | release `-O2` | CPU-backed params, Metal graph, 100 steps, 1 read | — | ~1.12 ms/run | wait/writeback per-run O(steps)->1; CPU-backed path now async |
| G3 | release `-O2` | metal 12.39M GPT train, T=64 (gpt-bench) | ~0.101s/iter | — | FA-2 sdpa_bwd: 298.9->56.7ms; step 345->101ms; 14->48 GFLOP/s (tracked in plan_gpt.md) |
| baseline | release `-O2` | metal 12.39M GPT train, **B=4 T=256** (gpt-bench) | ~1.370s/iter | — | canonical workload; 60 GFLOP/s, attention-bound (sdpa+bwd 74%) |
| G3-tiled | release `-O2` | metal 12.39M GPT train, **B=4 T=256** (gpt-bench) | ~0.500s/iter | — | FlashAttention tiling fwd+bwd: sdpa 264.9->48.4, sdpa_bwd 847.8->127.8ms; 60->164 GFLOP/s, 2055 tok/s; now GEMM-bound |

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
