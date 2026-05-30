# gradients.c — Metal Prerequisites Plan

Status: draft v0.1

Goal: refactor the CPU-only foundation into a **backend-dispatched architecture** so the Metal backend can be added without rewriting core, storage, transfers, or execution. No Metal code in this plan — only the seams Metal will plug into, validated entirely on CPU_REF.

Reference: [`docs/design_spec.md`](design_spec.md) §7 (device/backend), §8–9 (storage/memory), §13 (graph execution), §17 (private backend API), §20 (debugging/parity). Foundation: [`docs/plan_mlp.md`](plan_mlp.md).

Guiding invariant: **after this plan, CPU_REF is just "the backend that happens to be selected".** There must be exactly one execution path. Metal becomes a second registration, not a second path.

Non-goal: kernel fusion, layout planning, buffer reuse, arenas (tracked separately). Those are not required to bring up a correct Metal backend.

---

## Definition of done

1. `make check` passes, including a new ASan/UBSan build and a CPU↔CPU parity test.
2. No core/storage/op/autograd source references a backend by name; all device work goes through `_gd_backend`.
3. `gd_graph_run` executes via the target backend's hooks; `cpu_ref` is registered, not hardwired.
4. `gd_storage` holds an opaque backend allocation; host pointers are only exposed for host-visible memory and only by the CPU-capable allocator.
5. `gd_synchronize` is real; `gd_tensor_copy_to_cpu` is guaranteed to observe completed work.
6. Fallback policy is enforced at dispatch: missing kernel → `GD_ERR_UNSUPPORTED` under `GD_FALLBACK_NONE`, or CPU_REF execution under `GD_FALLBACK_CPU_REF`.
7. Adding a backend requires implementing one vtable and calling one registration function — provably, by a stub "null backend" test that registers and is selected.

---

## Phase status checklist

- [x] P0 — Backend interface + registry (`_gd_backend`, capabilities, registration)
- [x] P1 — Storage refactor: allocate through backend, opaque handle, memory kinds
- [x] P2 — Host transfer routing through backend (upload/download, blocking)
- [x] P3 — Executor dispatch: compile/run own the backend, CPU_REF becomes a backend
- [x] P4 — Synchronization semantics (`gd_synchronize`, copy/materialize ordering)
- [x] P5 — Fallback policy enforcement at dispatch
- [x] P6 — Device-aware `memory_kind` selection on tensor/storage creation
- [x] P7 — Build system: Objective-C / `.metal` toolchain wiring (compiles, no kernels yet)
- [x] P8 — Backend parity harness (`gd_graph_compare`) validated CPU↔CPU
- [ ] P9 — Memory-safety gate: ASan/UBSan/LSan test target in `make check`
- [ ] P10 — Null-backend integration test (proves the seam)

P0→P3 are strictly ordered. P4–P6 depend on P1–P3. P7–P10 can proceed once P3 lands.

---

## P0 — Backend interface + registry

- [x] Phase complete

### Intent
Define the single seam. Every device-touching operation routes through a vtable resolved from `gd_device.type`.

### Files
```
src/backends/backend.h          (new) _gd_backend, _gd_backend_vtable, capability flags
src/core/context.c              registry: array indexed by gd_device_type
src/core/internal.h             _gd_context_backend(ctx, device, **out)
src/backends/cpu_ref/cpu_ref.c  implements + registers the vtable (P3 wires execution)
```

### Private API (sketch, names final)
```c
typedef struct _gd_backend _gd_backend;
typedef struct _gd_executable _gd_executable;   /* opaque compiled artifact */

typedef struct _gd_backend_caps {
    bool host_visible;        /* can expose CPU pointers (CPU_REF, unified) */
    bool supports_cpu_ref;    /* is itself the reference backend */
} _gd_backend_caps;

typedef struct _gd_backend_vtable {
    gd_device_type type;
    const char *name;

    gd_status (*init)(_gd_backend *self, gd_context *ctx, int device_index);
    void      (*shutdown)(_gd_backend *self);

    /* Storage */
    gd_status (*storage_alloc)(_gd_backend *, const gd_storage_desc *, void **handle_out);
    void      (*storage_free)(_gd_backend *, void *handle);
    gd_status (*storage_host_ptr)(_gd_backend *, void *handle, void **ptr_out); /* host-visible only */

    /* Transfers (blocking in v1) */
    gd_status (*upload)(_gd_backend *, void *dst_handle, size_t off, const void *src, size_t n);
    gd_status (*download)(_gd_backend *, void *src_handle, size_t off, void *dst, size_t n);

    /* Execution */
    gd_status (*compile)(_gd_backend *, gd_graph *, _gd_executable **out);
    gd_status (*execute)(_gd_backend *, _gd_executable *);
    void      (*executable_free)(_gd_backend *, _gd_executable *);

    /* Per-op capability query for fallback decisions (P5) */
    bool      (*supports_node)(_gd_backend *, const _gd_node *);

    gd_status (*synchronize)(_gd_backend *);
} _gd_backend_vtable;

struct _gd_backend {
    const _gd_backend_vtable *vt;
    _gd_backend_caps caps;
    gd_context *ctx;
    int device_index;
    void *impl;   /* backend-private state */
};
```

### Context changes
```c
/* registry: at most one backend per gd_device_type for v1 */
_gd_backend *_gd_context_backend(const gd_context *ctx, gd_device device);
gd_status    _gd_context_register_backend(gd_context *ctx, const _gd_backend_vtable *vt);
```
- `gd_context_create` registers CPU_REF unconditionally.
- `_gd_device_validate_available` becomes "is a backend registered for this device.type/index".

### Acceptance
- Builds; `gd_context_default_device` still CPU; unsupported device → `GD_ERR_UNSUPPORTED` via registry, not a hardcoded check.
- No behavior change yet (CPU still executes via existing path until P3).

---

## P1 — Storage refactor

- [x] Phase complete

### Intent
`gd_storage` stops owning a raw `void*` from `posix_memalign`. It owns an **opaque backend handle** plus `gd_storage_desc`. Host pointer access is mediated and only valid for host-visible memory.

### Changes
- `struct gd_storage { gd_refcount; gd_storage_desc desc; _gd_backend *backend; void *handle; }`.
- `gd_storage_create`: resolve backend from `desc.device`, call `vt->storage_alloc`. CPU_REF's `storage_alloc` does the aligned host allocation (moves `posix_memalign` into cpu_ref).
- `gd_storage_release`: `vt->storage_free`.
- `_gd_storage_data_mut/_data`: call `vt->storage_host_ptr`; return NULL / error for non-host-visible. Internal callers that assume host pointers (CPU kernels) only run under the CPU backend, so this stays valid there.
- `gd_storage_data_cpu`: success only if backend caps `host_visible` and memory kind is host-accessible.
- Validation of `device/memory_kind` legality moves into the backend (`storage_alloc` rejects unsupported combos), per spec §9.2.

### Acceptance
- All existing storage/tensor tests pass unchanged (CPU host path identical bytes).
- `posix_memalign` appears only inside `cpu_ref`.
- `grep -rn posix_memalign src/core` → empty.

---

## P2 — Host transfer routing

- [x] Phase complete

### Intent
No core code `memcpy`s into `storage->data`. All host↔device movement goes through the backend.

### Changes
- `gd_storage_copy_from_cpu/to_cpu` → `vt->upload` / `vt->download` (bounds-checked in core, movement in backend). CPU_REF upload/download are `memcpy`.
- `gd_tensor_copy_from_cpu/to_cpu` keep their contiguity/offset checks, then call storage transfer (already layered this way — just re-point at backend).
- `gd_tensor_materialize` / `_gd_tensor_materialize_from_graph`: allocate via backend, download from the value buffer's storage, never assume host pointer.
- `gd_tensor_contiguous` eager host gather: requires `host_visible`; otherwise `GD_ERR_UNSUPPORTED` (Metal path uses a graph copy later — out of scope here).

### Acceptance
- `grep -rn "memcpy" src/core/storage.c src/core/tensor.c` shows only bounds-safe paths that are host-visible-guarded or moved to cpu_ref.
- Transfer round-trip tests pass; bounds checks unchanged.

---

## P3 — Executor dispatch (the keystone)

- [x] Phase complete

### Intent
`gd_graph_compile`/`gd_graph_run` delegate to the target backend. CPU_REF's interpreter becomes `cpu_ref`'s `compile`/`execute`.

### Changes
- `gd_graph_compile(g, target)`:
  - resolve backend for `target`
  - validate finalized; call `backend->vt->compile(backend, g, &g->exec)`
  - store `backend` + `_gd_executable*` on the graph; set state compiled.
- CPU_REF `compile`: current `allocate_buffers` logic → produces a CPU `_gd_executable` holding the per-value buffer table.
- CPU_REF `execute`: current node loop (`_gd_cpu_run_node`).
- `gd_graph_run`: `backend->vt->execute(backend, g->exec)`.
- `gd_graph_run_until`: CPU-ref-specific debug entry; keep but gate to backends that expose partial execution (CPU only for now) — document, return `GD_ERR_UNSUPPORTED` otherwise.
- `_gd_graph_value_data`: becomes CPU-ref executable accessor (used by parity/debug + virtual-tensor reads on host-visible backends). For non-host backends, virtual reads require download (P2).
- `_gd_graph_free_buffers` → `executable_free`.

### Acceptance
- Whole suite passes with execution flowing through `cpu_ref` vtable.
- `grep -rn "_gd_cpu_run_node" src/graph` → empty (graph no longer calls the kernel path directly).
- Build with cpu_ref registration removed (temporarily, in a scratch test) → compile returns `GD_ERR_UNSUPPORTED`, proving decoupling.

---

## P4 — Synchronization semantics

- [x] Phase complete

### Intent
Correct ordering for async backends; defined now so Metal inherits it.

### Changes
- `gd_synchronize(ctx, device)` → `backend->vt->synchronize`. CPU_REF: no-op (already ordered).
- Contract documented + enforced: `download` and `gd_tensor_copy_to_cpu`/`materialize` are **blocking** — they synchronize the backend before reading (v1).
- `execute` may be async internally; results are only guaranteed after `synchronize` or a blocking download.

### Acceptance
- New test: run graph, `gd_synchronize`, read — and run graph, read (blocking download) — both observe results. (CPU trivially passes; locks the contract.)

---

## P5 — Fallback policy enforcement

- [x] Phase complete

### Intent
Wire the existing `gd_fallback_policy` into dispatch. Spec pillar #7: fail loud unless CPU_REF fallback is explicitly enabled.

### Changes
- During `compile`, for each node the target backend reports `supports_node`. If unsupported:
  - `GD_FALLBACK_NONE` → compile fails `GD_ERR_UNSUPPORTED` naming the node id/op.
  - `GD_FALLBACK_CPU_REF` → that node is assigned to the CPU_REF backend (mixed execution). Requires host-visible buffers for the boundary tensors; if the target is not host-visible, insert backend download/upload at the boundary (define now; CPU↔CPU is a no-op so testable).
- v1 simplification allowed: fallback may execute the **entire** graph on CPU_REF if any node is unsupported, as long as the choice is explicit and logged. (Per-node mixing can come with the planner.) **Decision recorded here:** v1 = whole-graph CPU_REF fallback; per-node mixing deferred.

### Acceptance
- Test with a synthetic "unsupported on backend X" node:
  - `NONE` → `GD_ERR_UNSUPPORTED` with node id in `gd_last_error`.
  - `CPU_REF` → runs, correct result.

---

## P6 — Device-aware `memory_kind`

- [x] Phase complete

### Intent
Stop hardcoding `GD_MEM_HOST` in `gd_tensor_empty`.

### Changes
- `gd_tensor_empty` selects memory kind from backend capability: host-visible backend → `GD_MEM_HOST`; device backend → `GD_MEM_DEVICE` (or `GD_MEM_UNIFIED` if the backend prefers it). Add `vt`-level "preferred kind for a fresh tensor".
- Graph intermediate buffers (CPU_REF executable) likewise ask the backend.

### Acceptance
- CPU path unchanged (still host). Mechanism covered by null-backend test (P10) requesting `GD_MEM_DEVICE`.

---

## P7 — Build system for Metal toolchain

- [x] Phase complete

Note: the `.metal` shader compiler ships with full Xcode, not the Command Line
Tools. The Makefile compiles shaders only when `xcrun --find metal` succeeds and
otherwise skips the `.metallib` while still building the Objective-C path. On
this machine (CLT only) the `.m` path compiles + links the Metal frameworks; the
shader compile is gracefully skipped.

### Intent
Makefile can compile Objective-C and Metal shaders on macOS; non-macOS still builds CPU-only.

### Changes
- Detect platform (`uname`). On macOS:
  - compile `*.m` with `-fobjc-arc`
  - link `-framework Metal -framework Foundation -framework QuartzCore`
  - rule: `xcrun -sdk macosx metal -c %.metal -o %.air` then `metallib` → `.metallib`, staged into `build/`
- `src/backends/metal/` excluded from build unless `GD_ENABLE_METAL=1` (default on macOS, off elsewhere). No kernels yet — an empty registration stub that returns `GD_ERR_UNSUPPORTED` is acceptable to prove the build path.

### Acceptance
- `make check` on macOS compiles a placeholder `.m` + `.metal`; on Linux, CPU-only build still green.

---

## P8 — Backend parity harness

- [x] Phase complete

Note: `gd_graph_compare` compiles + runs the finalized graph on `reference` then
`target`, synchronizing each pass, and downloads every produced value into host
buffers for comparison. F32 values use `atol`/`rtol`; other dtypes are compared
byte-exact. The first mismatch is reported through `gd_last_error` as
`value id / producing op+node / coordinate / ref+target value / max abs+rel` and
returns `GD_ERR_BACKEND`. External leaf values are skipped unless
`compare_externals` is set (they share input bytes across runs). v1 assumes a
side-effect-free forward graph (executed twice); training graphs that mutate
external tensors in place are out of scope. The graph is left compiled on
`target`. Test (`tests/test_compare.c`) covers CPU↔CPU equality (incl. zero
tolerance) and a perturbing test backend that injects a known error, verified to
be reported with the producing node id.

### Intent
The tool that makes GPU development possible: node-by-node CPU↔target comparison.

### Files
```
include/gradients/graph.h      gd_graph_compare, gd_compare_options (already sketched in spec §20)
src/graph/compare.c            (new)
```
### Behavior
- Run the same graph on `reference` and `target` backends; for each materializable value compare with `atol/rtol`; report first mismatch: node id, op, value id, max abs/rel error, coordinate.
- v1: validated CPU↔CPU (must report "no mismatch"); structurally ready for CPU↔Metal.

### Acceptance
- Test: identical graphs compare equal; an injected perturbation is detected and reported with node id.

---

## P9 — Memory-safety gate

- [ ] Phase complete

### Changes
- Makefile target `make asan`: `-fsanitize=address,undefined -fno-omit-frame-pointer`, builds lib + tests, runs them.
- `make check` invokes `make asan` (or a `check-asan`) so leaks/UB fail CI.

### Acceptance
- All current tests pass under ASan/UBSan/LSan with zero reports.

---

## P10 — Null-backend integration test

- [ ] Phase complete

### Intent
Prove the seam without Metal: a test-only backend registered for an unused device type that allocates host memory but reports `host_visible=false` and supports no nodes.

### Acceptance
- Registering it and selecting it: storage alloc works, `storage_host_ptr` refused, compile under `NONE` → unsupported, under `CPU_REF` → whole-graph CPU fallback runs correctly, `synchronize` called.
- This single test exercises P0–P6 end to end and is the gate that says "Metal can start."

---

## Out of scope (explicitly deferred)

- Kernel fusion, pattern matching, layout selection, quant specialization.
- Buffer reuse / lifetime planner (replace naive per-value allocation).
- Graph/backward bump-pointer **arena** (recommended but not blocking).
- Per-node mixed CPU/GPU fallback (v1 = whole-graph fallback).
- `gd_graph_compile_ex` pass toggles.
- Streams/events/multi-device/collectives.

## Risks

- **Hidden host-pointer assumptions.** Mitigation: P1/P2 make `_gd_storage_data*` host-visible-guarded; grep gates in acceptance criteria.
- **Two execution paths drift.** Mitigation: P3 removes `_gd_cpu_run_node` from `graph.c` entirely (grep gate).
- **Async correctness.** Mitigation: P4 defines blocking download/synchronize before any async backend exists; P8 parity catches ordering bugs.
- **Fallback ambiguity.** Mitigation: P5 records the v1 whole-graph decision explicitly and tests both policies.
