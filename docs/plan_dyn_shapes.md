# gradients.c — Dynamic Shapes Production Plan

Status: planned (2026-06-02)

Goal: make variable-size batches correct and fast without rebuilding graphs per
batch. End state is **static op topology + runtime dimensions + bounded shape
specialization cache + persistent runner arenas**.

This is not padding-for-correctness. Bucketing stays optional perf policy.

---

## 0. Current diagnosis

Current code path is static-shape execution with runner rebinding:

- `gd_graph_add_input` stores concrete `gd_tensor_desc` on `_gd_value.desc`.
- `gd_graph_runner_bind` requires exact dtype/device/layout/**shape** match.
- Op meta runs at capture time and writes concrete output descs.
- `gd_graph_compile` builds one backend executable for those concrete descs.
- Metal compile allocates value buffers and scratch arena sized from concrete descs.
- MPS GEMM plans are shape-specialized at compile.
- GPT-VLM currently builds, compiles, runs, synchronizes, destroys graph each step.

Relevant files:

- `include/gradients/graph.h`
- `src/graph/runner.c`
- `src/graph/graph_internal.h`
- `src/graph/execute.c`
- `src/backends/metal/compile.m`
- `src/backends/metal/runtime.m`
- `src/backends/metal/mps_plan.m`
- `examples/gpt-vlm/gpt_vlm.c`

Important VLM-specific issue: topology is not fully shape-stable today.
`build_train_graph()` bakes per-sample `text_lens[b]` into `slice_start`,
`slice_len`, and concat attrs. Dynamic dims alone are insufficient; VLM needs a
runtime pack op or runtime-valued slice/concat semantics.

### Status

- [x] Confirmed current runner rejects alternate shapes.
- [x] Confirmed graph values carry concrete descriptors only.
- [x] Confirmed Metal executable owns concrete value storage/scratch.
- [x] Confirmed VLM training recompiles every step.
- [x] Confirmed MPS GEMM planning is concrete-shape.
- [ ] Add instrumentation to quantify compiles, shape-plan hits, allocations,
      memory high-water, and cache evictions in VLM run.

---

## 1. Product requirements

### Must have

- [ ] One captured graph per stable op topology.
- [ ] Runtime tensor dimensions may vary per runner execution.
- [ ] Runtime shape inference produces concrete descs before backend encode.
- [ ] Shape specialization cache is bounded and deterministic.
- [ ] Memory grows to high-watermark then plateaus.
- [ ] No per-step graph build/compile for common variable-size training loops.
- [ ] No retained backend objects tied to transient per-step buffers.
- [ ] CPU and Metal parity for alternating shapes.
- [ ] AMP `found_inf` reflects real numeric overflow, not memory/resource churn.

### Non-goals for first production slice

- [ ] Dynamic op topology/control flow.
- [ ] Dynamic rank.
- [ ] Cross-device dynamic shapes.
- [ ] Arbitrary symbolic algebra. Start with named runtime dims and simple
      validated relationships.
- [ ] Paged/ragged tensor storage. Packed tensors stay ordinary contiguous
      storage with runtime sizes.

### Compatibility

- [ ] Existing static graph API continues working.
- [ ] Existing `gd_graph_add_input` keeps exact-shape behavior.
- [ ] Dynamic shape API is additive.
- [ ] Static graphs may internally use same shape-plan path with one cached plan.

---

## 2. Architecture target

```text
capture once
  graph topology
  static attrs
  symbolic/runtime dim specs
  shape constraints

compile once
  topology executable
  backend-invariant resources (PSOs, op registry, static validation)
  runner shape-plan cache
  runner arenas

run many times
  bind tensors + runtime dim values
  validate constraints
  infer concrete descs
  lookup/build shape_plan(signature)
  ensure arenas >= plan high-water
  encode with current buffers/descs
```

### Core objects

- [ ] Add internal shape descriptor separate from `gd_tensor_desc`:
  - rank, dtype, device, layout static for v1
  - dims as either constant or runtime symbol
  - optional min/max bounds
  - optional equality constraints
- [ ] Add graph-level runtime shape symbols:
  - examples: `B`, `N_tokens`, `N_text`, `max_seq`
  - symbols can bind from tensor dimensions or explicit runner dim binding
- [ ] Add concrete shape environment per run.
- [ ] Add shape signature hash over all runtime dims and backend-relevant static
      toggles.
- [ ] Add shape plan cache with fixed max entries and LRU eviction.
- [ ] Add runner-owned memory arenas with high-water growth and reuse.

### Backend split

- [ ] Split backend executable into topology plan + shape plan.
- [ ] Topology plan owns backend-invariant resources:
  - Metal PSOs / libraries
  - static op attrs
  - op kind sequence
  - static validation result
- [ ] Shape plan owns concrete resources:
  - concrete value descs
  - memory offsets/lifetimes
  - scratch sizes
  - dispatch params
  - MPS descriptors/kernels for exact concrete GEMM shapes
- [ ] Runner owns shape-plan cache and arenas.

---

## 3. Public API sketch

Names are provisional; lock before implementation.

```c
typedef struct gd_graph_shape_var gd_graph_shape_var;

typedef struct gd_dim_spec {
    int64_t value;              /* >=0 constant, <0 means dynamic */
    gd_graph_shape_var *symbol; /* nullable */
    int64_t min_value;          /* optional, 0 means unset */
    int64_t max_value;          /* optional, 0 means unset */
} gd_dim_spec;

typedef struct gd_dynamic_tensor_desc {
    gd_dtype dtype;
    gd_device device;
    gd_layout layout;
    int ndim;
    gd_dim_spec dims[GD_MAX_DIMS];
} gd_dynamic_tensor_desc;

gd_status gd_graph_add_shape_var(gd_graph *graph,
                                 const char *name,
                                 int64_t min_value,
                                 int64_t max_value,
                                 gd_graph_shape_var **out);

gd_status gd_graph_add_dynamic_input(gd_context *ctx,
                                     gd_graph *graph,
                                     const char *name,
                                     const gd_dynamic_tensor_desc *desc,
                                     gd_tensor **tensor_out,
                                     gd_graph_input **input_out);

gd_status gd_graph_runner_bind_dim(gd_graph_runner *runner,
                                   gd_graph_shape_var *var,
                                   int64_t value);
```

Rules:

- [ ] Static API remains exact-shape.
- [ ] Dynamic input binding validates dtype/device/layout/rank, then binds dims.
- [ ] If multiple tensors bind same symbol, values must match.
- [ ] Explicit `gd_graph_runner_bind_dim` required for runtime attrs not present
      as tensor dims, e.g. `max_seq`.
- [ ] Dynamic symbols cannot silently infer from tensor data on device.
- [ ] Error messages include symbol name, expected constraint, and actual value.

---

## 4. Runtime shape inference

Use existing op meta functions where possible by feeding concrete descs after
runner binding.

### Tasks

- [ ] Store original op attrs plus dynamic attr bindings.
- [ ] Build concrete input desc table from bindings and externals.
- [ ] Run op meta in graph order into per-plan concrete desc table.
- [ ] Validate output descs satisfy static/dynamic graph constraints.
- [ ] Keep `gd_tensor_desc_nbytes` only on concrete descs.
- [ ] Add shape-inference cache tests for all public ops.
- [ ] Add dynamic attr support for ops that need runtime bounds:
  - `sdpa_varlen.max_seqlen`
  - VLM pack output rows
  - future decode/cache lengths

### Constraints

- [ ] Rank fixed.
- [ ] Dtype fixed unless op explicitly casts.
- [ ] Layout fixed to contiguous for first slice.
- [ ] Device fixed.
- [ ] `storage_offset_bytes == 0` for dynamic graph inputs in first slice.

---

## 5. Memory planner and arenas

Current Metal compile allocates one storage per value. Dynamic runner needs
persistent high-water arenas and lifetime reuse.

### Allocation contract

Hot cache-hit run must not allocate persistent memory.

Allowed allocations:

- graph capture/compile: graph nodes, topology executable, static backend resources
- first time a shape is seen: shape plan metadata, exact-shape MPS descriptors/kernels
- first time high-water increases: grow runner arenas
- optional short-lived encode wrappers inside autoreleasepool, measured and capped by tests

Disallowed after warmup on cache-hit runs:

- `gd_storage_create` for intermediates/scratch
- Metal `newBuffer` for graph values/scratch
- graph node/value allocation
- MPS plan growth
- unbounded autorelease buildup

If a workload produces many unique shapes, shape-plan creation can churn, but LRU
bounds memory. Performance-critical users can bucket for fewer misses; correctness
must not require bucketing.

### Design

- [ ] Compute value lifetimes from graph uses.
- [ ] Do not arena-allocate external values, graph inputs, mutation targets, or
      materialized outputs that must be exposed after run.
- [ ] Arena-allocate intermediates and private temps.
- [ ] Reuse offsets when lifetimes do not overlap.
- [ ] Maintain separate arenas as needed:
  - value/intermediate arena
  - scratch arena
  - optional staging arena for CPU-backed dynamic inputs
- [ ] Grow arenas only when new plan high-water exceeds capacity.
- [ ] Never shrink during hot run; optional explicit trim API later.
- [ ] Track arena generation so stale cached resource wrappers cannot outlive
      backing storage.

### Status

- [ ] Add generic lifetime analysis in graph layer.
- [ ] Add CPU_REF arena-backed value bindings.
- [ ] Add Metal arena-backed value bindings.
- [ ] Preserve direct aliasing for Metal device inputs when safe.
- [ ] Preserve external writeback semantics for optimizer/grads.
- [ ] Add profile counters:
  - arena bytes
  - arena grows
  - storage allocations
  - shape-plan bytes
  - scratch high-water

---

## 6. Shape-plan cache

### High-cardinality shape policy

Datasets can produce thousands of distinct sequence lengths. Correctness must stay
independent of shape reuse, but performance must degrade predictably.

Expected behavior with 10k unique shapes:

- memory remains bounded by LRU cap + runner arena high-water
- cache misses increase planning overhead
- shape-polymorphic ops should not rebuild backend resources per exact length
- exact-shape resources are created only where required, mainly MPS GEMM and some
  specialized kernels
- users may enable bucketing to reduce misses, but unbucketed runs remain correct

Design requirements:

- [ ] Separate **polymorphic plan** from **exact-shape specialization**.
- [ ] Default simple Metal kernels use runtime dims and share one polymorphic
      encode plan across all lengths.
- [ ] MPS GEMM exact `(M,N,K,batch,dtype,trans)` specializations use bounded LRU.
- [ ] Optional specialization threshold: after too many unique shapes, prefer
      generic dynamic kernel over exact specialized path when available.
- [ ] Optional bucket policy can round selected dims for perf, never correctness.
- [ ] Expose counters: unique shapes seen, cache miss rate, specialization
      evictions, generic fallback count.
- [ ] Add stress test with 10k unique lengths: bounded memory, no leaks, no
      correctness failures, clear perf counters.

### Signature contents

- [ ] Runtime shape symbol values.
- [ ] Concrete dynamic input dims.
- [ ] Dtype/layout/device/rank.
- [ ] Op attrs that affect backend plan.
- [ ] Backend feature flags (`GD_METAL_MPS`, SDPA fast toggles, compute policy).
- [ ] Graph version/topology hash.

### Policy

- [ ] Per-runner cache first; context/global cache later only if ownership is
      clear.
- [ ] Fixed max entries from env/config, default small (e.g. 16 or 32).
- [ ] LRU eviction after synchronization if backend resources may be in flight.
- [ ] Eviction frees descriptors/kernels/shape metadata, not runner arenas.
- [ ] Cache hit path does not allocate.
- [ ] Cache miss path can allocate/compile exact shape plan once.

### Tests

- [ ] Alternating 2-shape loop: 2 misses then all hits.
- [ ] 100 unique shapes with cap 8: cache size never exceeds 8.
- [ ] Evicted Metal shape plans wait/retire safely.
- [ ] Cache key changes on dtype/layout/backend flag changes.

---

## 7. Metal/MPS resource hygiene

Current `GDMPSGemmPlan` stores `MPSMatrix *left/right/result`; these are tied to
buffers and must not be retained across runs with dynamic storage.

### Tasks

- [ ] Remove strong `MPSMatrix *left/right/result` from `GDMPSGemmPlan`.
- [ ] Cache only MPS kernel and matrix descriptors in shape plan.
- [ ] Build `MPSMatrix` wrappers from current buffers at encode time.
- [ ] Keep encode-time wrappers inside autoreleasepool.
- [ ] Never store transient wrappers on plan object.
- [ ] Ensure command buffer retains resources until completion.
- [ ] Bound MPS shape-plan cache with same LRU as other shape resources.
- [ ] Add debug assert/instrumentation: no MPSMatrix retained by shape plan.

### Performance follow-up

- [ ] Measure MPSMatrix wrapper allocation overhead.
- [ ] If needed, cache wrappers only when tied to stable runner arena generation
      and invalidate on arena grow. Default safe path remains no retained wrappers.

---

## 8. Dynamic op coverage

Prioritize ops on GPT-VLM hot path.

### Elementwise / simple shape-polymorphic

- [ ] add/mul/scale/cast/relu/silu/gelu/powlu/dropout
- [ ] sum/mean/reduce_to
- [ ] copy/reshape/transpose/slice where attrs are static
- [ ] rms_norm fwd/bwd/wbwd
- [ ] embedding fwd/bwd

### GEMM

- [ ] linear dynamic rows, static feature dims.
- [ ] matmul dynamic batch/rows where backend supports it.
- [ ] MPS shape-specialized GEMM plans cached by exact `(M,N,K,batch,dtype,trans)`.
- [ ] Custom Metal tiled GEMM remains shape-polymorphic via runtime params.

### Attention / loss

- [ ] sdpa_varlen fwd dynamic `N_tokens`, `B`, `max_seq`.
- [ ] sdpa_varlen bwd dynamic `N_tokens`, `B`, `max_seq`, scratch.
- [ ] lm_cross_entropy fwd dynamic rows.
- [ ] lm_cross_entropy bwd dynamic rows and scratch.
- [ ] AMP unscale/clip/step dynamic param grad sizes where needed.

### VLM topology fix

- [ ] Add production op replacing per-sample slice/concat graph construction:

```text
gd_varlen_pack_prefix_suffix(
  prefix_embeds [B*P,D],
  text_embeds   [N_text,D],
  cu_seqlens    [B+1],      // total sequence offsets incl prefix
  prefix_len=P,
) -> inputs [N_tokens,D]
```

- [ ] CPU_REF implementation.
- [ ] Metal fwd kernel.
- [ ] Autograd rule:
  - scatter grad prefix rows to `prefix_embeds`
  - scatter grad text rows to `text_embeds`
- [ ] Metal bwd kernels or equivalent graph lowering.
- [ ] Replace GPT-VLM per-batch `slice/concat` loop with this op.
- [ ] Graph node count becomes stable for same model topology and max batch rank.

---

## 9. Backend API changes

Current vtable assumes `compile(graph) -> executable` and `execute(exe)`.
Keep public API stable, change internals.

### Internal shape

```text
backend.compile(graph) -> topology_executable
backend.get_or_create_shape_plan(topology_executable, shape_env) -> shape_plan
backend.execute_shape_plan(topology_executable, shape_plan, runner) -> status
```

### Tasks

- [ ] Add private backend shape-plan hooks.
- [ ] Default static path builds one shape plan at compile.
- [ ] CPU_REF supports dynamic shape plans first.
- [ ] Metal supports dynamic shape plans second.
- [ ] Keep `execute_bound` public behavior but route through shape-plan lookup.
- [ ] Add backend conformance tests for static and dynamic runners.

---

## 10. GPT-VLM integration

### Training loop target

- [ ] Build dynamic graph once before train loop.
- [ ] Create one runner.
- [ ] For each batch:
  - materialize/bind tensors
  - bind `B`, `N_tokens`, `N_text`, `max_seq`
  - run runner
  - update scaler
- [ ] No graph destroy/reset in hot loop.
- [ ] Optional shape bucketing via cache policy only, not correctness.

### Debug visibility

- [ ] Print compile count and shape-plan hit/miss stats at end.
- [ ] Print arena high-water and cache size.
- [ ] In `GD_VLM_DEBUG`, report footprint after warmup and after cleanup.

---

## 11. Acceptance tests

### Unit

- [ ] `test_dynamic_shapes_cpu_runner_rebinds_alternating_shapes`.
- [ ] `test_dynamic_shapes_rejects_rank_dtype_device_layout_mismatch`.
- [ ] `test_dynamic_shape_symbol_constraints`.
- [ ] `test_dynamic_shape_plan_lru_eviction`.
- [ ] `test_dynamic_arena_high_water_reuse_cpu`.
- [ ] `test_dynamic_arena_high_water_reuse_metal`.
- [ ] `test_mps_plan_does_not_retain_matrices`.

### Op parity

- [ ] CPU dynamic vs CPU static for each covered op.
- [ ] Metal dynamic vs CPU dynamic alternating shapes.
- [ ] Linear/matmul MPS parity on alternating rows.
- [ ] SDPA varlen fwd/bwd parity on alternating `N_tokens/max_seq`.
- [ ] LMCE fwd/bwd parity on alternating rows.
- [ ] VLM pack fwd/bwd parity.

### Integration

```sh
GD_DEVICE=metal GD_METAL_MPS=1 GD_VLM_AMP_SCALE=1 \
make gpt-vlm GPT_VLM_ARGS="... --batch-size 16 --steps 1000 --lr 0"
```

Pass criteria:

- [ ] No `found_inf` after warmup unless true numeric overflow is injected.
- [ ] Finite loss and grad norms.
- [ ] Memory footprint plateaus.
- [ ] Graph compile count is 1 for stable topology.
- [ ] Shape-plan cache bounded by configured cap.
- [ ] CPU/Metal parity on alternating shapes.
- [ ] No MTLBuffer/MPS object leak under Instruments/ASan where applicable.

### Performance

- [ ] Static-shape benchmark regression <= 2% after dynamic infrastructure.
- [ ] Dynamic alternating-shape loop faster than rebuild/compile path after warmup.
- [ ] No per-step `gd_storage_create` after arena high-water reached.
- [ ] No per-step Metal buffer pool growth after warmup.

---

## 12. Implementation order

### P0 — Baseline and instrumentation

- [ ] Add compile/cache/allocation counters.
- [ ] Add VLM debug output for compile count, cache stats, arena bytes.
- [ ] Add failing alternating-shape tests documenting current behavior.

### P1 — Core shape symbols and dynamic input validation

- [ ] Add internal shape symbol structs.
- [ ] Add dynamic input API.
- [ ] Add runner dim binding and constraint validation.
- [ ] Keep static runner behavior unchanged.

### P2 — Runtime concrete shape inference

- [ ] Build per-run concrete desc table.
- [ ] Reuse op meta with concrete descs.
- [ ] Add dynamic attr binding.
- [ ] Add CPU_REF dynamic execution for simple ops.

### P3 — Shape-plan cache skeleton

- [ ] Add shape signature.
- [ ] Add bounded LRU.
- [ ] Route static graphs through one shape plan.
- [ ] Tests for cache hits/misses/eviction.

### P4 — Runner arena memory planner

- [ ] Add lifetime analysis.
- [ ] Add CPU arena allocation/reuse.
- [ ] Add Metal arena allocation/reuse.
- [ ] Preserve external alias/writeback semantics.

### P5 — MPS resource fix

- [ ] Stop retaining `MPSMatrix` wrappers.
- [ ] Move MPS GEMM resources into shape plan.
- [ ] Add no-retained-matrix test/instrumentation.

### P6 — Hot-path dynamic op coverage

- [ ] Linear/matmul.
- [ ] RMS norm / activations / elementwise.
- [ ] SDPA varlen fwd/bwd.
- [ ] LMCE fwd/bwd.
- [ ] AMP/optimizer path validation.

### P7 — VLM topology stabilization

- [ ] Add `gd_varlen_pack_prefix_suffix` op.
- [ ] Add CPU/Metal/autograd.
- [ ] Replace per-sample slice/concat graph build.
- [ ] Build VLM graph once and run dynamic batches.

### P8 — Production hardening

- [ ] Stress tests with thousands of alternating/random shapes.
- [ ] Instruments memory plateau validation.
- [ ] Error messages and docs.
- [ ] Static perf regression check.
- [ ] Dynamic perf benchmark and acceptance command.

---

## 13. Definition of done

- [ ] No per-batch graph rebuild needed for GPT-VLM variable text lengths.
- [ ] Dynamic execution is correct on CPU_REF and Metal.
- [ ] Shape cache is bounded and observable.
- [ ] Memory plateaus under long VLM run.
- [ ] MPS uses current buffers only; no stale transient resource retention.
- [ ] Static graph users see no API break and no meaningful perf regression.
- [ ] Docs describe static, dynamic, and bucketing modes clearly.
