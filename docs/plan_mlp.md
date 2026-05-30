# gradients.c MLP Foundation Plan

Status: draft v0.1

Goal: implement enough of [`docs/design_spec.md`](design_spec.md) to train and debug a small MLP end-to-end on CPU_REF through the graph-only execution model.

Non-goal: GPU performance, quantized training, transformer kernels, dynamic shapes, async streams.

Reference codebase: `/Users/mascit/projects/dnn.c/`. Use it only as implementation reference for op math and backward formulas, especially `src/ops_*.c`, `src/nn.c`, `src/autograd.c`, and `src/optim.c`. Do not port fast paths, SIMD, OpenMP, BLAS optimizations, or pool-lifetime assumptions into this MLP foundation. CPU_REF kernels should be simple scalar/reference implementations first.

Success criterion: one example trains a tiny MLP with `gd_graph` + `gd_backward` + `gd_optimizer_step`, passes gradcheck on core ops, and can dump/inspect graph IR.

---

## 1. Scope

MLP foundation requires these spec pillars to work:

- C public API with opaque handles
- context/status/error handling
- tensor/storage separation
- graph-only IR execution
- CPU_REF backend as correctness oracle
- basic op schemas + CPU kernels
- autograd as IR transform
- module/parameter registration
- AdamW optimizer
- debug graph dump/validate/materialize

No Metal yet. No quant yet except type/descriptor stubs where needed for ABI compatibility.

---

## 2. Target user code

Desired end-state API shape:

```c
gd_context *ctx;
gd_context_create(&ctx);
gd_context_set_default_device(ctx, (gd_device){GD_DEVICE_CPU, 0});
gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF);

// params
mlp *model = mlp_create(ctx, in_dim, hidden_dim, out_dim);
int n_params;
gd_tensor **params;
gd_module_parameters(model->base, &params, &n_params);

gd_optimizer *opt;
gd_adamw_create(ctx, params, n_params, &adamw_cfg, &opt);

// static batch tensors reused each step
gd_tensor *x = gd_tensor_empty(... [B, in_dim] ...);
gd_tensor *y = gd_tensor_empty(... [B] I32 ...);

// build one train-step graph
gd_graph *g;
gd_graph_create(ctx, &g);
gd_graph_begin(ctx, g);

    gd_tensor *logits = mlp_forward(ctx, model, x);
    gd_tensor *loss;
    gd_cross_entropy(ctx, logits, y, 1, &loss);
    gd_backward(ctx, loss);
    gd_optimizer_step(ctx, opt);
    gd_optimizer_zero_grad(ctx, opt);

gd_graph_end(ctx);
gd_graph_validate(g);
gd_graph_compile(g, (gd_device){GD_DEVICE_CPU, 0});

for each batch:
    gd_tensor_copy_from_cpu(ctx, x, xb, xb_bytes);
    gd_tensor_copy_from_cpu(ctx, y, yb, yb_bytes);
    gd_graph_run(g);
```

---

## 3. Implementation phases

Phase status checklist:

- [x] Phase 0 — Build skeleton
- [x] Phase 1 — Status, context, dtype, device
- [x] Phase 2 — Storage and tensor metadata
- [x] Phase 3 — Graph IR core
- [x] Phase 4 — Op schemas and shape inference
- [x] Phase 5 — CPU_REF graph executor
- [x] Phase 6 — Autograd v1
- [x] Phase 7 — Module system
- [x] Phase 8 — AdamW optimizer
- [x] Phase 9 — Debug tools
- [x] Phase 10 — MLP example

### Phase 0 — Build skeleton

- [x] Phase complete

Files:

```text
include/gradients/
  gradients.h
  status.h
  dtype.h
  device.h
  tensor.h
  graph.h
  ops.h
  module.h
  optim.h
  quant.h        // stubs only for now

src/core/
  status.c
  context.c
  dtype.c
  storage.c
  tensor.c
  refcount.h

src/graph/
  graph.c
  ir.c
  validate.c
  dump.c

src/ops/
  op_schema.c
  shape.c

src/backends/cpu_ref/
  cpu_ref.c
  cpu_kernels.c

src/autograd/
  autograd.c

src/nn/
  module.c
  optim.c

tests/
examples/mlp/
```

Deliver:

- compile static library
- one smoke test creates/destroys `gd_context`

---

### Phase 1 — Status, context, dtype, device

- [x] Phase complete

Implement:

- `gd_status`
- thread-local `gd_last_error()`
- `gd_context_create/destroy`
- default device
- fallback policy
- default compute policy
- `gd_synchronize` CPU no-op
- dtype helpers:
  - `gd_dtype_sizeof`
  - `gd_dtype_name`

CPU only supported initially:

```text
GD_DEVICE_CPU index 0
```

Return `GD_ERR_UNSUPPORTED` for Metal/CUDA/Vulkan until implemented.

Tests:

- context defaults
- fallback policy setters/getters
- dtype size/name
- invalid device errors

---

### Phase 2 — Storage and tensor metadata

- [x] Phase complete

Implement storage:

- `gd_storage_desc`
- `gd_storage_create`
- retain/release
- CPU aligned allocation
- CPU copy from/to
- `gd_storage_data_cpu`

Implement tensor:

- `gd_tensor_desc_contiguous`
- `gd_tensor_desc_nbytes`
- `gd_tensor_empty`
- `gd_tensor_from_storage`
- retain/release
- shape/dtype/device/layout queries
- `gd_tensor_copy_from_cpu/to_cpu`
- metadata views:
  - reshape when contiguous-compatible
  - transpose
  - slice

For this phase, `GD_LAYOUT_CONTIGUOUS` and `GD_LAYOUT_STRIDED` only.

Defer:

- packed quant physical sizing beyond returning unsupported
- backend opaque/block layouts

Tests:

- create contiguous F32/I32 tensors
- copy roundtrip CPU
- reshape/transpose/slice metadata
- view retains storage
- bad desc validation

---

### Phase 3 — Graph IR core

- [x] Phase complete

Implement core IR structs privately:

```text
_gd_value
_gd_node
_gd_graph_state: empty/building/finalized/compiled
_gd_op_kind
_gd_op_attrs
```

Implement public graph API:

- `gd_graph_create`
- `gd_graph_begin`
- `gd_graph_end`
- `gd_graph_validate`
- `gd_graph_reset`
- `gd_graph_destroy`
- `gd_graph_dump(GD_DUMP_TEXT)`

Rules:

- only one active graph per context
- compute ops require active graph
- graph reset/destroy checks live virtual tensors
- graph nodes get stable ids

Virtual tensors:

- op outputs are `gd_tensor` handles backed by `_gd_value`
- no storage until run/materialize
- release decrements graph live virtual count

Tests:

- begin/end lifecycle
- nested graph begin fails
- op outside graph fails once ops exist
- dump contains nodes/shapes
- graph reset fails with live virtual tensor

---

### Phase 4 — Op schemas and shape inference

- [x] Phase complete

Implement op recording and shape inference for:

- `gd_add`
- `gd_mul`
- `gd_scale`
- `gd_matmul` / `gd_matmul_ex`
- `gd_linear` / `gd_linear_ex`
- `gd_relu`
- `gd_silu`
- `gd_sum`
- `gd_mean`
- `gd_softmax`
- `gd_cross_entropy`
- `gd_cast`

V1 rules:

- no implicit promotion
- elementwise same dtype
- broadcasting for add/mul
- matmul supports 2D and batched ND
- reductions accept negative dims
- CE targets I32/I64
- only F32 CPU kernels initially; add I32 support for labels/cast/copy

Tests:

- shape inference for each op
- dtype errors
- broadcast errors
- negative dim normalization
- linear `trans_w` for tied LM-head-style use

---

### Phase 5 — CPU_REF graph executor

- [x] Phase complete

Implement CPU reference executor:

- `gd_graph_compile` for CPU:
  - validate finalized graph
  - simple executable = node list
  - naive allocation plan: allocate each materialized value separately first
- `gd_graph_run`
  - execute nodes in order
  - materialize graph outputs/intermediates needed by later nodes

CPU kernels F32 first:

- add/mul with broadcasting
- scale
- matmul 2D + batched ND simple loops
- linear as fused or composition in executor
- relu/silu
- sum/mean
- softmax stable rowwise
- cross entropy mean
- cast F32/I32 where needed

Debug/materialization:

- `gd_tensor_materialize`
- `gd_tensor_to_cpu`
- `gd_debug_print_tensor`
- `gd_graph_run_immediate`

Tests:

- each op numeric expected
- one-shot immediate vs explicit graph
- MLP forward only
- CE on small logits

---

### Phase 6 — Autograd v1

- [x] Phase complete

Implement autograd metadata:

- `requires_grad`
- leaf grad slot
- `gd_tensor_grad`
- `gd_zero_grad`
- no-grad guard

Implement backward graph generation for:

- add
- mul
- scale
- matmul
- linear
- relu
- silu
- sum
- mean
- softmax + cross_entropy path
- cast: no grad unless safe float-to-float cast; can be unsupported in v1 backward

Approach:

- `gd_backward(ctx, loss)` traverses forward IR reachable from loss
- appends backward IR nodes into active graph
- accumulates into leaf grad tensors with add nodes if multiple paths
- targets/indices marked non-differentiable

Keep first implementation simple:

- F32 gradients only
- no higher-order gradients
- no in-place ops

Implement gradcheck:

- CPU finite difference
- small tensors only
- compare analytic grad from CPU_REF graph

Tests:

- gradcheck add/mul/scale/matmul/relu/silu/sum/mean/CE
- multi-use tensor gradient accumulation
- requires_grad invalid on I32/bool/quant

---

### Phase 7 — Module system

- [x] Phase complete

Implement:

- `gd_module_create/destroy`
- param registration
- child registration
- recursive `gd_module_parameters`
- dedup by `(storage, byte offset, extent)`
- `gd_module_zero_grad(ctx, m)`

Defer full state dict binary format until after MLP. Provide optional text/debug summary if useful.

Tests:

- param collection order
- child recursion
- tied parameter dedup
- module-owned flat array lifetime

---

### Phase 8 — AdamW optimizer

- [x] Phase complete

Implement:

- `gd_adamw_create`
- optimizer retains params
- state tensors `m`, `v`, step counter
- `gd_optimizer_step(ctx,opt)` records IR nodes
- `gd_optimizer_zero_grad(ctx,opt)` records zero-grad nodes

For CPU_REF v1, optimizer update can be a dedicated IR op:

```text
ADAMW_STEP(params, grads, m, v, config, step)
ZERO_GRAD(params)
```

This avoids needing many small public ops for update math.

Tests:

- one param AdamW update vs hand calculation
- zero grad
- tied param only one optimizer state

---

### Phase 9 — Debug tools

- [x] Phase complete

Implement now, before examples grow:

- `gd_scope_push/pop`
- `gd_tensor_set_name`
- graph dump includes:
  - node id
  - op
  - tensor names
  - shapes
  - dtype
  - device
  - requires_grad
- `gd_assert_finite`
- `gd_assert_close`
- `gd_graph_run_until`

Tests:

- names/scopes appear in dump
- assert finite catches NaN
- run_until materializes expected intermediate

---

### Phase 10 — MLP example

- [x] Phase complete

Implement example:

```text
examples/mlp/mlp.c
```

Model:

```text
x -> linear -> relu/silu -> linear -> cross_entropy
```

Dataset: XOR (implemented in `examples/mlp/mlp.c`).

Other options considered:

1. spiral synthetic classification
2. tiny generated Gaussian blobs

Use fixed batch shape to match static graph policy.

Training loop:

1. allocate batch tensors once
2. build train-step graph once
3. for each step, copy new batch into same tensors
4. run graph
5. periodically materialize loss

Target:

- loss decreases
- accuracy improves on synthetic set
- graph dump readable

---

## 4. Minimal test matrix

Required before calling foundation working:

```text
Core:
  context/status/dtype/device
  storage/tensor/view/copy
  graph lifecycle/dump/validate

Ops forward:
  add/mul/scale/matmul/linear/relu/silu/sum/mean/softmax/CE/cast

Autograd:
  gradcheck for all differentiable MVP ops
  gradient accumulation
  no-grad guard

Training:
  AdamW numeric update
  MLP loss decreases
  tied param dedup smoke
```

---

## 5. Deliberate deferrals

Not needed for MLP foundation:

- Metal backend
- CUDA/Vulkan
- quant pack/dequant
- packed quant ops
- state dict binary format
- transformer/attention/RoPE
- memory reuse planner beyond naive allocation
- activation checkpointing
- dynamic shapes
- async streams/events
- graph if/while

---

## 6. Risks and mitigations

### Risk: graph-only makes simple tests verbose

Mitigation: implement `gd_graph_run_immediate` early and use it in op tests.

### Risk: autograd graph transform gets complex

Mitigation: keep MVP ops small, F32-only, no in-place ops, no higher-order grad.

### Risk: static graph plus changing batches confusing

Mitigation: examples reuse same input tensor handles and copy new bytes into them before each run.

### Risk: optimizer as IR op feels special

Mitigation: accept special optimizer nodes in v1; lower them in CPU_REF. Later decompose or fuse per backend.

### Risk: matmul backward bugs

Mitigation: gradcheck many tiny shape cases, including batched and transposed flags.

---

## 7. Definition of done

Foundation complete when:

1. `make test` passes all core/op/autograd tests.
2. `examples/mlp` trains a synthetic classifier with decreasing loss.
3. `gd_graph_dump` shows readable forward + backward + optimizer nodes.
4. `gd_gradcheck` passes for MVP differentiable ops.
5. No compute op executes outside graph path; immediate helper uses one-shot graph internally.

---

## 8. Post-foundation increments

Done after the core foundation (still CPU_REF, XOR remains the integration check):

- [x] Broadcast backward for `add`/`mul` via internal `reduce_to` op (sum over broadcast dims). Gradchecked with `add_broadcast`.
- [x] `gd_debug_print_tensor` for F32/I32 (materialized and post-run virtual tensors), used in the MLP example and tests.

- [x] Transposed/batched matmul and `trans_w` linear backward. All input grads reformulated as trans-flagged matmuls + broadcast reduction (no transpose op needed). Gradchecked with `matmul_trans_a`, `matmul_batched_bcast`, `linear_trans_w`.

- [x] Materializing `gd_tensor_contiguous` for non-contiguous inputs. Eager host gather producing a fresh contiguous tensor from strided materialized views (transpose/slice); identity (retain) when already contiguous. Tested via transpose -> contiguous gather. Note: a graph-recorded/layout-planner variant is deferred to the compiler passes.

- [x] `gd_tensor_view` / `gd_tensor_reshape` of virtual (graph) tensors. Recorded as a functional reshape (element-order-preserving, equal numel) via a dtype-agnostic `COPY` node; observationally a zero-copy view since values are immutable. Backward flows the gradient back reshaped. Tested numerically and gradchecked (`reshape`).

Still open (next):

- [ ] Metal backend (first GPU target per design spec).
- [ ] Layout planner / buffer reuse (graph-recorded contiguous + true aliasing views).
