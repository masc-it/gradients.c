# Production activation checkpointing design

This document is the production design for activation / recompute
checkpointing in `gradients.c`, with `examples/gpt_lm/` as the first concrete
user.

The goal is **generic region checkpointing** with a small, maintainable runtime
and autograd redesign. This is not a GPT-only feature, and it is not an
automatic compiler/planner.

## Design summary

A checkpointed region behaves like the same region run normally, under explicit
purity and determinism requirements.

Forward:

1. Push a scratch frame.
2. Run the region once with autograd recording suppressed.
3. Pack the region's fresh output tensors to the bottom of the frame.
4. Record one checkpoint node on the main tape using the packed output
   descriptors.

Backward:

1. Ensure main-tape grad slots exist for every differentiable boundary input and
   explicit parameter.
2. Push a recompute scratch frame.
3. Recompute the region forward while recording on a temporary recompute tape.
4. Run backward on the recompute tape.
5. Accumulate recompute grads into the already-existing main-tape grad slots.
6. Drop the recompute tape and pop the recompute scratch frame.

First production user: `gpt_block_forward()` in
`examples/gpt_lm/gpt_lm_shared.c`.

## Deliberate non-goals

Do not build these as part of this feature:

- automatic min-cut / memory-budget planning;
- selective per-op checkpoint policies;
- CPU/off-device activation offload;
- checkpointing KV-cache mutation paths;
- generic mutation tracking for arbitrary user C code;
- a general tape stack.

Nested checkpoint calls are handled transparently during recompute: if
`gd_checkpoint_region()` is called while already recomputing a checkpoint, it
runs the supplied function normally and records the inner function's ordinary ops
on the current recompute tape. It does not create a nested checkpoint node.

## Key production invariants

These invariants should drive the implementation and review:

1. **Checkpoint outputs are fresh.** A differentiable checkpoint output must have
   a fresh tensor id and must be produced by ops in the callback. Do not support
   pass-through outputs or duplicate output ids.
2. **Callbacks use provided params.** Parameters are passed to the callback as a
   separate array. The callback must use those descriptors, not model-owned
   descriptors captured through payload pointers.
3. **Recompute uses a sub-tape.** Recomputed ops must never be appended to the
   main tape.
4. **Main grad slots exist before recompute.** Main-tape gradient storage is
   allocated before opening the recompute scratch frame.
5. **Recompute grads are borrowed.** Accumulating recompute grads into main-tape
   slots must not release or adopt recompute-owned storage.
6. **Scratch frames are LIFO.** Frame cleanup is deterministic and local.
7. **Aux tensors are non-differentiable.** If an aux tensor requires grad, reject
   the checkpoint call.
8. **State writes are rejected.** Checkpointed regions cannot write state
   objects.

## Public API

Add either `include/gradients/checkpoint.h` or extend
`include/gradients/autograd.h`; include it from `include/gradients/gradients.h`.

Use a versioned options struct and a callback call-struct. The call-struct keeps
the callback signature stable as the API grows and, importantly, passes explicit
parameter descriptors to recompute.

```c
typedef struct gd_checkpoint_call {
    uint32_t size; /* sizeof(gd_checkpoint_call) */

    const gd_tensor *const *inputs;
    uint32_t n_inputs;

    const gd_tensor *const *params;
    uint32_t n_params;

    const gd_tensor *const *aux;
    uint32_t n_aux;

    gd_tensor *outputs;
    uint32_t n_outputs;
} gd_checkpoint_call;

typedef gd_status (*gd_checkpoint_fn)(gd_context *ctx,
                                      const void *payload,
                                      const gd_checkpoint_call *call);

typedef struct gd_checkpoint_options {
    uint32_t size;  /* sizeof(gd_checkpoint_options) */
    uint32_t flags; /* reserved; must be 0 for now */

    const gd_tensor *const *params;
    uint32_t n_params;

    const gd_tensor *const *aux;
    uint32_t n_aux;

    const void *payload;
    uint32_t payload_size;

    const char *debug_name;
    uint32_t reserved[4];
} gd_checkpoint_options;

gd_checkpoint_options gd_checkpoint_options_default(void);

gd_status gd_checkpoint_region(gd_context *ctx,
                               gd_checkpoint_fn fn,
                               const gd_checkpoint_options *options,
                               const gd_tensor *const *inputs,
                               uint32_t n_inputs,
                               gd_tensor *outputs,
                               uint32_t n_outputs);
```

Supported semantics:

- `n_outputs >= 1`;
- counts must fit existing tape node fields (`uint16_t` today);
- `inputs` are differentiable boundary inputs;
- `params` are explicit differentiable parameters captured by the callback;
- `aux` tensors are non-differentiable metadata and must not require grad;
- outputs must be fresh, unique tensor ids, not aliases of inputs/params/aux;
- outputs must be contiguous, non-view scratch tensors allocated inside the
  checkpoint frame;
- payload bytes are copied into tape attrs;
- pointers stored inside payload must remain valid until backward completes.

If a callback naturally returns an input unchanged, the caller should not wrap
that pass-through branch in a checkpoint region. Do not silently synthesize
identity edges in the checkpoint node.

## Callback parameter discipline

`params` exist for two reasons:

1. They are gradient targets.
2. They are the descriptors recompute must use.

Do not rely on payload pointers to model structs for parameter descriptors during
recompute. The payload may contain topology/static metadata and stable object
pointers, but math should read tensors from `call->params`.

For GPT this means either:

- add a helper that accepts explicit block parameter descriptors; or
- build a local shallow `gpt_block` / layer view from `call->params` and call the
  existing block forward helper.

The second option is acceptable only if the forward helpers read tensor fields
and static layer metadata, not module ownership state.

## Runtime purity rules

A checkpoint region must be deterministic and effectively pure:

- no mutation of params or input tensors;
- no state-object writes;
- no KV-cache append/update;
- no dependence on mutable global RNG state.

This must be enforced where practical. Add a context flag set while inside a
checkpoint forward or recompute region, and make `gd_state_object_acquire_span()`
reject `GD_STATE_WRITE` / `GD_STATE_READ_WRITE` while the flag is active.

Do not introduce global tensor mutation tracking as part of checkpointing. The
current tensor descriptors are caller-owned, so full mutation tracking would
require a separate tensor/storage registry redesign. Instead, checkpointing
supports the library's functional training graph and explicitly rejects state
writes. Mutating tensors between checkpoint forward and backward is an API
violation.

Dropout is acceptable because current dropout uses explicit stateless seeds.
GPT block seeds are deterministic from model seed, block index, site, and step.

## Backend support: ordered copies

Checkpoint output packing needs an ordered backend copy primitive:

```c
gd_status gd_backend_copy(gd_backend *backend,
                          gd_backend_buffer *dst_buffer,
                          size_t dst_offset,
                          gd_backend_buffer *src_buffer,
                          size_t src_offset,
                          size_t nbytes);
```

Contract:

- the copy is enqueued in backend order after prior kernels;
- later kernels observe the copied value;
- exact same-buffer no-op is valid;
- overlapping same-buffer copies must be handled with memmove semantics or
  avoided by scratch-frame staging before descriptor mutation.

Do not implement checkpoint packing with plain host `memcpy`; it must preserve
normal backend command ordering.

Files:

- `src/core/backend.h`
- `src/backends/metal/*`
- `src/backends/null/*`

## Scratch frames: memory-system redesign

The memory system needs a small frame abstraction, not loose ad-hoc scratch
marks. Frames make output packing and recompute cleanup predictable.

Files:

- `src/core/memory_internal.h`
- `src/core/memory.c`

Add internal APIs:

```c
#define GD_MAX_SCRATCH_FRAMES 8U

typedef struct gd_scratch_frame {
    int32_t slot;
    uint64_t generation;
    size_t mark;
} gd_scratch_frame;

gd_status gd_context_scratch_frame_push(gd_context *ctx,
                                        gd_scratch_frame *out);

/* Packs fresh output tensors down to frame.mark and updates descriptors in
 * place only after all copy commands have been successfully enqueued. */
gd_status gd_context_scratch_frame_pack_outputs(gd_context *ctx,
                                                const gd_scratch_frame *frame,
                                                gd_tensor *outputs,
                                                uint32_t n_outputs);

gd_status gd_context_scratch_frame_pop_abort(gd_context *ctx,
                                             const gd_scratch_frame *frame);
```

Context state should include a small fixed stack of frames and a depth counter.
Frames are LIFO only.

### Frame allocation policy

While a scratch frame is active, scratch allocation may reuse free-list blocks
**inside the top frame**, but must never reuse free-list blocks below the frame
mark. This preserves autograd's within-frame reuse during recompute backward
without letting checkpoint allocations consume older main-graph holes.

Rules:

- allocation from free-list is allowed only if the candidate block is fully
  within `[top_frame.mark, arena.offset)`;
- allocation from older free-list blocks below `top_frame.mark` is forbidden;
- `gd_context_free_span()` for a span inside the top frame may add a normal
  free-list block inside the frame;
- popping/aborting the frame removes all free-list blocks at or above the mark
  and resets `arena.offset` to the chosen post-frame offset.

### Why output packing is required

GPT block output is often allocated after large internal tensors. If the runtime
keeps that high-offset output span in place, the scratch arena offset remains
high and the next checkpointed block appends after old internals. That defeats
activation checkpointing.

Packing output tensors to `frame.mark` lets the next checkpointed block reuse the
same internal scratch range.

### Packing algorithm

`gd_context_scratch_frame_pack_outputs()` should:

1. Validate the frame is the top LIFO frame.
2. Validate every output:
   - fresh/unique tensor id;
   - scratch arena;
   - current slot/generation;
   - allocated inside `[frame.mark, arena.offset)`;
   - contiguous;
   - non-view.
3. Compute compact destination spans starting at `frame.mark`, preserving each
   output's alignment where possible.
4. Build a relocation plan before mutating any descriptor.
5. If a planned same-buffer copy overlaps another preserved source in a way that
   could clobber data, use a temporary scratch staging span inside the frame or
   fail before descriptor mutation.
6. Enqueue all backend copies.
7. Update output descriptors in place.
8. Remove free-list blocks at/after `frame.mark`.
9. Set `arena.offset` to the end of the compacted outputs.
10. Pop the frame.

If any step fails before descriptor mutation, abort the frame and return the
error. If copy enqueue succeeds but descriptor mutation would fail, that is an
internal bug; validate everything before enqueueing.

### Abort behavior

`gd_context_scratch_frame_pop_abort()` should:

- require the given frame to be top-of-stack;
- remove free-list entries at/after the mark;
- restore `arena.offset = frame.mark`;
- pop the frame;
- leave earlier allocations untouched.

## Avoid saved stats when recording is disabled

Some ops save forward stats when inputs require grad. That must also depend on
actual autograd recording.

Update at least:

- `src/ops/rms_norm/core_rms_norm.c`
- `src/ops/sdpa_varlen/core_sdpa_varlen.c`

Use:

```c
need_stats = gd_is_grad_enabled(ctx) &&
             (x->requires_grad || weight->requires_grad) &&
             gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
```

During checkpoint recompute, recording is enabled on the recompute tape, so
stats are allocated normally for backward. During the first no-record forward,
stats are not allocated.

## Autograd redesign: first-class tapes

Files:

- `src/autograd/autograd_internal.h`
- `src/autograd/autograd.c`

The current `gd_autograd_state` mixes engine/control state with tape storage.
Production checkpointing needs `gd_tape` as a first-class internal object.

```c
typedef struct gd_tape {
    gd_tape_node *nodes;
    uint32_t n_nodes;
    uint32_t cap_nodes;

    gd_tape_ref *refs;
    uint32_t n_refs;
    uint32_t cap_refs;

    unsigned char *attrs;
    uint32_t attrs_used;
    uint32_t attrs_cap;

    gd_grad_slot *grads;
    uint32_t n_grads;
    uint32_t cap_grads;

    gd_live_span_slot *live_spans;
    uint32_t n_live_spans;
    uint32_t cap_live_spans;

    bool owns_scratch_since_mark;
    int32_t owned_slot;
    uint64_t owned_generation;
    size_t owned_min_offset;
} gd_tape;

struct gd_autograd_state {
    bool recording;
    bool user_enabled;
    uint32_t checkpoint_recompute_depth;

    gd_tape main_tape;
    gd_tape recompute_tape;
    gd_tape *active_tape;
};
```

`gd_autograd_record()` appends to `state->active_tape`.

Autograd rules should receive a backward context pointing at exactly one tape:

```c
typedef struct gd_bwd_ctx {
    gd_context *ctx;
    gd_tape *tape;
} gd_bwd_ctx;
```

This is a focused redesign, not a general tape stack. Two tapes are sufficient
when nested checkpoint calls are transparent during recompute.

### Tape ref ownership

Add ownership metadata:

```c
typedef struct gd_tape_ref {
    gd_tensor tensor;
    uint64_t id;
    uint32_t version;
    bool owns_storage;
} gd_tape_ref;
```

Borrowed tensors from the outer graph appear as inputs/params on the recompute
tape. The recompute tape must not free their storage.

Ownership rule for scratch refs:

```c
owns_storage = tensor is scratch &&
               (!tape->owns_scratch_since_mark ||
                (tensor.storage.slot == tape->owned_slot &&
                 tensor.storage.generation == tape->owned_generation &&
                 tensor.storage.offset >= tape->owned_min_offset));
```

`gd_autograd_build_live_spans()` should count only refs with
`owns_storage == true`.

### Autograd guards

Do not use public `gd_set_grad_enabled()` internally; it mutates user intent.
Add internal guards:

```c
typedef struct gd_autograd_guard {
    bool recording;
    gd_tape *active_tape;
} gd_autograd_guard;

gd_status gd_autograd_no_record_begin(gd_context *ctx,
                                      gd_autograd_guard *guard);

gd_status gd_autograd_record_on_tape_begin(gd_context *ctx,
                                           gd_tape *tape,
                                           gd_autograd_guard *guard);

void gd_autograd_guard_end(gd_context *ctx,
                           const gd_autograd_guard *guard);
```

Every checkpoint path should have one cleanup block that restores:

- autograd guard state;
- scratch frame state;
- recompute depth;
- context checkpoint-purity flags.

### Attribute alignment

Checkpoint attrs contain a function pointer and variable payload. Make the tape
attribute arena alignment-safe before adding checkpoint attrs.

Preferred fix: align every node attr offset in `gd_autograd_record()` to
`alignof(max_align_t)` and zero any padding. This also removes latent alignment
risk from existing attrs.

Also add byte-oriented attr access for variable-size attrs:

```c
const void *gd_tape_attrs_bytes(const gd_tape *tape,
                                const gd_tape_node *node,
                                uint32_t min_size);
```

### Explicit-tape backward

Factor public backward into an internal helper:

```c
gd_status gd_backward_tape(gd_context *ctx,
                           gd_tape *tape,
                           uint32_t n_outputs,
                           const gd_tensor *const *outputs,
                           const gd_tensor *const *grad_outputs,
                           uint32_t flags);
```

Required flag:

```c
#define GD_BACKWARD_SEED_COPY 0x1U
```

Checkpoint backward must seed the recompute tape by copying the outer output
gradient. It must not let the recompute tape adopt a main-tape grad span.

## Gradient transfer discipline

Do not allocate main-tape grad slots inside the recompute scratch frame.

Checkpoint backward should:

1. Identify gradient targets:
   - boundary inputs;
   - explicit params.
2. Before opening the recompute frame, ensure a zero grad slot exists on the
   main tape for every target that requires grad.
3. Open the recompute frame.
4. Recompute and run sub-tape backward.
5. Accumulate recompute grads into existing main-tape slots.
6. Drop the recompute frame.

Add helpers:

```c
gd_status gd_autograd_ensure_zero_grad_slot(gd_bwd_ctx *bwd,
                                            uint64_t tensor_id,
                                            const gd_tensor *like);

gd_status gd_autograd_accumulate_borrowed(gd_bwd_ctx *bwd,
                                          uint64_t tensor_id,
                                          const gd_tensor *contrib);
```

`gd_autograd_accumulate_borrowed()` must not release `contrib`; the recompute
tape/frame owns it.

## Checkpoint node

Files:

- `src/autograd/checkpoint.c`
- `src/ops/checkpoint/autograd_checkpoint.c`
- generated registry files via `tools/gen_ops.c`

Add an op capsule under `src/ops/checkpoint/` with only an autograd rule.
`tools/gen_ops.c` supports ops with `autograd_*.c` and no public core op.

Attrs:

```c
typedef struct gd_checkpoint_attrs {
    gd_checkpoint_fn fn;
    uint32_t n_boundary_inputs;
    uint32_t n_params;
    uint32_t n_aux;
    uint32_t n_outputs;
    uint32_t payload_size;
    /* payload bytes follow */
} gd_checkpoint_attrs;
```

### Forward algorithm

Record the checkpoint node after output packing so the tape stores compact output
descriptors.

```c
if (!gd_is_grad_enabled(ctx) || gd_context_scope_mode(ctx) != GD_SCOPE_TRAIN) {
    build gd_checkpoint_call from caller arrays;
    return fn(ctx, payload, &call);
}

if state->checkpoint_recompute_depth != 0:
    /* transparent nested call */
    build call;
    return fn(ctx, payload, &call);

validate options;
validate aux tensors do not require grad;
if no input/param requires grad:
    build call;
    return fn(ctx, payload, &call);

push scratch frame;
begin no-record guard;
build call using caller descriptors;
st = fn(ctx, payload, &call);
end guard;
if st != GD_OK:
    pop_abort frame;
    return st;

validate outputs are fresh, unique, non-view, contiguous scratch tensors inside frame;
pack outputs;

record GD_OP_CHECKPOINT:
    inputs = boundary inputs followed by explicit params
    saved = aux tensors
    outputs = packed outputs
    attrs = copied checkpoint attrs + payload bytes
```

### Backward algorithm

```c
checkpoint_backward(outer_bwd, node):
    read attrs;
    collect boundary inputs, params, aux, original outputs;
    get outer output grads;

    ensure zero main grad slots for every requires-grad boundary input/param;

    push recompute scratch frame;
    reset recompute_tape and configure its owned scratch mark;

    checkpoint_recompute_depth++;
    begin record-on-recompute-tape guard;
    build gd_checkpoint_call using tape input refs, tape param refs, tape aux refs;
    call fn(ctx, payload, &call);
    end guard;
    checkpoint_recompute_depth--;

    validate recomputed outputs are fresh, unique, and match original dtype/rank/shape;

    run gd_backward_tape(..., GD_BACKWARD_SEED_COPY) on recompute_tape;

    for each requires-grad boundary input/param:
        if recompute grad exists:
            gd_autograd_accumulate_borrowed(main_bwd, original_id, recompute_grad);

    reset recompute_tape;
    pop_abort recompute scratch frame;
```

The core invariant: **recompute nodes never enter the main tape**.

## GPT integration

Files:

- `examples/gpt_lm/gpt_lm_shared.h`
- `examples/gpt_lm/gpt_lm_shared.c`
- `examples/gpt_lm/main.c`
- `examples/gpt_lm/README.md`

Config:

```c
typedef enum gpt_activation_checkpointing {
    GPT_ACT_CKPT_NONE = 0,
    GPT_ACT_CKPT_BLOCK = 1,
} gpt_activation_checkpointing;
```

Add to both `gpt_config` and `gpt_lm`:

```c
gpt_activation_checkpointing activation_checkpointing;
```

YAML:

```yaml
activation_checkpointing: none   # none | block
```

Callback payload:

```c
typedef struct gpt_block_ckpt_payload {
    gpt_lm *model;
    uint32_t block_index;
    uint64_t step;
} gpt_block_ckpt_payload;
```

The callback must use `call->params`, not `model->block_items[i]` tensor
descriptors directly. It may use the model/block for static metadata.

Callback outline:

```c
static gd_status gpt_block_ckpt_fn(gd_context *ctx,
                                   const void *payload,
                                   const gd_checkpoint_call *call)
{
    const gpt_block_ckpt_payload *p = (const gpt_block_ckpt_payload *)payload;
    gpt_block block_view;

    if (p == NULL || call == NULL || call->n_inputs != 1U ||
        call->n_aux != 2U || call->n_outputs != 1U) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    /* Build block_view from model static metadata and call->params in the
     * documented order, then call gpt_block_forward(). */

    return gpt_block_forward(ctx,
                             p->model,
                             &block_view,
                             p->block_index,
                             call->inputs[0],
                             call->aux[0], /* positions */
                             call->aux[1], /* cu_seqlens */
                             p->step,
                             &call->outputs[0]);
}
```

Explicit block param order:

```c
params[n_params++] = &block->attn_norm_w;
params[n_params++] = &block->mlp_norm_w;
params[n_params++] = &block->qkv_proj.weight;
if (block->qkv_proj.has_bias) params[n_params++] = &block->qkv_proj.bias;
params[n_params++] = &block->attn_proj.weight;
if (block->attn_proj.has_bias) params[n_params++] = &block->attn_proj.bias;
params[n_params++] = &block->up_gate.weight;
if (block->up_gate.has_bias) params[n_params++] = &block->up_gate.bias;
params[n_params++] = &block->down_proj.weight;
if (block->down_proj.has_bias) params[n_params++] = &block->down_proj.bias;
```

In `gpt_lm_forward()`:

```c
if (model->activation_checkpointing == GPT_ACT_CKPT_BLOCK &&
    model->mod.training &&
    gd_is_grad_enabled(ctx)) {
    st = gpt_block_forward_checkpointed(...);
} else {
    st = gpt_block_forward(...);
}
```

Do not checkpoint `gpt_block_prefill_cached()`, decode, generation, or any path
that mutates KV cache state.

## Test plan

Add `tests/test_checkpoint.c`.

Required tests:

1. **Gradient equality**
   - checkpointed vs normal MLP region;
   - compare input and parameter grads.

2. **Multiple-output correctness**
   - region with two fresh outputs and two output grad seeds.

3. **Pass-through rejection**
   - callback returns an input descriptor or duplicate output id;
   - checkpoint call rejects it clearly.

4. **Callback uses param array**
   - recompute callback receives param descriptors from tape refs;
   - test fails if callback uses stale external descriptors.

5. **Scratch packing**
   - region allocates large internal tensors and smaller outputs;
   - verify outputs are compacted near the frame mark;
   - verify a following checkpoint reuses scratch rather than appending after old
     internals.

6. **Packing overlap/staging**
   - construct output layouts that require safe overlapping same-buffer handling
     or verify the function rejects before descriptor mutation.

7. **Scratch memory reduction**
   - chain checkpointed regions;
   - watermark is close to one region plus packed boundaries, not sum of all
     region internals.

8. **Dropout determinism**
   - fixed seed;
   - checkpointed and normal gradients match.

9. **Borrowed tensor safety**
   - recompute tape references outer boundary tensors and params;
   - those tensors are not freed by recompute cleanup.

10. **Seed-copy safety**
    - recompute backward cannot adopt/free the outer output grad.

11. **Main grad slot allocation discipline**
    - main grad slots are allocated before recompute frame;
    - recompute frame abort does not invalidate them.

12. **Aux validation**
    - aux tensor requiring grad is rejected.

13. **State mutation rejection**
    - checkpoint callback attempting state-object write is rejected.

14. **Transparent nested checkpoint call**
    - checkpoint call inside recompute records normal inner ops, not a nested
      checkpoint node.

15. **Error cleanup**
    - callback returns error during forward and recompute;
    - recording state, active tape, recompute depth, purity flags, and scratch
      frame stack are restored.

16. **GPT smoke**
    - tiny GPT config;
    - one train step with `activation_checkpointing: block`.

Run:

```sh
make test
SAN=1 make test
```

## Work split for five maintainers

1. **Memory/backend owner**
   - `gd_backend_copy()`;
   - scratch frame stack;
   - frame-bounded free-list reuse;
   - output packing and overlap/staging tests.

2. **Autograd owner**
   - `gd_tape` split;
   - attr alignment;
   - active tape switching;
   - recording guards;
   - owned/borrowed refs;
   - explicit-tape backward.

3. **Checkpoint owner**
   - public API and callback call-struct;
   - checkpoint node/rule;
   - output freshness validation;
   - zero main-grad-slot discipline;
   - borrowed accumulation;
   - transparent nested behavior.

4. **GPT owner**
   - config parsing;
   - block wrapper;
   - callback uses explicit params;
   - README/config examples;
   - GPT smoke test.

5. **QA/docs owner**
   - numerical parity tests;
   - determinism tests;
   - mutation/error tests;
   - memory benchmarks;
   - reviewer checklist.

## Review checklist

Before merging:

- [ ] checkpoint outputs are fresh, unique, non-view, contiguous scratch tensors;
- [ ] pass-through outputs are rejected;
- [ ] callbacks receive and use explicit param descriptors;
- [ ] checkpoint outputs are packed before the checkpoint node is recorded;
- [ ] following checkpoint regions reuse scratch memory;
- [ ] scratch frames are LIFO and frame-bounded allocation is enforced;
- [ ] recompute nodes are never appended to the main tape;
- [ ] recompute backward seed grads are copied, never adopted from main tape;
- [ ] main-tape grad slots are allocated before recompute frame begins;
- [ ] recompute grad accumulation treats recompute grads as borrowed;
- [ ] borrowed outer tensors are not freed by recompute tape cleanup;
- [ ] aux tensors requiring grad are rejected;
- [ ] state-object writes inside checkpoint regions are rejected;
- [ ] nested checkpoint calls during recompute are transparent;
- [ ] attrs are alignment-safe;
- [ ] guards restore recording state and active tape on every error path;
- [ ] GPT checkpointing is disabled for eval/infer/KV-cache paths;
- [ ] dropout recomputation is deterministic;
- [ ] public docs state purity, determinism, aux, output freshness, and payload
      lifetime rules.

## Do not do

Do not:

- solve this by only calling `gd_set_grad_enabled(false)`;
- keep checkpoint outputs at high offsets after internal temporaries;
- allow pass-through outputs without explicit identity-edge support;
- let callbacks recompute from model-owned param descriptors instead of
  `call->params`;
- append recompute ops to the main tape;
- let recompute tape adopt main-tape grad storage;
- allocate main-tape grad slots inside recompute scratch frames;
- silently allow differentiable aux tensors;
- store stack payload pointers in tape attrs;
- checkpoint KV-cache mutation paths;
- introduce a global tensor registry or broad mutation-tracking redesign as part
  of this feature;
- start with min-cut planning, offload, or selective policies.

Production target: **generic region checkpointing with fresh compacted boundary
outputs, explicit parameter capture, scratch frames, a recompute sub-tape, and
GPT block integration**.
