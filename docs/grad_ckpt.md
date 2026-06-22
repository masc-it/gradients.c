# Activation checkpointing implementation plan

This document describes a maintainable implementation plan for activation / recompute checkpointing in `gradients.c`, with `examples/gpt_lm/` as the first concrete user.

The goal is **block-level GPT activation checkpointing first**, implemented as a reusable autograd/runtime feature. Do not implement this as a GPT-only shortcut.

## Scope

MVP:

- checkpoint pure training forward regions;
- save only region boundary tensors, explicit params, aux tensors, and a small copied payload;
- run the region forward once with recording disabled;
- free internal scratch tensors from that first forward;
- during backward, recompute the region on a temporary tape;
- run backward on the temporary tape and accumulate grads into the outer tape.

Out of scope for MVP:

- KV-cache / inference checkpointing;
- automatic min-cut planning;
- selective per-op policies;
- CPU/off-device activation offload;
- multi-output checkpoint regions, unless easy after the single-output implementation.

## High-level design

Add a generic checkpoint API, then wire GPT blocks through it.

Proposed public API:

```c
typedef gd_status (*gd_checkpoint_fn)(
    gd_context *ctx,
    const void *payload,
    const gd_tensor *const *inputs,
    uint32_t n_inputs,
    const gd_tensor *const *aux,
    uint32_t n_aux,
    gd_tensor *outputs,
    uint32_t n_outputs);

typedef struct gd_checkpoint_options {
    const gd_tensor *const *params;
    uint32_t n_params;
    const gd_tensor *const *aux;
    uint32_t n_aux;
    const void *payload;
    uint32_t payload_size;
    const char *debug_name;
} gd_checkpoint_options;

gd_status gd_checkpoint_region(gd_context *ctx,
                               gd_checkpoint_fn fn,
                               const gd_checkpoint_options *options,
                               const gd_tensor *const *inputs,
                               uint32_t n_inputs,
                               gd_tensor *outputs,
                               uint32_t n_outputs);
```

For the first pass, require `n_outputs == 1`. Keep the shape of the API ready for multiple outputs.

## Implementation phases

## Phase 1: scratch scopes

Files:

- `src/core/memory_internal.h`
- `src/core/memory.c`

Add an internal scratch scope API:

```c
typedef struct gd_scratch_scope {
    int32_t slot;
    uint64_t generation;
    size_t offset;
} gd_scratch_scope;

gd_status gd_context_scratch_scope_begin(gd_context *ctx,
                                         gd_scratch_scope *out);

gd_status gd_context_scratch_scope_end(gd_context *ctx,
                                       const gd_scratch_scope *scope,
                                       const gd_span *const *keep,
                                       uint32_t n_keep);
```

The checkpoint forward path should use it like this:

```c
gd_context_scratch_scope_begin(ctx, &scope);
/* run region forward with autograd recording disabled */
gd_context_scratch_scope_end(ctx, &scope, output_spans, n_outputs);
```

### Important allocator rule

`gd_arena_alloc()` currently prefers scratch free-list holes. While a scratch scope is open, scratch allocation should be monotonic from `arena->offset` and should not reuse older free-list holes. This makes scope cleanup tractable:

- allocations before `scope.offset` are outside the scope;
- allocations at or after `scope.offset` are owned by the scope;
- at scope end, free/drop everything after the mark except the explicit keep spans.

Add a small per-context counter/flag for active scratch scopes, for example:

```c
uint32_t scratch_scope_depth;
```

Then skip `gd_arena_alloc_from_free()` for scratch when `scratch_scope_depth > 0`.

### Keep-span behavior

The MVP can support the common case where checkpoint output storage is the last allocation in the scope. That is enough for GPT blocks because the block output is produced at the end.

For higher quality, implement general keep-span support:

1. Validate all keep spans are scratch spans in the current slot/generation and are fully within `[scope.offset, arena->offset)`.
2. Sort keep intervals by offset.
3. Free gaps between the scope mark and keep intervals.
4. Leave keep spans live.
5. Do not roll `arena->offset` below the highest kept interval end.
6. If no keep spans exist, set `arena->offset = scope.offset` and drop free blocks above the mark.

This logic belongs in `memory.c`; checkpoint/autograd code should not manipulate arena internals directly.

### Avoid saved-stat allocations when recording is disabled

Some ops already avoid saved stats when not training, but they also need to check autograd recording.

Update examples:

- `src/ops/rms_norm/core_rms_norm.c`
- `src/ops/sdpa_varlen/core_sdpa_varlen.c`

Use:

```c
need_stats = gd_is_grad_enabled(ctx) &&
             (x->requires_grad || weight->requires_grad) &&
             gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
```

Dropout still needs a forward mask to produce the output. The scratch scope will free that mask after checkpoint forward.

## Phase 2: split autograd state from tape

Files:

- `src/autograd/autograd_internal.h`
- `src/autograd/autograd.c`

Right now `gd_autograd_state` is both the engine state and the tape. Checkpointing needs a temporary recompute tape, so split this conceptually:

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
    gd_tape main_tape;
    gd_tape recompute_tape;
    gd_tape *active_tape;
};
```

`gd_autograd_record()` should append to `state->active_tape`.

Main backward uses `main_tape`. Checkpoint backward switches `active_tape` to `recompute_tape` temporarily.

### Borrowed vs owned tape refs

Add ownership metadata to refs:

```c
typedef struct gd_tape_ref {
    gd_tensor tensor;
    uint64_t id;
    uint32_t version;
    bool owns_storage;
} gd_tape_ref;
```

Why this matters:

- recompute tape inputs include borrowed boundary tensors from the outer forward;
- recompute outputs/intermediates are owned by the recompute scratch scope;
- recompute tape cleanup must not free borrowed outer tensors.

Ownership rule:

```c
owns_storage = tensor is scratch &&
               (!tape->owns_scratch_since_mark ||
                (tensor.storage.slot == tape->owned_slot &&
                 tensor.storage.generation == tape->owned_generation &&
                 tensor.storage.offset >= tape->owned_min_offset));
```

Then `gd_autograd_build_live_spans()` should only add refs whose `owns_storage` is true.

### Nested backward helper

Factor current backward loop into an internal helper that accepts an explicit tape:

```c
gd_status gd_backward_tape(gd_context *ctx,
                           gd_tape *tape,
                           uint32_t n_outputs,
                           const gd_tensor *const *outputs,
                           const gd_tensor *const *grad_outputs,
                           float scale);
```

Public `gd_backward()`, `gd_backward_many()`, and AMP variants use `main_tape`.

Checkpoint backward uses `recompute_tape`.

### Safe grad transfer

Add an internal accumulate-copy helper:

```c
gd_status gd_autograd_accumulate_copy(gd_bwd_ctx *bwd,
                                      uint64_t tensor_id,
                                      const gd_tensor *contrib);
```

Checkpoint backward should transfer recompute grads into the outer tape deliberately. Do not rely on accidental adoption/move semantics across two different tapes.

## Phase 3: checkpoint op and API

Files:

- `include/gradients/checkpoint.h` or `include/gradients/autograd.h`
- `include/gradients/gradients.h`
- `src/autograd/checkpoint.c`
- `src/ops/checkpoint/autograd_checkpoint.c`
- generated files via `tools/gen_ops.c`

Add an internal op kind `GD_OP_CHECKPOINT` by adding an op capsule under `src/ops/checkpoint/` with an autograd rule.

Forward behavior in `gd_checkpoint_region()`:

```c
if (!gd_is_grad_enabled(ctx) || gd_context_scope_mode(ctx) != GD_SCOPE_TRAIN) {
    return fn(ctx, payload, inputs, n_inputs, aux, n_aux, outputs, n_outputs);
}

begin scratch scope;
disable autograd recording;
fn(ctx, payload, inputs, n_inputs, aux, n_aux, outputs, n_outputs);
restore autograd recording;

record GD_OP_CHECKPOINT node:
    inputs = boundary inputs followed by explicit params
    saved = aux tensors
    outputs = checkpoint outputs
    attrs = fn pointer, counts, debug id/name if used, copied payload bytes

end scratch scope keeping only output spans;
```

Store the payload by value in tape attrs. Do not store a pointer to stack payload memory.

Attrs should look roughly like:

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

Validate `payload_size` against the existing autograd attr capacity.

Backward behavior in `autograd_checkpoint.c`:

```c
checkpoint_backward(bwd, node):
    get outer output grad;

    begin recompute scratch scope;
    reset recompute_tape;
    configure recompute_tape ownership mark;

    switch active_tape = recompute_tape;
    enable recording;
    call fn(...) again using:
        boundary inputs from node inputs[0:n_boundary_inputs]
        params from node inputs[n_boundary_inputs:]
        aux from node saved[]
    disable recording;
    restore active_tape;

    validate recomputed output shape/dtype against original output;

    run gd_backward_tape() on recompute_tape seeded by a copy of outer grad;

    for each boundary input and explicit param:
        if recompute grad exists:
            accumulate-copy into outer tape for that original tensor id;

    release recompute tape grads/refs;
    end recompute scratch scope;
```

This is the main quality rule: **the recompute forward/backward must use a temporary tape, not append nodes to the outer tape.**

## Phase 4: GPT integration

Files:

- `examples/gpt_lm/gpt_lm_shared.h`
- `examples/gpt_lm/gpt_lm_shared.c`
- `examples/gpt_lm/main.c`
- `examples/gpt_lm/README.md`

Add config enum:

```c
typedef enum gpt_activation_checkpointing {
    GPT_ACT_CKPT_NONE = 0,
    GPT_ACT_CKPT_BLOCK = 1,
} gpt_activation_checkpointing;
```

Add field to `gpt_config` and `gpt_lm`:

```c
gpt_activation_checkpointing activation_checkpointing;
```

YAML key:

```yaml
activation_checkpointing: none   # none | block
```

### GPT checkpoint callback

Payload:

```c
typedef struct gpt_block_ckpt_payload {
    gpt_lm *model;
    gpt_block *block;
    uint32_t block_index;
    uint64_t step;
} gpt_block_ckpt_payload;
```

Callback:

```c
static gd_status gpt_block_ckpt_fn(gd_context *ctx,
                                   const void *payload,
                                   const gd_tensor *const *inputs,
                                   uint32_t n_inputs,
                                   const gd_tensor *const *aux,
                                   uint32_t n_aux,
                                   gd_tensor *outputs,
                                   uint32_t n_outputs)
{
    const gpt_block_ckpt_payload *p = (const gpt_block_ckpt_payload *)payload;
    if (p == NULL || n_inputs != 1U || n_aux != 2U || n_outputs != 1U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gpt_block_forward(ctx,
                             p->model,
                             p->block,
                             p->block_index,
                             inputs[0],
                             aux[0], /* positions */
                             aux[1], /* cu_seqlens */
                             p->step,
                             &outputs[0]);
}
```

Wrapper:

```c
static gd_status gpt_block_forward_checkpointed(...)
{
    const gd_tensor *inputs[1] = {x};
    const gd_tensor *aux[2] = {positions, cu_seqlens};
    const gd_tensor *params[10];
    uint32_t n_params = 0U;
    gpt_block_ckpt_payload payload;
    gd_checkpoint_options opts;

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

    payload.model = model;
    payload.block = block;
    payload.block_index = block_index;
    payload.step = step;

    memset(&opts, 0, sizeof(opts));
    opts.params = params;
    opts.n_params = n_params;
    opts.aux = aux;
    opts.n_aux = 2U;
    opts.payload = &payload;
    opts.payload_size = (uint32_t)sizeof(payload);
    opts.debug_name = "gpt_block";

    return gd_checkpoint_region(ctx,
                                gpt_block_ckpt_fn,
                                &opts,
                                inputs,
                                1U,
                                out,
                                1U);
}
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

Do not checkpoint the KV-cache generation/prefill paths.

## Phase 5: tests

Add `tests/test_checkpoint.c`.

Required tests:

1. Gradient equality:
   - simple `linear -> relu/swiglu -> linear -> reduce` region;
   - compare checkpointed vs non-checkpointed input and param grads.

2. Scratch memory reduction:
   - chain multiple checkpointed regions;
   - verify scratch watermark is materially lower, or use a small scratch slot where non-checkpointed fails but checkpointed succeeds.

3. Dropout determinism:
   - fixed seed;
   - checkpointed and non-checkpointed grads match within tolerance.

4. Borrowed input safety:
   - two checkpoint regions in sequence;
   - output of first region remains valid for second and for backward.

5. Error cleanup:
   - checkpoint callback returns an error;
   - autograd recording state, active tape, and scratch scope state are restored.

6. GPT smoke:
   - tiny GPT config;
   - one train step with `activation_checkpointing: block`.

Run normal tests and sanitizer tests if available:

```sh
make test
SAN=1 make test
```

## Work split for 5 maintainers

1. Memory owner:
   - scratch scope API;
   - allocator monotonic mode during scopes;
   - memory tests.

2. Autograd owner:
   - tape split;
   - active tape switching;
   - owned/borrowed refs;
   - nested backward helper.

3. Checkpoint owner:
   - public checkpoint API;
   - `GD_OP_CHECKPOINT` autograd rule;
   - error cleanup paths.

4. GPT owner:
   - config parsing;
   - block wrapper;
   - README/example config update;
   - GPT smoke test.

5. QA/docs owner:
   - gradient equality tests;
   - dropout determinism tests;
   - scratch watermark benchmark;
   - design docs and reviewer checklist.

## Review checklist

Before merging MVP:

- [ ] checkpointed and non-checkpointed gradients match on deterministic tests;
- [ ] recompute nodes are not appended to the outer tape;
- [ ] checkpoint payload is copied into tape attrs;
- [ ] scratch scope frees internal forward tensors;
- [ ] borrowed outer tensors are not freed by recompute tape cleanup;
- [ ] recording state is restored on all errors;
- [ ] GPT block checkpointing is disabled for eval/infer/KV-cache paths;
- [ ] dropout seeds are deterministic across forward and recompute;
- [ ] public API docs explain purity/determinism requirements.

## Non-goals / traps

Do not:

- just call `gd_set_grad_enabled(false)` in GPT and call it done;
- append recompute ops to the outer tape;
- store pointers to stack payloads in tape attrs;
- checkpoint mutating/stateful KV-cache paths;
- start with min-cut planning or selective policies.

MVP should be: **block-level GPT activation checkpointing backed by scratch scopes and a recompute sub-tape**.
