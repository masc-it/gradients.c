# gradients.c v2 design spec

## Goal

Build neural architectures from plain C with minimal framework surface.

Run on GPUs only.

Metal is first target and uses F16 storage.

CUDA is next target and uses BF16 storage.

Optimize ad-hoc by adding explicit fused ops and kernels.

Keep implementation easy for AI coding assistants and humans to extend.

Prefer predictable memory and simple control flow over general compiler features.

## Non-goals

No public graph API.

No general-purpose IR compiler.

No automatic fusion planner.

No hidden fallback inside kernels.

No CPU execution backend.

No CPU reference kernels.

No general public FP32 tensor kernel variants.

No quantization support.

No per-tensor heap allocation during training.

## Core model

Execution is eager and scoped.

Each op call is semantically executed immediately on the selected backend.

Backends may batch/enqueue eager ops inside the current scope and submit them to a command buffer/stream for performance, as long as program order and blocking-read semantics are preserved.

Training uses an internal autograd tape.

Tape is not a compiler IR.

Tape entries store op kind, inputs, outputs, attrs, saved tensors, and backward callback.

Backward walks tape in reverse and accumulates gradients into leaf grad buffers.

Model code is normal C code calling tensor ops.

## Memory model

Memory management is a core API contract, not an optimization detail.

v2 uses fixed-capacity arenas selected up front from model/batch config and prior watermarks. Arenas may be resized by rebuilding context/model state, but they do not grow with heap allocation inside train/eval/infer scopes.

There are two allocation planes:

- host metadata plane: C structs, module metadata, tensor descriptors, tape entries, names, manifests
- backend storage plane: tensor bytes in backend buffers, addressed by arena + byte offset

A tensor descriptor may live in one arena while its storage lives in another, but descriptor lifetime must not exceed storage lifetime unless it is only a persistent handle that is revalidated before use.

Use four logical arenas per context.

| Arena | Lifetime | Contents | Reset policy |
| --- | --- | --- | --- |
| `params` | model / optimizer lifetime | model structs, parameter descriptors/storage, persistent buffers, leaf grads, optimizer slots, scaler/scheduler state | never resets during training; sealed after init |
| `state` / `state_arena` | explicit runtime-state lifetime | KV caches, recurrent state, generation buffers, long-lived non-checkpoint workspaces | object-level reset/reuse only |
| `scratch` | one train/eval/infer scope | activations, temporary outputs, tape entries, saved tensors, workspaces, temporary grads | ring slot reset at `gd_begin` |
| `data` | one batch / input scope | decoded samples, token/image buffers, input tensor descriptors/storage | ring slot reset at `gd_begin` |

No tensor owns memory with `malloc`.

No op frees memory.

No op performs heap allocation in the step hot path. All per-op tensor descriptors, tape entries, temporaries, and workspaces come from `scratch` unless an API explicitly says otherwise.

Arena allocation returns an aligned span: backend buffer handle, actual byte offset after alignment, size, arena kind, slot index when ring-backed, and generation. Code must never infer a tensor storage offset from the pre-alignment bump pointer.

Each arena tracks capacity, current offset, high watermark, required alignment, and generation. Debug builds should poison or generation-check reset slots when possible.

Out-of-memory returns `GD_ERR_OUT_OF_MEMORY` and sets context error state. It must not partially publish a tensor descriptor. Recovery is scope abort / context reset / rebuild with larger arenas.

### Persistent memory

`params` allocation is allowed during model construction, parameter/buffer registration, optimizer construction, scaler/scheduler creation, and checkpoint load that rebuilds state.

After init, `params` is sealed. Training steps may mutate parameter, grad, and optimizer storage in place, but may not allocate new `params` memory.

Leaf grad buffers are persistent by default and live beside parameters/optimizer slots. `gd_optimizer_zero_grad` enqueues device fills/memsets and allocates nothing.

Optimizer state dtype and placement are explicit. Adam-family moments use FP32 storage/accumulation even when parameter/grad storage is F16/BF16.

### Runtime state memory

`state_arena` is for caller-visible runtime state, not temporary activations.

State objects are handles containing name, offset, size, dtype/layout metadata as needed, generation, and last-use backend fence.

When async work reads or mutates a state object, runtime records the command fence on that object.

Reset/reuse of an in-flight state object must either:

- wait for the object's last-use fence, then reuse the same block, or
- allocate a fresh state block and update the object handle without waiting.

Whole `state_arena` reset is only valid for explicit session/model/cache teardown after all object fences are complete. Runtime state is not saved in training checkpoints unless caller opts into a separate application-level state save.

### Scratch/data rings and scope lifecycle

`scratch` and `data` are ring-buffered on async GPU backends.

`gd_begin(ctx, mode)` selects completed `scratch` and `data` slots, resets their bump offsets, bumps slot generations, prepares tape when training, and opens an ordered backend scope.

If no slot is complete, `gd_begin` waits for the oldest in-flight slot only. It must not call global synchronize as normal lifecycle.

`gd_end(ctx)` closes tape, submits queued backend work as needed, and records the resulting fence on every ring slot and state object touched by the scope.

A tensor descriptor or view that references a ring slot is valid only until that slot is reset. Debug descriptors store arena kind, slot index, and generation so stale use can be caught when the descriptor is still reachable.

Ring defaults for the 256 hidden / 4 head / T=512 Metal profile are `scratch = 3 x 64 MiB` and `data = 3 x 8 MiB`. These are defaults, not ABI; watermarks drive tuning.

### Device memory

Arenas are device-aware.

Compute storage lives in backend compute buffers. Tensor storage is buffer + allocation byte offset + view byte offset, not a separate allocation.

Core code must not assume a compute tensor has a CPU-addressable pointer. Generic runtime code may inspect descriptors but may not dereference compute storage.

Metal tensor arenas use `MTLResourceStorageModeShared` buffers only. No Metal tensor arena uses `Private` storage or staging copies. Metal shared storage permits direct CPU writes/readbacks after relevant command-buffer waits; this is the Metal transfer implementation, not a portable core semantic.

CUDA tensor arenas use device allocations by default, via `cudaMalloc`, CUDA driver allocation, or a backend memory pool. CUDA hot tensors are not host-visible. CUDA host writes/readbacks use explicit upload/download operations through pinned staging memory or caller-owned host buffers on an ordered stream. `cudaMallocManaged` or mapped pinned memory may exist as opt-in debug/bringup modes, but they are not the v2 performance contract.

Future Vulkan/other backends follow the same logical arena contract: large backend buffers, tensor storage as arena plus byte offset, ring slots for `scratch` and `data`, and explicit backend fences/events. They do not need shared host/device memory.

Host-visible access is explicit:

- `gd_span_upload` / `gd_span_download` copy host bytes to/from an arena span byte range.
- `gd_upload` / `gd_download` copy full contiguous tensor payloads to/from host memory.
- blocking `gd_tensor_write` / `gd_tensor_read` are convenience wrappers over upload/download plus required waits.
- transfer helpers wait relevant arena-slot or conservative persistent-storage fences before accessing bytes.
- checkpoint save/load uses the same transfer path, not raw compute-tensor host pointers.

GPU `scratch` and `data` reuse is protected by ring-buffered arena slots and backend completion fences/events.

Backend owns pending command buffers/events until every arena slot and state object that references them has observed completion or released the fence.

Global `gd_synchronize` is for explicit user/debug/checkpoint boundaries, not normal arena lifecycle.

Small scalar kernel inputs such as cache start positions, sequence lengths in lockstep decode, dropout flags, and scaling constants are passed as by-value op attrs / uniforms where possible, not as mutable host-written GPU tensors.

## Tensor model

`gd_tensor` is always a concrete tensor or view.

There are no virtual graph tensors.

Tensor descriptor stores dtype, device, rank, shape, strides, layout, arena/storage reference, allocation byte span, view byte offset, and debug lifetime generation.

Views allocate only a tensor header. They share the base allocation span and change view offset/shape/strides only.

`reshape`, `slice`, and `transpose` are metadata ops when legal. They must prove the resulting logical byte span stays inside the base allocation.

`contiguous` is explicit and allocates a packed output.

Foundation C descriptors are caller/storage-owned structs: `gd_tensor_empty` allocates arena storage and fills a concrete descriptor; `gd_tensor_slice` produces a view descriptor over the same base allocation; `gd_tensor_validate` checks arena kind, slot, generation, buffer identity, and logical byte span. The foundation `gd_tensor_contiguous` path allocates packed output storage/metadata only; backend copy/materialization kernels are separate op work and no core path performs an implicit contiguous copy.

Tensor creation helpers are explicit about initialization:

- `gd_tensor_empty(...)` allocates storage and does not promise contents.
- `gd_tensor_zeros(...)` allocates then enqueues a backend zero-fill.
- `gd_tensor_ones(...)` allocates then enqueues a dtype-aware backend fill-one kernel.
- `gd_tensor_rand_uniform(...)` / `gd_tensor_rand_normal(...)` allocate then enqueue backend RNG kernels using explicit seed/state. They must not use C `rand()` or CPU-filled tensor bytes for normal training paths.
- in-place forms (`gd_tensor_zero_`, `gd_tensor_fill_`, `gd_tensor_rand_uniform_`, `gd_tensor_rand_normal_`) are ordered backend ops, allocate no tensor storage, and obey arena generation/fence rules.

Leaf tensors may require gradients.

Leaf grad buffers live in persistent memory, `params` by default, with parameter/optimizer lifetime.

Temporary grads live in `scratch` during backward.

Leaf gradient accumulation is additive (`+=`), never overwrite.

Ops with indexed backward paths, such as embedding, must handle duplicate indices with correct GPU scatter-add/reduce behavior.

`slice` may produce non-contiguous views, e.g. VLM text suffix `hidden[:, img_tokens:, :]` from `[B, img_tokens + text_tokens, D]`.

Core consumers must either support strided views directly or provide an explicit fused variant that consumes base tensor plus slice metadata. Hot paths must not require an implicit `contiguous` copy.

`concat` produces a compact output when it joins distinct tensors. Its backward rule splits incoming gradients by view metadata and accumulates into each source; it must handle non-contiguous gradient views correctly.

## Op model

Ops live in per-op capsules under `src/ops/<op>/`.

Keep generated registries for op discovery and coverage.

Each op capsule owns:

- public wrapper
- shape/dtype/device validation
- by-value attrs / command-uniform schema
- GPU backend kernels
- tape rule or custom backward op
- PyTorch reference script for forward/backward correctness
- tests

Capsule layout:

```text
src/ops/<op>/
  core_<op>.c          public forward wrapper, validation, output allocation, backend dispatch
  core_<op>_bwd.c      public backward wrapper or explicit `GD_ERR_NOT_IMPLEMENTED` stub
  metal_<op>.m         Metal/MPS encoder implementation
  metal_<op>.metal     custom kernels when needed
  fwd.py               PyTorch/numeric forward oracle
  bwd.py               PyTorch/numeric backward oracle
```

`fwd.py` and `bwd.py` use PEP 723 script metadata with dependencies and are run through `uv run path/to/script.py`. They are reference/oracle programs for numerical correctness, not runtime dependencies.

Forward wrappers allocate outputs from `scratch` by default and publish the output descriptor only after backend enqueue succeeds. Backward wrappers may be present before implementation and must return `GD_ERR_NOT_IMPLEMENTED` without publishing grad descriptors.

Ops with persistent outputs use explicit destination APIs.

Example:

```c
gd_add(ctx, a, b, &out);       // allocates out from scratch
gd_add_to(ctx, a, b, out);     // writes caller-owned out
```

Stateful destination ops are explicit ordered side effects.

Example:

```c
gd_kv_cache_write_at(ctx, k_cache, v_cache, start_pos, k_new, v_new);
```

`start_pos` is a by-value attr for lockstep decode/prefill, not a mutable cache-position tensor read by kernels.

For uneven batched generation, cache ops support per-row length/position tensors plus active masks.

Ad-hoc optimization means adding explicit fused ops.

Examples: `rms_norm_qkv`, `lm_cross_entropy`, `lm_cross_entropy_suffix`, `loss_sum`, `sdpa_varlen`, `sdpa_cached_prefill`, `sdpa_cached_decode`, `gather_positions`, `pool_mean_masked`, `contrastive_loss`, architecture-specific blocks.

No optimizer rewrites unfused ops into fused ops automatically.

Device-side sampling ops are part of the op set for production inference: `argmax`, `topk`, `topk_sample`, RNG state update, and optional EOS/active-mask update.

CPU sampling/readback remains a debug path.

## Attention and LM head hot paths

Decoder-only text and early-fusion VLM use three attention execution modes.

Training/full-sequence mode uses packed variable-length attention:

```c
gd_sdpa_varlen(ctx, q, k, v, cu_seqlens,
    &(gd_sdpa_varlen_config){
        .causal = true,
        .prefix_len = img_tokens,
        .sliding_window = text_window,
        .max_seqlen = bucket_seq,
    },
    &out);
```

For text-only models, `prefix_len=0` gives normal causal/window attention. For early-fusion VLM, prefix tokens are image tokens. Prefix queries attend bidirectionally within prefix only. Text/suffix queries attend causally to all prefix keys plus prior/current suffix keys. Sliding window applies only to suffix/suffix attention; prefix keys remain globally visible to all suffix queries.

Metal hot path must support the optimized packed VLM case: F16, `Dh=64`, `causal=true`, `prefix_len>0`, `sliding_window>0`, grouped-query heads, forward and backward kernels.

Prefill mode computes K/V for a chunk and fills or appends the KV cache. Fresh prefill may use the same packed prefix-window attention for the chunk, then write K/V to `state_arena`. Appending prefill to an existing cache uses an explicit cached-prefill op:

```c
gd_kv_cache_write_at(ctx, k_cache, v_cache, start_pos, k_new, v_new);
gd_sdpa_cached_prefill(ctx, q_new, k_cache, v_cache,
                       start_pos, chunk_len, prefix_len, text_window, &out);
```

Decode mode uses a dedicated cached decode kernel, not full `sdpa_varlen`:

```c
gd_kv_cache_write_at(ctx, k_cache, v_cache, start_pos, k_new, v_new);
gd_sdpa_cached_decode(ctx, q, k_cache, v_cache,
                      start_pos, prefix_len, text_window, &out);
```

`start_pos` and lockstep lengths are by-value attrs/uniforms. Uneven batched generation uses device tensors `lengths[B]` and `active[B]`. No kernel reads a mutable host-written scalar cache position.

LM head loss must handle early-fusion suffixes efficiently. The training hot path is:

```c
gd_lm_cross_entropy_suffix(ctx, hidden, weight, targets,
                           suffix_start = img_tokens,
                           suffix_len = text_tokens,
                           &loss);
```

This op consumes base hidden tensor plus suffix metadata or an equivalent non-contiguous suffix view. It must not materialize full logits `[B, img_tokens + text_tokens, vocab]`, and must not require copying `hidden[:, img_tokens:, :]` into a compact tensor. Backward writes gradients only for suffix hidden rows and leaves prefix hidden-gradient contribution from LMCE as zero.

## Autograd

Autograd is eager tape-based reverse mode.

Tape entries live in `scratch`.

Tape records only when grad mode is enabled and an input requires grad.

Saved tensors live in `scratch` unless they are persistent inputs.

Backward callbacks may allocate temporary tensors from `scratch`.

Leaf gradients accumulate in persistent grad buffers.

`no_grad` disables tape recording for eval, inference, optimizer internals, state updates, and sampling.

## Backend model

Backends implement eager GPU op dispatch.

Backend contract:

- validate support before execution
- run/enqueue op or return exact `GD_ERR_UNSUPPORTED`
- preserve program order within a scope
- preserve ordered side effects to destination/state tensors
- never silently fallback inside backend code
- expose synchronization for async devices

There is no CPU execution path.

If target backend lacks an op, execution fails.

Blocking CPU reads, blocking CPU writes, upload/download transfers, checkpoint writes, and explicit `gd_synchronize` wait for relevant queued backend work and make requested bytes visible.

Correctness is checked against PyTorch reference scripts, not CPU kernels.

## Backend portability abstractions

Core v2 depends on a small backend portability layer before CUDA is added.

Required abstractions:

- `gd_backend_stream`: ordered scope execution lane. Metal maps this to a command buffer/queue. CUDA maps this to a `cudaStream_t` plus recorded events.
- `gd_fence`: completion token with query/wait/error behavior. Metal maps this to a retained `id<MTLCommandBuffer>`. CUDA maps this to `cudaEvent_t`.
- `gd_buffer`: backend allocation handle plus size, memory kind, device pointer, optional host pointer, and alignment. Metal shared buffers expose host pointers. CUDA device buffers normally do not.
- `gd_arena` / `gd_ring_arena`: bump allocation over `gd_buffer`, aligned span returns, capacity/offset/watermark tracking, freeze/seal state, generations, ring-slot selection, and slot-fence retention.
- `gd_transfer`: upload/download/read/write path that hides Metal direct shared access versus CUDA staged copies.
- `gd_kernel`: backend kernel/module/pipeline handle plus argument packing, by-value uniforms, grid/block or threadgroup launch dimensions, and optional dynamic shared/threadgroup memory.
- `gd_matmul`: backend matmul descriptor for dtype, accumulation dtype, transpose/layout, row strides, batch strides, pointer offsets/origins, and epilogue support. Metal maps first to MPS matrix multiplication. Linear-style epilogues are encoded as a custom Metal epilogue kernel in the same command buffer after MPS GEMM, not as a naive single-kernel matmul for transformer-sized shapes. CUDA maps to cuBLAS/cuBLASLt or custom kernels.

No generic runtime code may dereference compute tensor storage directly. Only backend implementations and transfer helpers may turn a `gd_buffer` plus offset into an address.

## API shape

Context owns arenas, backend state, current scope, tape, grad mode, RNG, and error state.

Model/optimizer/scaler/scheduler constructors allocate persistent metadata/storage from `params` through context before `params` is sealed.

Forward ops take context and tensors.

Use scopes for train/eval/infer lifecycle.

Training loop shape:

```c
gd_begin(ctx, GD_SCOPE_TRAIN);
load_batch(ctx, &batch);
gd_optimizer_zero_grad(ctx, opt);
model_forward(ctx, model, batch.x, &loss);
gd_amp_backward(ctx, scaler, loss);
gd_optimizer_step_amp(ctx, opt, scaler);
gd_end(ctx);
```

Eval/inference loop shape:

```c
gd_begin(ctx, GD_SCOPE_INFER);
model_prefill_or_decode(ctx, model, state, input_tokens, &logits);
gd_topk_sample(ctx, logits, rng_state, next_tokens);
gd_end(ctx);
```

`gd_begin(ctx, mode)` advances `scratch`/`data` ring slots to completed generations, prepares tape when training, and opens a backend command batch/stream scope. If the ring is exhausted, it waits for the oldest slot only.

`gd_end(ctx)` closes the tape when training, submits queued backend work as needed, and records backend completion fences for arena slots and state objects used by the scope. It does not reset `params` or `state`.

`gd_optimizer_zero_grad(ctx, opt)` is explicit so callers can choose gradient accumulation by omitting it.

`gd_optimizer_zero_grad(ctx, opt)` enqueues device fills/memsets for leaf grad buffers and allocates nothing.

`gd_optimizer_step_amp(ctx, opt, scaler)` performs unscale, found-inf check, skip/update decision, scaler update, and optimizer math on device. Host readback of loss/found-inf/scale is optional logging, not step control.

Multi-head / multi-loss training sums named device losses and runs one backward pass:

```c
gd_loss_term terms[] = {
    {.name = "lm",   .loss = lm_loss,   .weight = 1.0f},
    {.name = "repr", .loss = repr_loss, .weight = 0.1f},
};
gd_loss_sum(ctx, terms, 2, &total_loss);
gd_amp_backward(ctx, scaler, total_loss);
```

Calling backward separately for each head is not the default path; shared-backbone gradients should accumulate from one total scalar loss.

## State save/load

v2 needs first-class save/load for resumable training, not model weights only.

Checkpoint categories:

- model state: parameters, persistent buffers, module tree names, tensor shapes/dtypes, tied-weight metadata
- optimizer state: param groups, per-param slots such as moments, optimizer step counters, optional master/accumulator tensors when explicitly supported
- AMP scaler state: current scale, growth tracker, config; transient found-inf flags reset on load
- LR scheduler state: scheduler kind/config, current step, warmup/decay counters, last LR values if stateful

Training checkpoints save all categories together by default:

```c
gd_checkpoint_save(ctx, "train.gdc", &(gd_checkpoint_state){
    .model = &model->mod,
    .optimizer = opt,
    .scaler = scaler,
    .lr_scheduler = sched,
});
```

Resume loads in the same order after constructing the same model/optimizer/scheduler objects:

```c
gd_checkpoint_load(ctx, "train.gdc", &(gd_checkpoint_state){
    .model = &model->mod,
    .optimizer = opt,
    .scaler = scaler,
    .lr_scheduler = sched,
}, GD_LOAD_STRICT);
```

Model-only load is supported for fine-tune/export. Strict model load requires exact key/shape/dtype match. Partial model load may ignore missing/extra named heads when requested.

Optimizer load is strict by default because slots are keyed by deduplicated parameter identity/path and param-group layout. Partial optimizer load is allowed only with an explicit flag; missing slots initialize as fresh optimizer state.

Scaler and LR scheduler load must be tied to resume semantics. Fine-tune from pretrained weights should usually reset scaler and scheduler unless caller explicitly loads them.

Runtime `state_arena` objects such as KV caches are not saved in training checkpoints by default. They may opt into separate application-level serialization, but model/optimizer/scaler/scheduler resume must not depend on cache contents.

## Modules and models

Modules are plain C structs with an embedded registration object. The registration object is for traversal and metadata, not a mandatory virtual base for forward.

Each model owns parameters allocated in `params`.

Forward functions are normal C functions with model-specific signatures.

No framework inheritance required.

`gd_module` supports:

- named parameter registration
- named buffer registration
- named child module registration
- recursive train/eval mode
- recursive freeze/unfreeze
- checkpoint save/load
- optimizer parameter collection
- tied-weight deduplication by tensor identity

Buffers are persistent checkpointed tensors that are not trainable parameters. Runtime state, such as KV cache, lives in `state` / `state_arena` and is not checkpointed by default.

Core containers:

- `nn.Module`: base registration/traversal object embedded in every module struct
- `nn.ModuleList`: ordered child list without a forward function; primary container for transformer blocks
- `nn.Sequential`: `ModuleList` plus unary tensor forward callbacks for simple feed-forward stacks
- `nn.ModuleDict`: named child dictionary; primary container for multi-head models and optional adapters

Containers organize the model tree. Typed C forward functions do computation.

`nn.Module` is not a virtual class, not a tensor-memory owner, and not required to expose a generic `forward` callback. It stores names and traversal metadata only.

Example module API shape:

```c
gd_module_init(ctx, &m->mod, "linear");
gd_module_add_param(&m->mod, "weight", m->weight);
gd_module_add_param(&m->mod, "bias", m->bias);
gd_module_add_buffer(&m->mod, "running_mean", buf);
gd_module_add_child(&parent->mod, "proj", &m->mod);

gd_module_set_training(&model->mod, true);
gd_module_freeze(&model->mod, "backbone.*");
gd_module_collect_params(ctx, &model->mod, groups, n_groups, &param_set);
```

`nn.ModuleList` stores ordered child registrations, but typed arrays hold concrete structs for fast C access:

```c
gd_module_list_init(ctx, &m->blocks, "blocks", n_layers);
for (int i = 0; i < n_layers; ++i) {
    gpt_block_init(ctx, &m->block[i], i);
    gd_module_list_set(&m->blocks, i, &m->block[i].mod);
}
gd_module_add_child(&m->mod, "blocks", &m->blocks.mod);
```

Forward over a `ModuleList` remains typed:

```c
for (int i = 0; i < m->blocks.n; ++i) {
    gpt_block_forward(ctx, &m->block[i], x, pos, cache, &x);
}
```

Checkpoint paths for `ModuleList` use numeric child names such as `model.blocks.0.attn.wq.weight`.

Example transformer shape:

```c
typedef struct {
    gd_module mod;
    gd_embedding tok_emb;
    gd_module_list blocks;
    gpt_block *block;
    gd_rms_norm ln_f;
} gpt_model;
```

Example multi-head VLM shape:

```c
typedef struct {
    gd_module mod;
    vlm_backbone backbone;
    gd_module_dict heads; /* "lm", "repr", ... */
    vlm_lm_head lm;
    vlm_repr_head repr;
} vlm_model;
```

Head selection is explicit so training/eval can skip inactive heads and avoid unused tape/scratch allocation:

```c
typedef enum {
    VLM_HEAD_LM   = 1u << 0,
    VLM_HEAD_REPR = 1u << 1,
} vlm_head_mask;
```

Multi-head forward functions return plain C output structs, not a generic dynamic dict:

```c
typedef struct {
    gd_tensor *hidden;
    gd_tensor *lm_loss;
    gd_tensor *repr_loss;
    gd_tensor *total_loss;
    gd_tensor *repr;
    gd_tensor *logits;
} vlm_output;
```

Optimizer parameter collection supports named groups:

```c
gd_param_group groups[] = {
    {.name = "backbone", .match = "vlm.backbone.*",   .lr_mult = 0.1f},
    {.name = "lm",       .match = "vlm.heads.lm.*",   .lr_mult = 1.0f},
    {.name = "repr",     .match = "vlm.heads.repr.*", .lr_mult = 1.0f},
};
```

Parameter groups support per-group learning-rate multipliers, weight decay, trainable flags, and freeze/unfreeze. Collection must deduplicate tied tensors, e.g. GPT token embedding tied to LM head weight.

Tied-weight param-group policy is canonical-owner wins. The first registered path for a tensor defines its optimizer group and slot identity; later paths are aliases recorded for checkpoint/debug metadata. If an alias path matches a different param group, collection reports it as an alias/group conflict in debug/strict mode, but it must not create a second optimizer entry.

Checkpoint loading supports strict and partial modes. Partial mode allows adding/removing named heads while loading a shared backbone.

Decoder-only transformer KV cache design:

- cache tensors: `k[layer]` and `v[layer]` in `state_arena`
- lockstep prefill/decode uses host metadata `cache->len` and passes `start_pos` by value to cache/attention ops
- uneven batched generation uses device tensors `lengths[B]` and `active[B]`
- cache reset records/waits on state-object fences before reuse
- cache state is not checkpointed by default

Early-fusion VLM remains decoder-only:

- image patches become prefix embeddings `[B, img_tokens, D]`
- text tokens become suffix embeddings `[B, text_tokens, D]`
- model concatenates prefix+suffix before decoder blocks
- training attention uses `sdpa_varlen` with `prefix_len=img_tokens` and suffix-only sliding window
- LMCE consumes only suffix hidden rows via `lm_cross_entropy_suffix`
- representation heads consume selected/pool hidden states, e.g. EOS/CLS gather or masked mean pool
- inference prefill writes image+prompt K/V into cache; decode appends text K/V only
- decode attention keeps `prefix_len=img_tokens` so image prefix remains globally visible while generated text uses causal/window rules

## Data path

Dataloader writes batch tensors into `data`.

Data tensors survive through forward and backward for that batch.

On async GPUs, `data` is ring-buffered like `scratch`; next batch writes into a completed slot, never memory still referenced by in-flight work.

Variable-length batches use explicit packed tensors and metadata tensors.

Padding policy is model or dataloader choice, not core runtime magic.

Generation token ping-pong buffers may live in `state_arena` or `data` ring slots. Production generation should write next tokens device-side to avoid per-token CPU synchronization.

## Precision

Supported training storage dtypes are GPU-first low precision.

Metal uses F16 tensor and leaf-grad storage with AMP.

CUDA uses BF16 tensor and leaf-grad storage; AMP machinery remains available for shared training control.

Kernels may use FP32 accumulators internally where numerically required: matmul accumulation, reductions, norms, softmax/attention, gradient norm, loss, and optimizer math.

Optimizer state dtype is explicit per optimizer/backend. Adam-family moments use FP32 storage/accumulation and should be backed by explicit GPU optimizer kernels.

FP32 master weights are out of v2 scope unless backed by explicit GPU kernels.

Lower-precision formats may be added later only with explicit GPU kernels.

No general public FP32 tensor/kernel variants are part of v2.

FP32 may appear in kernel accumulators, optimizer/scaler state, test references, or host-side scalar configuration.

There is no quantization API.

## Error handling

Public functions return `gd_status`.

No public API aborts on normal errors.

Error messages must include op name and reason.

Assertions are for internal invariants only.

## Testing

Every op needs a PyTorch reference script.

Every op test compares GPU forward numbers against PyTorch.

Every differentiable op test compares GPU backward gradients against PyTorch.

Every fused op compares against PyTorch reference composition.

Memory tests verify no heap allocation in step hot path.

Arena tests verify aligned span offsets, OOM status returns, watermark tracking, reset generations, and sealed `params` rejecting late allocations.

Arena watermark tests report peak `params`, `state`, `scratch`, and `data` usage.

Ring-buffer tests verify `scratch` and `data` are not reused before backend fences complete and slot generations bump before reuse.

State reset tests verify KV cache/state objects wait or allocate fresh blocks before reuse.

Backend portability tests verify:

- generic runtime code never dereferences compute tensor storage directly
- Metal shared buffers satisfy upload/download/read/write semantics without staging copies
- CUDA uses explicit upload/download staging for host access and never requires host-visible hot tensors
- ring slots are guarded by backend fences: Metal command buffers and CUDA events
- state-object reset waits on backend fences before reuse
- matmul abstraction covers byte offsets, padded row strides, strided batches, sliced/origin operands, and same-stream GEMM-to-kernel ordering

Checkpoint tests verify:

- model-only strict load roundtrip
- model partial load with added/removed heads
- full training resume roundtrip: model + optimizer + scaler + LR scheduler
- optimizer tied-weight dedup and param-group slot mapping
- fine-tune path resets optimizer/scaler/scheduler unless explicitly loaded

Decoder-only GPT stress tests cover:

- teacher-forced training with tied embedding/lm head
- embedding backward duplicate-token scatter-add
- AMP optimizer step without host found-inf control dependency
- prefill and decode KV cache writes using by-value `start_pos`
- uneven batched generation with `lengths[B]` and `active[B]`
- device-side argmax/top-k sampling without per-token CPU readback

Early-fusion VLM stress tests cover:

- image-prefix/text-suffix concat before decoder blocks
- packed `sdpa_varlen` prefix-window fast path: F16, `Dh=64`, causal, bidirectional prefix, suffix-only sliding window
- backward through optimized `sdpa_varlen` prefix-window kernels
- suffix LMCE without full logits materialization and without suffix contiguous copy
- `slice` views and `concat` backward with non-contiguous gradients
- KV-cache prefill for image+prompt and decode for text-only continuation

Multi-head model stress tests cover:

- `ModuleDict` head registration and checkpoint naming
- LM head + representation head sharing one decoder backbone
- active head masks that skip inactive heads
- `gd_loss_sum` followed by one backward pass
- optimizer parameter groups for backbone vs heads
- tied-weight dedup with LM head / token embedding
- strict vs partial checkpoint load when heads are added or removed

## Migration direction

Remove public graph/runner APIs from v2 core.

Remove virtual tensor state.

Replace storage refcount churn with arena offsets, explicit lifetimes, aligned spans, and generation/fence validation.

Keep op capsules and generated registries.

Start with scoped eager Metal F16 dispatch.

Add PyTorch reference scripts alongside each op.

Build the decoder-only text GPT first: training, prefill, decode, KV cache, AMP, and device-side sampling.

Build early-fusion decoder-only VLM next: image-prefix concat, optimized varlen prefix-window training attention, suffix LMCE, cached prefill/decode.

Extract and lock the backend portability layer: `gd_buffer`, `gd_fence`, `gd_backend_stream`, `gd_transfer`, `gd_kernel`, `gd_matmul`, and arena/ring abstractions.

Add CUDA BF16 eager dispatch after Metal path and portability layer are stable.

Then rebuild GPT/VLM examples as plain C models on eager ops.
