# gradients.c — GPT KV Cache Design

Status: planned v0.2 (reviewed; sliding-window decode included in v1)

Goal: add production inference KV cache for decoder-only GPT while reusing the
existing SDPA implementation family. Training stays unchanged. Prefill uses the
current full-sequence SDPA path. Decode adds a cache-backed SDPA entrypoint, but
shares the same math, layout, GQA mapping, online-softmax logic, and Metal
plumbing style.

Non-goal for this plan: speculative decoding. The cache design will not block it
later, but v1 does not implement draft/verify/rollback.

References:
- [`docs/plan_gpt_preparation.md`](plan_gpt_preparation.md) §12 for KV-cache
  requirement.
- [`docs/plan_gpt.md`](plan_gpt.md) §6/G3 for current SDPA kernels and §7/G6
  for original KV-cache sketch.
- [`docs/plan_prefix_sdpa.md`](plan_prefix_sdpa.md) for recent SDPA mask work.

---

## 1. Current SDPA inventory

Already available:

- `gd_sdpa(q,k,v,bias,config)` public op.
- Head-major layout: `q[B,Tq,Hq,Dh]`, `k/v[B,Tk,Hkv,Dh]`.
- GQA: `hkv = hq / (Hq / Hkv)`.
- Stable softmax and causal/prefix/window masking.
- Optional additive bias, broadcast over `[B,Hq,Tq,Tk]`.
- CPU_REF oracle in `src/ops/sdpa/kernel_sdpa.c`.
- Metal kernels in `src/ops/sdpa/metal_sdpa_fwd.metal` and bwd files:
  tiled forward, causal specializations, split-K, and backward.
- Existing GPT forward in `src/nn/nn.c` already uses:

```text
rms_norm -> q/k/v projections -> reshape [B,T,H,Dh]
         -> RoPE(q,k) -> gd_sdpa(causal) -> out projection
```

KV cache must plug into this exact seam after RoPE(k) and before attention.

---

## 2. Design decisions

### 2.1 Cache ownership outside graph

Host/runtime owns cache allocation and sequence state:

```text
gd_kv_cache
  k_cache[layer]
  v_cache[layer]
  base_len_tensor[B]
  append_len_tensor[B]
  src_offset_tensor[B]
  committed_len[B]
  last_append_len[B]
  capacity
  batch_size
  dtype
```

Graph does not allocate pages, resize buffers, or own serving policy. Cache owns
persistent length tensors on the cache device. Host `committed_len` is source of
truth; `set_lengths` mirrors host lengths into those device tensors before graph
run.

### 2.2 Cache reads/writes inside graph

Graph must perform device writes and reads because `k_new/v_new` are intermediate
layer tensors. Pulling them out to host would split every layer and force syncs.

Decode graph shape:

```text
for layer l:
  q,k,v = projections(...)
  k = RoPE(k)
  kv_append(cache[l], k, v, base_len, append_len)
  o = sdpa_decode(q, cache[l], base_len, append_len)
```

One graph run per token/block. No per-layer host orchestration.

### 2.3 Flat right-sized cache v1

V1 uses flat cache tensors:

```text
k_cache[l] [B,capacity,Hkv,Dh]
v_cache[l] [B,capacity,Hkv,Dh]
```

`capacity` is not always model `max_seq_len`. Allocate per generation batch:

```text
capacity = min(model_max_seq_len, max_prompt_len + max_new_tokens)
```

Optionally bucket capacities (`64,128,256,512`) for graph/cache reuse.

Paged/vLLM-style cache is deferred. API stays opaque so storage can become paged
later without changing GPT/generation API.

### 2.4 Batched + ragged decode from start

Decode supports per-row lengths:

```text
base_len[B]     # committed cache length before graph run
append_len[B]   # valid new tokens per row, 0..Tnew
src_offset[B]   # source column in k_new/v_new; 0 for decode, T-len for left-padded prefill
```

Same-length batch remains fastest. Ragged decode avoids padded cache work because
kernels loop only over each row's live length.

`gd_kv_cache_set_lengths` validates host arrays, stores `last_append_len`, and
copies `base_len/append_len/src_offset` into cache-owned device tensors. Prefill
and decode graphs import those tensors as external values, so one compiled graph
can be reused while host updates lengths between runs.

### 2.5 Block decode / future MTP support

Do not bake `Tnew=1` into the op contracts.

```text
tokens      [B,Tnew]
q           [B,Tnew,Hq,Dh]
k_new/v_new [B,Tnew,Hkv,Dh]
```

Normal autoregressive decode uses `Tnew=1`. Future MTP/block acceptance can use
`Tnew=4` or another small block.

Causal-in-block semantics:

```text
query t attends old cache tokens 0..base_len[b)-1
             plus new block tokens 0..t
```

For invalid padded query positions (`t >= append_len[b]`), output is zero/ignored.

---

## 3. Public API sketch

### 3.1 Cache API

```c
typedef struct gd_kv_cache gd_kv_cache;

typedef struct gd_kv_cache_config {
    int batch_size;
    int capacity;       /* token capacity per row */
    gd_device device;
    gd_dtype dtype;     /* 0/GD_DTYPE_INVALID => model activation dtype */
} gd_kv_cache_config;

gd_status gd_kv_cache_create(gd_context *ctx,
                             const gd_gpt_config *gpt_cfg,
                             const gd_kv_cache_config *cache_cfg,
                             gd_kv_cache **out);
void gd_kv_cache_destroy(gd_kv_cache *cache);

gd_status gd_kv_cache_reset(gd_kv_cache *cache);

gd_status gd_kv_cache_set_lengths(gd_kv_cache *cache,
                                  const int32_t *base_len,
                                  const int32_t *append_len,
                                  const int32_t *src_offset,
                                  int n);

gd_status gd_kv_cache_set_prefill_lengths(gd_kv_cache *cache,
                                          const int32_t *prompt_lens,
                                          int tmax,
                                          bool left_padded,
                                          int n);

gd_status gd_kv_cache_commit(gd_kv_cache *cache,
                             const int32_t *accepted_len); /* NULL => append_len */
```

`set_lengths` requires `n == batch_size`; `append_len` is required,
`base_len == NULL` means current `committed_len`, and `src_offset == NULL` means
zero. `set_prefill_lengths` is convenience for ragged prompts: it writes
`base_len=0`, `append_len=prompt_lens`, and
`src_offset=(left_padded ? tmax-prompt_lens : 0)`.

`commit` is called only after successful graph execution. `accepted_len == NULL`
commits every appended token. Non-NULL `accepted_len[b]` must satisfy
`0 <= accepted_len[b] <= last_append_len[b]`; host updates
`committed_len[b] += accepted_len[b]`. This supports future partial block accept
without GPU rollback: uncommitted cache rows are ignored and overwritten by the
next append. `commit` does not read GPU data.

Cache dtype is fixed at creation and must match `k_new/v_new` dtype at append;
there is no implicit cast in v1. `GD_DTYPE_INVALID` resolves to the GPT activation
path dtype (`F32` for normal models, `F16` for forward-only F16 models).

### 3.2 GPT inference APIs

```c
gd_status gd_gpt_prefill(gd_context *ctx,
                         gd_gpt *gpt,
                         gd_kv_cache *cache,
                         gd_tensor *tokens,       /* int32 [B,T] */
                         gd_tensor *positions,    /* int32 [B,T] */
                         gd_tensor *prefill_bias, /* nullable additive SDPA bias */
                         gd_tensor **logits_out);

gd_status gd_gpt_decode_block(gd_context *ctx,
                              gd_gpt *gpt,
                              gd_kv_cache *cache,
                              gd_tensor *tokens,      /* int32 [B,Tnew] */
                              gd_tensor *positions,   /* int32 [B,Tnew] */
                              gd_tensor **logits_out);
```

`gd_gpt_prefill` and `gd_gpt_decode_block` use the cache's
`base_len/append_len/src_offset` tensors set before recording/running the graph.
`prefill_bias` is nullable and broadcastable to `[B,Hq,T,T]`; generation builds
it from the same host prompt lengths used for `gd_kv_cache_set_prefill_lengths`.

Generation v1 calls `gd_gpt_decode_block` with `Tnew=1`.

---

## 4. Internal ops

### 4.1 KV append

```c
gd_status gd_kv_cache_append(gd_context *ctx,
                             gd_tensor *k_cache,    /* [B,capacity,Hkv,Dh] */
                             gd_tensor *v_cache,    /* [B,capacity,Hkv,Dh] */
                             gd_tensor *k_new,      /* [B,Tnew,Hkv,Dh] */
                             gd_tensor *v_new,      /* [B,Tnew,Hkv,Dh] */
                             gd_tensor *base_len,   /* int32 [B] */
                             gd_tensor *append_len, /* int32 [B] */
                             gd_tensor *src_offset);/* nullable int32 [B], NULL => 0 */
```

Semantics:

```text
for b in B:
  for t in 0..append_len[b)-1:
    src = (src_offset ? src_offset[b] : 0) + t
    dst = base_len[b] + t
    k_cache[b,dst,:,:] = k_new[b,src,:,:]
    v_cache[b,dst,:,:] = v_new[b,src,:,:]
```

Validation:

```text
0 <= append_len[b] <= Tnew
0 <= src_offset[b]
src_offset[b] + append_len[b] <= Tnew
0 <= base_len[b]
base_len[b] + append_len[b] <= capacity
```

Op is inference-only, mutating, no backward. `_GD_OP_KV_APPEND` has no outputs
and is flagged `GD_OPF_MUTATES | GD_OPF_SIDE_EFFECT`. `k_cache/v_cache` must be
materialized external contiguous tensors on the same device/dtype as
`k_new/v_new`; they must not require grad and must not alias `k_new/v_new`.
Backend execution must preserve program order so a following `sdpa_decode` reads
newly written rows.

### 4.2 SDPA decode

```c
gd_status gd_sdpa_decode(gd_context *ctx,
                         gd_tensor *q,          /* [B,Tnew,Hq,Dh] */
                         gd_tensor *k_cache,    /* [B,capacity,Hkv,Dh] */
                         gd_tensor *v_cache,    /* [B,capacity,Hkv,Dh] */
                         gd_tensor *base_len,   /* int32 [B] */
                         gd_tensor *append_len, /* int32 [B] */
                         const gd_sdpa_config *config,
                         gd_tensor **out);      /* [B,Tnew,Hq,Dh] */
```

`config` reuses SDPA fields where meaningful:

- `scale`: same as `gd_sdpa`.
- `causal`: must be true for decode/block decode.
- `sliding_window`: supported in v1; `0` means full causal cache.
- `prefix_len`: not used in decode v1; reject `prefix_len > 0`.
- `bias`: omitted in v1 decode path.

Per query row:

```text
valid_q = t < append_len[b]
qpos = base_len[b] + t
key_end = qpos + 1
key_start = (sliding_window > 0) ? max(0, key_end - sliding_window) : 0
if !valid_q: out[b,t,h,:] = 0
else attend j in [key_start, key_end)
```

This assumes `kv_append` ran earlier in the same graph, so newly appended rows
`base_len[b]..base_len[b]+t` are already visible.

---

## 5. SDPA reusability plan

### 5.1 Keep full SDPA as prefill/training path

Do not route prompt prefill through decode kernels by default.

```text
prefill attention: gd_sdpa(q,k,v, causal=true)
decode attention : gd_sdpa_decode(q,k_cache,v_cache, base_len, append_len)
```

Why:

- Existing full SDPA kernels are optimized for `Tq ~= Tk` and training.
- Decode has different input shape and live-length bounds.
- Forcing one megakernel would add branches to the hot training path.

### 5.2 Add separate entrypoint, not separate math

`_GD_OP_SDPA_DECODE` is a new op because arity/shape/autograd differ. Its
implementation must reuse the SDPA implementation family:

Shared with existing SDPA:

- head-major `[B,T,H,Dh]` layout
- `Hq/Hkv` grouped-query mapping
- scale resolution (`0 => 1/sqrt(Dh)`)
- stable online softmax update
- output accumulation
- CPU_REF numerical oracle style
- Metal parameter struct conventions
- tile constants where applicable (`BK`, `DHT`)
- profile/dispatch/planning patterns

Different from existing SDPA:

- K/V read from cache capacity dimension.
- live key range is `[key_start, base_len[b] + t + 1)`, with `key_start`
  derived from `sliding_window`, not `Tk` or causal offset.
- no backward.
- no dense additive bias in v1.
- no split-K initially unless profiling shows decode needs it.

### 5.3 CPU helper factoring

Current CPU reference code lives in `src/ops/sdpa/kernel_sdpa.c`. Refactor into
small shared helpers before adding decode:

```c
/* shared */
int  gd_sdpa_hkv_for_hq(int hq, int Hq, int Hkv);
float gd_sdpa_resolve_scale(float requested, int Dh);
void gd_sdpa_online_init(...);
void gd_sdpa_online_update(...);
void gd_sdpa_online_finish(...);

/* existing full path */
_gd_cpu_k_sdpa(...)
_gd_cpu_k_sdpa_bwd(...)

/* new cache path */
_gd_cpu_k_sdpa_decode(...)
```

No need to share every load expression. Full SDPA and cache SDPA have different
storage accessors. Share math, not index boilerplate.

### 5.4 Metal helper factoring

Current Metal forward helpers are local to `metal_sdpa_fwd.metal`. Add a small
include-style helper file, for example:

```text
src/ops/sdpa/metal_sdpa_common.metal.h
```

Shared inline helpers:

```metal
static inline int gd_sdpa_hkv_for_hq(int hq, int Hq, int Hkv);
static inline void gd_sdpa_online_init(thread float &m, thread float &l);
static inline void gd_sdpa_online_step(...);
static inline float gd_sdpa_dot(...);
```

Then:

```text
metal_sdpa_fwd.metal       includes common helpers
metal_sdpa_decode.metal    includes common helpers
metal_sdpa_bwd.metal       may include only safe non-conflicting helpers
```

If Metal include churn becomes risky, acceptable fallback is duplicating tiny
load wrappers only. Do not duplicate online-softmax/GQA logic.

### 5.5 Decode Metal kernel shape

Flat cache decode kernel:

```text
grid: one threadgroup per (batch row, query head, query block)
q block: small Tnew block, optimized for Tnew=1..4
key loop: tiles over [key_start, base_len[b] + t + 1)
```

Pseudo-kernel:

```metal
b, hq, qb = grid ids
hkv = hq / (Hq / Hkv)
for each query t in q block:
  if t >= append_len[b]: write zero
  else:
    end = base_len[b] + t + 1
    start = (window > 0) ? max(0, end - window) : 0
    online_softmax over j=start..end-1 using k_cache[b,j,hkv,:]
    accumulate v_cache[b,j,hkv,:]
```

For `Tnew=1`, this is the normal fast generation path. For future `Tnew=4`, the
same staged K/V tile can be reused by several query rows in one threadgroup.

### 5.6 No backward path

`gd_sdpa_decode` is inference-only. It must not appear in `gd_backward` graphs.
Training keeps using `gd_sdpa`, so existing SDPA backward stays untouched.

---

## 6. GPT integration

Factor the current attention block into shared projection helpers:

```text
project q/k/v
reshape to [B,T,H,Dh]
RoPE(q,k)
```

Then two attention consumers:

### 6.1 Training/full forward

Existing behavior remains:

```text
o = gd_sdpa(q_rope, k_rope, v, NULL, causal_cfg)
```

### 6.2 Prefill with cache population

```text
q,k,v = projections + RoPE
kv_append(cache[layer], k, v,
          base_len=cache.base_len_tensor,      # 0 for prefill
          append_len=cache.append_len_tensor,  # prompt_lens or T
          src_offset=cache.src_offset_tensor)
o = gd_sdpa(q, k, v, prefill_bias, causal_cfg)
```

Same-length prompts call `gd_kv_cache_set_prefill_lengths` with all lengths `T`,
use `src_offset=0`, `prefill_bias=NULL`, and existing fast causal SDPA.

Ragged padded prompts:

- Host `prompt_lens[B]` are source of truth. Generation calls
  `gd_kv_cache_set_prefill_lengths(cache, prompt_lens, Tmax, true, B)` before
  recording/running the prefill graph.
- Generation default is left padding: valid prompt tokens occupy the right side
  of `[B,Tmax]`, so last-token logits are at column `Tmax-1` for every row.
- `positions` for valid tokens are logical positions `0..prompt_lens[b)-1`; pad
  positions are don't-care because pad-key bias hides them and pad-query outputs
  are ignored.
- Cache append uses `src_offset[b] = Tmax - prompt_lens[b]` so cache rows are
  compacted to logical positions `0..prompt_lens[b)-1`.
- `prefill_bias` is generated from host `prompt_lens` as additive pad-key mask
  (broadcastable `[B,1,1,Tmax]` is enough). Causal structure still comes from
  `gd_sdpa(causal=true)`.
- Correctness via existing additive bias mask in v1.
- Not fully compute-fast: dense SDPA still scans `Tmax^2`.
- Production use should length-bucket prompts to reduce padding waste.
- Native variable-length prefill can come later, likely via paged/packed SDPA.

After successful prefill graph execution, host calls `gd_kv_cache_commit(cache,
NULL)` before first decode so `committed_len[b] == prompt_lens[b]`.

### 6.3 Decode/block decode

```text
q,k,v = projections + RoPE
kv_append(cache[layer], k, v,
          cache.base_len_tensor,
          cache.append_len_tensor,
          src_offset=NULL)
o = gd_sdpa_decode(q, cache[layer].k, cache[layer].v,
                   cache.base_len_tensor,
                   cache.append_len_tensor,
                   causal_cfg_with_sliding_window)
```

For valid decode tokens, `positions[b,t]` must equal `committed_len[b] + t`
(the same value as `base_len[b] + t` loaded into the cache tensors). This keeps
RoPE aligned with cache coordinates and makes sliding-window parity exact.

After all layers and LM head, host samples/accepts tokens and calls
`gd_kv_cache_commit(cache, accepted_len_or_null)`.

---

## 7. Padding and batching behavior

### Fast path

```text
B fixed
Tprefill fixed per compiled prefill graph
Tnew fixed per compiled decode graph
all prompt_lens equal
append_len[b] = Tnew
```

This path uses existing SDPA fast kernels for prefill and decode scans exactly
live cache lengths, or exactly the live sliding-window range when
`attention_window > 0`.

### Ragged prefill

Supported for correctness:

```text
tokens [B,Tmax] left-padded by default for generation
prompt_lens [B]
src_offset = Tmax - prompt_lens
append_len = prompt_lens
positions count valid tokens from 0, not physical columns
prefill SDPA uses additive pad bias
```

Performance caveat: prefill compute is `O(B*Tmax^2)`, not sum of per-row lengths.
Length bucketing recommended.

### Ragged decode

Supported efficiently:

```text
base_len[B]
append_len[B]
sliding_window optional per model config
```

Each row loops only to its own live key range:
`[max(0, base_len[b] + t + 1 - window), base_len[b] + t + 1)` when windowed,
or `[0, base_len[b] + t + 1)` when unwindowed.

---

## 8. Validation tests

### 8.1 KV append

- CPU append writes exact rows for `Tnew=1` and `Tnew=4`.
- Per-row `append_len` skips padded query rows.
- `src_offset` compacts left-padded prefill K/V into cache row 0.
- Bounds errors when `base_len + append_len > capacity` or `src_offset + append_len > Tnew`.
- Reject dtype/device mismatches, grad-enabled cache tensors, and cache/new aliasing.
- Metal append matches CPU and persists across graph runs.
- Reused decode graph sees updated cache length tensors across many runs.
- `reset` clears host committed lengths and length tensors.

### 8.2 SDPA decode unit tests

- `Tnew=1`: decode output matches `gd_sdpa` over explicitly materialized
  `k/v[:, :live_len]`.
- `Tnew=4`: causal-in-block output matches full causal `gd_sdpa` on
  `old_cache + new_block`.
- Sliding-window decode (`window=1,4,17`) matches `gd_sdpa(causal=true,
  sliding_window=window)` over materialized K/V.
- GQA case `Hq > Hkv` matches CPU_REF.
- Ragged `base_len[B]` matches per-row reference.
- Invalid padded query positions output zero.
- `prefix_len > 0` is rejected in v1 decode.

### 8.3 GPT parity tests

Definitive test:

```text
full logits = gd_gpt_forward(tokens[0:T])
cache logits = prefill first N tokens, then decode tokens N..T-1
compare logits at each decoded position
```

Run on:

- CPU_REF
- Metal
- batched same-length
- batched ragged decode
- `Tnew=1`
- `Tnew=4` block decode
- `attention_window=0` and `attention_window>0`
- F32 and F16 cache dtypes where backend supports them

Acceptance tolerance mirrors existing GPT/SDPA parity.

### 8.4 Generation sanity

- Greedy generation stops on `<|im_end|>`.
- `prompt_len + max_new_tokens <= capacity` validation.
- EOS rows can set `append_len=0` and remain stable while other rows continue.
- `accepted_len < append_len` commits only accepted prefix and later append
  overwrites uncommitted rows.
- No recompute of old K/V during decode, verified by profiler/op trace.

---

## 9. Phases

### K0 — API + cache object

- Add `gd_kv_cache` opaque type.
- Allocate flat per-layer K/V tensors with resolved dtype.
- Add host committed-length mirror plus device `base_len/append_len/src_offset`
  tensors.
- Add reset/set/set-prefill/commit helpers, including partial accepted commit.

### K1 — CPU_REF ops

- `_GD_OP_KV_APPEND` CPU kernel.
- `_GD_OP_SDPA_DECODE` CPU kernel with sliding-window support.
- Shared CPU SDPA helper factoring.
- Unit tests vs existing `gd_sdpa`, including windowed decode.

### K2 — Metal KV append

- Metal append kernel.
- Mark `_GD_OP_KV_APPEND` as mutating cache inputs in Metal compile/writeback
  bookkeeping.
- Ensure device-resident cache aliases persistent Metal tensors.

### K3 — Metal SDPA decode

- Add `metal_sdpa_common.metal.h` helpers.
- Add `metal_sdpa_decode.metal` entrypoint with sliding-window start bound.
- Reuse SDPA params/planning style.
- CPU↔Metal parity for `Tnew=1`, `Tnew=4`, and windowed decode.

### K4 — GPT prefill/decode APIs

- Factor projection/RoPE helpers in `src/nn/nn.c`.
- Add `gd_gpt_prefill` and `gd_gpt_decode_block`.
- Full-forward vs cached-decode parity tests, with and without sliding window.

### K5 — Generation CLI

- Generation uses KV cache by default.
- Sampling reads only last-token logits.
- Validate `<|im_end|>` stop and capacity cap.

### K6 — Later: paged KV cache

- Replace flat storage inside `gd_kv_cache` with page pool + block table.
- Keep GPT API stable.
- Add page-table SDPA decode kernel.

---

## 10. Acceptance criteria

- No duplicate large SDPA implementation: decode shares SDPA math helpers and
  conventions; full prefill/training remains on existing `gd_sdpa`.
- Cached decode logits match full-forward logits token-by-token.
- Sliding-window cached decode is supported in v1 and matches full-forward.
- Batched decode supports per-row lengths.
- `Tnew=4` block decode works for future MTP plumbing.
- CPU_REF and Metal parity pass.
- `make check` green.
