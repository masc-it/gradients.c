# gradients.c — GPT KV Cache Design

Status: planned v0.1

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
  committed_len[B]
  capacity
  batch_size
```

Graph does not allocate pages, resize buffers, or own serving policy.

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

gd_status gd_kv_cache_commit(gd_kv_cache *cache);
```

`commit` updates host-side `committed_len[b] += append_len[b]` after successful
graph execution. It does not need to read GPU data.

### 3.2 GPT inference APIs

```c
gd_status gd_gpt_prefill(gd_context *ctx,
                         gd_gpt *gpt,
                         gd_kv_cache *cache,
                         gd_tensor *tokens,       /* int32 [B,T] */
                         gd_tensor *positions,    /* int32 [B,T] */
                         gd_tensor *prompt_lens,  /* nullable int32 [B] */
                         gd_tensor **logits_out);

gd_status gd_gpt_decode_block(gd_context *ctx,
                              gd_gpt *gpt,
                              gd_kv_cache *cache,
                              gd_tensor *tokens,      /* int32 [B,Tnew] */
                              gd_tensor *positions,   /* int32 [B,Tnew] */
                              gd_tensor **logits_out);
```

`gd_gpt_decode_block` uses the cache's `base_len/append_len` tensors set before
recording/running the graph.

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

Op is inference-only, mutating, no backward.

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
- `sliding_window`: optional later.
- `prefix_len`: not used in decode v1.
- `bias`: omitted in v1 decode path.

Per query row:

```text
valid_q = t < append_len[b]
key_end = base_len[b] + t + 1
if !valid_q: out[b,t,h,:] = 0
else attend j in [0, key_end)
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
- live key bound is `base_len[b] + t + 1`, not `Tk` or causal offset.
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
key loop: tiles over [0, base_len[b] + t + 1)
```

Pseudo-kernel:

```metal
b, hq, qb = grid ids
hkv = hq / (Hq / Hkv)
for each query t in q block:
  if t >= append_len[b]: write zero
  else:
    end = base_len[b] + t + 1
    online_softmax over j=0..end-1 using k_cache[b,j,hkv,:]
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
          base_len=0,
          append_len=prompt_lens or T,
          src_offset=left_pad_offsets or 0)
o = gd_sdpa(q, k, v, prefill_bias_or_null, causal_cfg)
```

Same-length prompts use `bias=NULL` and existing fast causal SDPA.

Ragged padded prompts:

- Generation default is left padding: valid prompt tokens occupy the right side
  of `[B,Tmax]`, so last-token logits are at column `Tmax-1` for every row.
- `positions` for valid tokens are logical positions `0..prompt_lens[b)-1`; pad
  positions are don't-care because mask hides them.
- Cache append uses `src_offset[b] = Tmax - prompt_lens[b]` so cache rows are
  compacted to logical positions `0..prompt_lens[b)-1`.
- Correctness via existing additive bias mask in v1.
- Not fully compute-fast: dense SDPA still scans `Tmax^2`.
- Production use should length-bucket prompts to reduce padding waste.
- Native variable-length prefill can come later, likely via paged/packed SDPA.

### 6.3 Decode/block decode

```text
q,k,v = projections + RoPE
kv_append(cache[layer], k, v, base_len, append_len)
o = gd_sdpa_decode(q, cache[layer].k, cache[layer].v, base_len, append_len)
```

After all layers and LM head, host samples/accepts tokens and calls
`gd_kv_cache_commit(cache)`.

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
live cache lengths.

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
```

Each row loops only to its own `base_len[b] + t + 1`.

---

## 8. Validation tests

### 8.1 KV append

- CPU append writes exact rows for `Tnew=1` and `Tnew=4`.
- Per-row `append_len` skips padded query rows.
- `src_offset` compacts left-padded prefill K/V into cache row 0.
- Bounds errors when `base_len + append_len > capacity` or `src_offset + append_len > Tnew`.
- Metal append matches CPU and persists across graph runs.

### 8.2 SDPA decode unit tests

- `Tnew=1`: decode output matches `gd_sdpa` over explicitly materialized
  `k/v[:, :live_len]`.
- `Tnew=4`: causal-in-block output matches full causal `gd_sdpa` on
  `old_cache + new_block`.
- GQA case `Hq > Hkv` matches CPU_REF.
- Ragged `base_len[B]` matches per-row reference.
- Invalid padded query positions output zero.

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

Acceptance tolerance mirrors existing GPT/SDPA parity.

### 8.4 Generation sanity

- Greedy generation stops on `<|im_end|>`.
- `prompt_len + max_new_tokens <= capacity` validation.
- No recompute of old K/V during decode, verified by profiler/op trace.

---

## 9. Phases

### K0 — API + cache object

- Add `gd_kv_cache` opaque type.
- Allocate flat per-layer K/V tensors.
- Add length mirror/tensors.
- Add reset/set/commit helpers.

### K1 — CPU_REF ops

- `_GD_OP_KV_APPEND` CPU kernel.
- `_GD_OP_SDPA_DECODE` CPU kernel.
- Shared CPU SDPA helper factoring.
- Unit tests vs existing `gd_sdpa`.

### K2 — Metal KV append

- Metal append kernel.
- Mark `_GD_OP_KV_APPEND` as mutating cache inputs in Metal compile/writeback
  bookkeeping.
- Ensure device-resident cache aliases persistent Metal tensors.

### K3 — Metal SDPA decode

- Add `metal_sdpa_common.metal.h` helpers.
- Add `metal_sdpa_decode.metal` entrypoint.
- Reuse SDPA params/planning style.
- CPU↔Metal parity for `Tnew=1` and `Tnew=4`.

### K4 — GPT prefill/decode APIs

- Factor projection/RoPE helpers in `src/nn/nn.c`.
- Add `gd_gpt_prefill` and `gd_gpt_decode_block`.
- Full-forward vs cached-decode parity tests.

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
- Batched decode supports per-row lengths.
- `Tnew=4` block decode works for future MTP plumbing.
- CPU_REF and Metal parity pass.
- `make check` green.
