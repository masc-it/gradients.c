# gradients.c — Decoder-only Transformer (GPT) Implementation Spec

Status: draft v0.1

Goal: specify everything required to build and train a **state-of-the-art
decoder-only transformer (GPT-style)** in gradients.c, and run efficient
inference, on both **CPU_REF** and **Metal**. Scope includes RoPE positional
encoding, scaled dot-product attention (SDPA) with a **block-sparse** kernel
path, grouped-query attention (GQA), a SwiGLU/GELU MLP, RMSNorm pre-norm blocks,
weight-tied LM head, and a **KV cache** for autoregressive decode.

This is a *what to build* document. It defines the new public ops, the IR
additions, autograd, the CPU and Metal kernels, the attention/KV-cache data
model, and a phased, parity-gated delivery plan. It does not write kernels; it
specifies their contracts precisely enough to implement and test.

Reference: [`docs/design_spec.md`](design_spec.md) §10–12 (ops/IR), §15–16
(autograd), §17–19 (private backend API, fusion IR), §20–21 (parity/testing),
§22 Phase 4 (transformer path), §23 (closed decisions — esp. #5 static shapes,
#8 straight-line graphs). Foundations already shipped:
[`plan_metal_prereqs.md`](plan_metal_prereqs.md) (backend seam, parity harness),
[`plan_metal.md`](plan_metal.md) (Metal backend M0–M4: elementwise, matmul,
linear, reductions, softmax, rms_norm, cross_entropy, full backward + AdamW,
MLP training parity).

Guiding invariants (inherited, must not break):
1. **Graph-only core.** Every op lowers to typed IR (`_gd_op_kind`). No second
   execution path.
2. **CPU_REF is the oracle.** Every new op ships a simple, correct CPU kernel
   first; optimized/fused/sparse/Metal paths prove parity against it via
   `gd_graph_compare` (forward) and separate-run gradient comparison.
3. **Backend = one vtable + kernels.** New ops add an `_gd_op_kind`, shape
   inference, an autograd rule, a CPU kernel, and a Metal kernel; `supports_node`
   gates fallback. No core/executor changes.
4. **Static compiled shapes (decision #5).** Prefill `[B,T]` and decode `[B,1]`
   are *separate compiled graphs*. Dynamic sequence growth lives in the KV cache
   (persistent tensors), not in dynamic graph shapes.

Non-goals (v1 of this plan): training-time block-sparsity autograd beyond dense
+ causal (sparse is inference/forward-first), bf16/fp16/quantized attention
kernels (fp32 first), tensor/sequence/pipeline parallelism, paged-attention
across devices, speculative decoding, MoE, learned/ALiBi positional schemes
other than RoPE, fused QKV+RoPE+attention single-kernel (kept as a later fusion
rung; we land correct unfused ops first).

---

## 1. Target model

A configurable decoder-only transformer:

```
tokens[B,T] (int32)
  -> token_embedding (gather rows of Wte[V, d_model])           # x[B,T,d]
  -> for each of L blocks:
        h  = x + attn( rms_norm(x, w_ln1) )                     # pre-norm residual
        x  = h + mlp( rms_norm(h, w_ln2) )
  -> x = rms_norm(x, w_lnf)
  -> logits = x @ Wte^T            (weight-tied LM head)         # logits[B,T,V]
  -> loss = cross_entropy(logits, targets)                      # training
```

Attention block (per layer), with GQA:
```
rms_norm -> qkv projection (linear, no bias) -> split q[B,T,Hq,Dh], k,v[B,T,Hkv,Dh]
         -> RoPE(q), RoPE(k)
         -> SDPA(q, k, v, mask)   (causal; optional block-sparse)   -> o[B,T,Hq,Dh]
         -> merge heads -> out projection (linear) -> residual add
```
MLP block (SwiGLU default; GELU selectable):
```
rms_norm -> w_gate: linear(d, d_ff), w_up: linear(d, d_ff)
         -> silu(gate) * up    (SwiGLU)   [or gelu(linear) for vanilla GPT]
         -> w_down: linear(d_ff, d)
```

Configuration surface (`gd_gpt_config`): `vocab_size V`, `d_model d`,
`n_layers L`, `n_heads Hq`, `n_kv_heads Hkv` (Hkv==Hq → MHA; 1 → MQA; else GQA),
`head_dim Dh` (d == Hq*Dh), `d_ff`, `max_seq_len`, `rope_theta`, `norm_eps`,
`mlp_kind {SWIGLU, GELU}`, `tie_embeddings`, `attn_mask` (see §5).

---

## 2. Gap analysis vs current ops

Already available (CPU + Metal, with autograd): `add`, `mul`, `scale`,
`matmul` (+trans, batched, broadcast), `linear` (+bias), `relu`, `silu`,
`sum`, `mean`, `softmax`, `rms_norm`, `cross_entropy`, `cast`, `copy`
(reshape), plus all backward kernels and AdamW.

**Missing ops to add** (each = public API + `_gd_op_kind` + attrs + shape
inference in `src/ops/shape.c` + autograd rule in `src/autograd/autograd.c` +
CPU kernel in `src/backends/cpu_ref/cpu_kernels.c` + Metal kernel in
`src/backends/metal/kernels.metal` + `supports_node`/encode wiring +
CPU↔Metal parity test):

| Op | Purpose | Notes |
|----|---------|-------|
| `embedding` (gather) | token ids → rows of `Wte` | backward = scatter-add into `Wte.grad` |
| `transpose`/`permute` | head/seq reshuffles | needed because views of *virtual* tensors are contiguous-only today |
| `rope` | rotary positional encoding on q,k | precomputed cos/sin or position ids |
| `sdpa` | scaled dot-product attention | coarse fused op; dense+causal first, block-sparse path next; FlashAttention-style |
| `gelu` | vanilla-GPT MLP activation | exact (erf) + tanh approx variants |
| `add3`/residual | (optional) fuse residual adds | not required; `add` suffices |
| `kv_cache_append` | write new k,v into cache slots | inference-only; in-place external write (KV tensors) |

Reuse without new ops: SwiGLU = `silu(gate) * up` via existing `silu`+`mul`;
RMSNorm exists; LM head = `matmul`/`linear` with tied weight; residual = `add`;
loss = `cross_entropy`.

**Why a coarse `sdpa` op (not just matmul+softmax+matmul):** spec §18 wants the
IR to expose whole patterns so backends can lower to FlashAttention-style
kernels. A single `sdpa` node carries the mask/causal/scale/GQA metadata,
enables the online-softmax tiled kernel (O(T) memory, not O(T²) score matrix),
and is the natural home for the block-sparse schedule and the KV cache. We still
ship an *unfused reference* (the explicit matmul→softmax→matmul subgraph) for
parity (the GPU debug ladder rung `GPU_SAFE`).

---

## 3. IR and op additions

### 3.1 `_gd_op_kind` additions (`src/graph/graph_internal.h`)
```
_GD_OP_EMBEDDING, _GD_OP_EMBEDDING_BWD
_GD_OP_TRANSPOSE            /* permute; its own backward is another transpose */
_GD_OP_ROPE, _GD_OP_ROPE_BWD
_GD_OP_SDPA, _GD_OP_SDPA_BWD
_GD_OP_GELU, _GD_OP_GELU_BWD
_GD_OP_KV_APPEND           /* inference-only, in-place, no backward */
```

### 3.2 `_gd_op_attrs` additions
Extend the existing flat attrs struct (keep POD; backends read it directly):
```c
/* transpose */            int perm[GD_MAX_DIMS]; int perm_ndim;
/* rope */                 float rope_theta; int rope_n_dims; int rope_interleaved;
/* sdpa */                 float attn_scale;   /* default 1/sqrt(Dh) */
                           int   n_q_heads, n_kv_heads, head_dim;
                           int   causal;        /* bool */
                           int   sliding_window;/* 0 = none */
                           int   attn_block_q, attn_block_k; /* sparse block sizes */
                           /* block-sparse pattern is a side input value, not an attr (see §5) */
/* gelu */                 int   gelu_tanh;     /* bool: tanh approx vs exact */
```
Rationale: small fixed fields stay in attrs; variable-size data (block-sparse
mask, RoPE tables, KV cache) are passed as *graph values* (tensors), not attrs.

### 3.3 Shape inference (`src/ops/shape.c`)
- `embedding`: `ids[...]` (int) + `table[V,d]` → `out[..., d]`.
- `transpose`: permute `sizes`/strides by `perm`. **Note:** for virtual tensors
  this must produce a *contiguous* output (it is a real op that physically
  permutes), unlike the metadata-only `gd_tensor_transpose` view.
- `rope`: out shape == in shape `[B,T,H,Dh]`.
- `sdpa`: `q[B,Tq,Hq,Dh]`, `k[B,Tk,Hkv,Dh]`, `v[B,Tk,Hkv,Dh]` → `o[B,Tq,Hq,Dh]`.
- `gelu`: out == in.

### 3.4 Public API (`include/gradients/ops.h`)
```c
gd_status gd_embedding(gd_context*, gd_tensor *table, gd_tensor *ids, gd_tensor **out);
gd_status gd_transpose(gd_context*, gd_tensor *x, const int *perm, int ndim, gd_tensor **out);
gd_status gd_gelu(gd_context*, gd_tensor *x, bool tanh_approx, gd_tensor **out);

typedef struct gd_rope_config {
    float theta;          /* default 10000 */
    int   n_dims;         /* rotary dims; 0 => full head_dim */
    bool  interleaved;    /* GPT-NeoX (false: half-split) vs GPT-J (true) */
} gd_rope_config;
gd_status gd_rope(gd_context*, gd_tensor *x, gd_tensor *pos_ids,
                  const gd_rope_config*, gd_tensor **out);

typedef struct gd_sdpa_config {
    float scale;          /* 0 => 1/sqrt(head_dim) */
    bool  causal;
    int   sliding_window; /* 0 => none */
    /* block-sparse (G4): NULL => dense (respecting causal/window/bias). */
    gd_tensor *block_mask;/* see §5: int32 [n_q_blocks, max_k_blocks] of key-block ids, -1 padded */
    int   block_q, block_k;
} gd_sdpa_config;
/* `bias` (nullable, landed in G2) is an additive score bias broadcast over
 * [B,Hq,Tq,Tk]: padding masks, ALiBi, relative-position bias. n_kv_heads is
 * derived from k's shape (GQA). */
gd_status gd_sdpa(gd_context*, gd_tensor *q, gd_tensor *k, gd_tensor *v,
                  gd_tensor *bias, const gd_sdpa_config*, gd_tensor **out);
```
A thin convenience layer (`include/gradients/nn.h`, optional) can offer
`gd_attention_block`, `gd_mlp_swiglu`, and a `gd_gpt` builder that records the
whole model into the active graph; these are pure compositions of the ops above
and need no backend support.

---

## 4. Autograd rules (`src/autograd/autograd.c`)

Each new op adds a `case` in `backward_node`. Patterns reuse `emit_custom`,
`accumulate`, `accumulate_broadcast`, `matmul_backward`.

- **embedding**: `dtable = scatter_add(d_out, ids)`. New `_GD_OP_EMBEDDING_BWD`
  kernel accumulates rows by id. `ids` has no grad. Must zero `dtable` then
  scatter (CPU: serial; Metal: single-thread scatter like `reduce_to`, or
  atomics later).
- **transpose**: `dx = transpose(d_out, inverse_perm)`. Emits another transpose.
- **rope**: rotary is orthogonal/linear in the rotated plane → `dx =
  rope(d_out, pos, -theta-rotation)` i.e. apply the *transpose* (negative-angle)
  rotation. `_GD_OP_ROPE_BWD` reuses the rope kernel with conjugated sin.
- **gelu**: `dx = d_out * gelu'(x)`; `_GD_OP_GELU_BWD` elementwise.
- **sdpa**: see §6.3 — `_GD_OP_SDPA_BWD` produces `dq, dk, dv` (three outputs;
  use the multi-output node path or emit three nodes sharing recomputed stats).
  v1 backward supports dense + causal (training path); block-sparse backward is a
  later rung (inference uses forward-only).
- **kv_append**: inference-only, not differentiable; never appears in a graph
  built for `gd_backward`.

Training graphs (prefill with labels) must use only differentiable ops; KV
cache + sdpa-decode are inference graphs (no `gd_backward`).

---

## 5. Block-sparse attention mask model

Two complementary masking mechanisms:
- **Additive bias** (dense, landed in G2): an optional `[B,Hq,Tq,Tk]`-broadcast
  float tensor added to scores before softmax — padding masks, ALiBi, arbitrary
  relative-position bias. Cheap to author, fully general, but materializes an
  O(Tq·Tk) bias (or a broadcast slice). Use for masks/biases that are dense or
  cheaply broadcast.
- **Block-sparse schedule** (this section, G4): for *high* sparsity where we
  want to *skip* work, not just bias it. Expressed at block granularity so the
  kernel visits only allowed key blocks.

The block-sparse pattern separates *what is allowed* (sparsity pattern) from
*how it is computed* (kernel schedule), at **block granularity** over the
(query, key) position grid, which is what hardware-efficient sparse attention
needs (and what causal/sliding-window degenerate to).

Definitions:
- Tile the query axis into blocks of `block_q` and the key axis into blocks of
  `block_k` (e.g. 64/128). `n_q_blocks = ceil(Tq/block_q)`,
  `n_k_blocks = ceil(Tk/block_k)`.
- **Block mask** = for each query block, the list of key blocks that may
  contribute. Represented as an int32 tensor `block_mask[n_q_blocks, max_kb]`
  holding key-block indices, `-1` padded (a CSR-like dense-of-lists). This is a
  *graph value* (so it can be precomputed on host and uploaded), not an attr.
- **Causal** and **sliding_window** are handled two ways that must agree:
  (a) as implicit rules the kernel applies *within* a visited block (per-element
  masking at the diagonal / window edge), and (b) the host helper that *builds*
  the block list only includes blocks that intersect the causal/window region.
  The kernel always applies the fine-grained causal/window predicate on
  elements, so a coarse block list is safe (never leaks future tokens).

Provided patterns (host builders in `nn.h`, returning a `block_mask` tensor):
- `dense` (block_mask = NULL): every key block; kernel still applies causal.
- `causal` : lower-triangular block grid.
- `sliding_window(w)`: blocks within `w` of the diagonal.
- `block_sparse(custom)`: caller-supplied (qb→[kb]) lists (e.g. BigBird:
  window + global + random).

Determinism: softmax normalization is over the *union of visited keys*; because
the kernel visits exactly the allowed (and causal-valid) keys, the result is the
exact dense-masked softmax restricted to the allowed set. The reference
(`GPU_SAFE`) computes the full score matrix, applies the same boolean mask
(expanded from blocks + causal), and softmaxes — these must match bit-for-block
within tolerance.

---

## 6. SDPA kernel design

### 6.1 Math
For each (batch b, query head h, query position i):
```
scale = config.scale or 1/sqrt(Dh)
scores_j = scale * dot(q[b,i,h,:], k[b,j, h_kv(h), :])  for allowed/causal j
p = softmax_j(scores_j)           # online/streaming
o[b,i,h,:] = sum_j p_j * v[b,j, h_kv(h), :]
```
GQA: `h_kv(h) = h / (Hq/Hkv)`. Causal: `j <= i` (offset by KV cache past length
during decode, see §7). Sliding window: `i - j < w`.

### 6.2 Forward kernels
Two implementations, parity-checked against each other and against the unfused
subgraph:

- **Reference (CPU_REF; Metal `GPU_SAFE`)**: materialize `scores[Tq,Tk]` per
  (b,h), apply mask, stable softmax (max-subtract), `@ v`. O(T²) memory. Simple,
  obviously correct, the oracle.
- **Tiled / FlashAttention-style (Metal `GPU_FUSED`)**: one threadgroup per
  (b, h, query-block). Stream key blocks (only allowed ones via `block_mask`):
  keep running max `m`, running denom `l`, running output accumulator `acc`;
  rescale on new max. No T² score matrix. Applies causal/window per element.
  This is the production path and the home of block sparsity (it simply iterates
  the visited key-block list).

CPU gets the reference only in v1 (correctness oracle); a blocked CPU variant is
optional later.

### 6.3 Backward (training; dense + causal in v1)
SDPA backward needs `p` (softmax probs) which the forward did not store (tiled
kernel is memory-light). Two options, decided here:
- **v1: reference backward** recomputes `scores`, `p` (O(T²) per (b,h)) and
  produces `dq, dk, dv` with the standard SDPA-grad formulas:
  ```
  dv = p^T @ d_o
  dp = d_o @ v^T
  ds = p * (dp - rowsum(dp * p))      # softmax jacobian
  dq = scale * ds @ k ;  dk = scale * ds^T @ q
  ```
  Correct and simple; matches the unfused-subgraph gradient. Good enough for
  training small/medium models.
- **later: FlashAttention-2 backward** (recompute `p` in tiles using stored
  per-row `(m,l)` logsumexp). Deferred; parity vs reference backward.

The `sdpa` forward (tiled) optionally outputs the per-row logsumexp `L[B,Tq,Hq]`
as a second value so a future tiled backward can recompute without the full
score matrix. v1 may ignore it.

### 6.4 Backend wiring
- New encode functions in `metal_backend.m`; new CPU kernels in
  `cpu_kernels.c`. SDPA needs several buffers (q,k,v,out, optional block_mask,
  optional L) + an `sdpa_params` struct (Tq,Tk,Hq,Hkv,Dh,scale,causal,window,
  block sizes, n_kb). `supports_node` returns true when the pipeline exists; a
  block-sparse node with no kernel falls back to CPU under `GD_FALLBACK_CPU_REF`.
- Dispatch: reference = per (b,h,i) thread; tiled = 2D/3D grid over
  (b*Hq, n_q_blocks) with threadgroup memory for the K/V tile and the softmax
  running state.

---

## 7. KV cache (inference)

Per closed decision #5 (static shapes) and #8 (straight-line graphs, host loop),
decode is a *host loop running a compiled decode graph* once per new token; the
growing sequence lives in persistent **KV cache tensors**, not in graph shapes.

### 7.1 Data model
Per layer, two persistent tensors (materialized, owned by the runtime, like
optimizer state — they survive across `gd_graph_run` calls and Metal write-back
keeps them current):
```
K_cache[B, max_seq, Hkv, Dh]
V_cache[B, max_seq, Hkv, Dh]
```
A scalar/host `past_len` tracks how many positions are valid. Optionally a
`cache_pos[B]` tensor for ragged batches (later; v1 assumes uniform `past_len`).

### 7.2 Two compiled graphs
- **Prefill graph** (shape `[B, T0]`): embeds the prompt, runs all blocks,
  writes K,V for positions `[0, T0)` into the cache via `kv_append`, produces
  logits for the last position (or all positions). RoPE uses positions
  `[0, T0)`. SDPA is causal over `[0,T0)`.
- **Decode graph** (shape `[B, 1]`): embeds one token, RoPE at position
  `past_len`, `kv_append` writes K,V at slot `past_len`, SDPA attends query (1
  position) over keys `[0, past_len]` from the cache, produces next-token
  logits. Re-run each step; host increments `past_len`.

`kv_append` is an in-place op: it writes the freshly-projected k,v for the
current positions into the cache tensors at offset `past_len` (a graph value or
attr). On Metal this reuses the **external write-back** mechanism from
[`plan_metal.md`](plan_metal.md) M3 (in-place mutation of external leaf tensors
is copied back to CPU storage; for a pure-GPU decode loop the cache can stay a
device-resident tensor and skip per-step write-back — a later optimization).

SDPA in decode reads K,V directly from the cache tensors (shape `[B, max_seq,
Hkv, Dh]`) but only attends `[0, past_len]`; `past_len` is passed as an sdpa
param (`k_len`) each step. Because graph shapes are static (`max_seq`), the kernel
bounds the key loop by the runtime `k_len`, not the tensor extent.

### 7.3 `past_len` as a runtime scalar
Static shapes mean we cannot recompile per step. `past_len` and the current
write offset are passed as **small device/host scalars updated between runs**
(a 1-element int tensor uploaded each step, or an sdpa-param the host patches via
a tiny upload). The decode graph references it; the kernels read it. This keeps
one compiled decode graph for the whole generation.

### 7.4 Paging (later)
For long context / batched serving, replace the flat `[B,max_seq,...]` cache
with block/paged KV (fixed-size pages indexed by a block table) — a natural
extension of the block-sparse machinery (§5). Out of scope for v1; the
`block_mask`/`cache_pos` abstractions are chosen to allow it.

---

## 8. Layout and the transpose problem

Attention needs `[B,T,H,Dh]` ↔ `[B,H,T,Dh]` reshuffles. Today, views of *virtual*
graph tensors are contiguous-only (`gd_tensor_transpose` is metadata and only
valid on materialized tensors). Two resolutions, both specified:

1. **Add a real `transpose`/`permute` op** (CPU + Metal kernels + autograd) that
   physically permutes into a contiguous result. General, reusable, and required
   anyway for robust graph authoring.
2. **Keep attention head-major throughout**: define `sdpa` to consume
   `q,k,v` in `[B,T,H,Dh]` and handle the `(T,H)` access pattern *inside* the
   kernel (stride math), avoiding explicit transposes. The QKV projection
   produces `[B,T, (Hq+2Hkv)*Dh]`; a `reshape` (contiguous, already supported as
   `_GD_OP_COPY`) splits it into `[B,T,H,Dh]`. RoPE and SDPA index heads via
   strides. This avoids transposes on the hot path.

**Decision:** do both — implement `transpose` (item 1) for generality and
backward correctness of arbitrary graphs, but author the GPT model to prefer the
head-major, transpose-free path (item 2) so the attention hot path needs no
permute kernels. `sdpa` therefore takes `[B,T,H,Dh]` directly.

### Why not `[B,H,T,Dh]` (batch-head-first)?
A reasonable instinct is that GPU attention "wants" `[B,H,T,Dh]` (PyTorch's SDPA
layout) and that callers must `transpose` into it. For *this* design that is
backwards:
- The QKV projection emits `[B,T,d]`; the natural reshape is `[B,T,H,Dh]`
  (head-major). RoPE and `sdpa` index heads by stride, so the GPT forward path
  has **zero** transpose nodes. Switching the op to `[B,H,T,Dh]` would *force* a
  `transpose` (a real kernel + autograd) in every model on the hot path — the
  opposite of the goal.
- `sdpa` is a coarse op, not a raw matmul: it owns its internal access pattern.
  The reference kernels already stride over `(T,H)` with no penalty to
  correctness.

The genuine, narrower issue is **physical K/V tiling for the fused kernel**
(G3): in `[B,T,H,Dh]` the per-head key stride is `Hkv·Dh`, so a FlashAttention
tile gathers strided rows (weaker coalescing). This is a *backend layout-planning*
concern, resolved inside the kernel, in preference order:
1. The G3 tiled kernel stages each head's K/V block into threadgroup/scratch
   memory once per tile (a strided gather on load, then contiguous reuse) — the
   standard FlashAttention tiling, which already does a blocked load.
2. A backend-internal repack of K/V to `[B,Hkv,Tk,Dh]` scratch before the
   attention loop when profitable (the deferred "layout planning" pass).
3. An opaque `layout` attr on the node letting the backend choose the physical
   layout it lowered for — never exposed to callers.
None of these change the public op signature or push a transpose onto users.
The `[B,T,H,Dh]` contract stays; physical layout is the backend's to optimize.

---

## 9. Numerics and dtypes

- **v1: fp32 everywhere** (params, activations, KV cache, attention compute).
  Matches the existing CPU/Metal parity tolerance regime (1e-4 forward, looser
  for accumulation-heavy paths).
- Softmax in attention uses max-subtraction (stable); tiled kernel uses online
  rescaling. RoPE computed in fp32.
- RMSNorm `eps` from config; reductions accumulate in fp32 on Metal (CPU uses
  fp64 — document the parity tolerance, as in M2).
- **Later**: bf16/fp16 storage with fp32 accumulation (compute policy already in
  attrs), then quantized weights for `linear`/`matmul`/attention projections
  (spec §11 allows quant tensors on these ops). KV-cache quantization (int8)
  later still.

---

## 10. Testing and parity plan

Every op and the assembled model follow the established ladder:
1. **CPU op correctness**: small hand-checked cases + finite-difference
   gradcheck (`gd_gradcheck`) for differentiable ops (embedding, transpose,
   rope, gelu, sdpa-dense-causal).
2. **CPU↔Metal forward parity** via `gd_graph_compare` (1e-4) per op.
3. **CPU↔Metal gradient parity** via separate per-device backward runs (the M3
   pattern), per differentiable op.
4. **SDPA cross-implementation parity**: unfused subgraph (matmul+softmax+
   matmul) vs reference `sdpa` vs tiled `sdpa` vs block-sparse `sdpa` (with a
   block mask equivalent to dense/causal) — all must agree.
5. **Block-sparse correctness**: a block mask that equals the causal pattern
   must match dense-causal exactly; an arbitrary sparse mask must match a
   reference that applies the expanded boolean mask.
6. **KV-cache parity**: prefill+decode of `T` tokens must equal a single
   full-sequence forward over the same `T` tokens (logits at each position
   match within tolerance). This is the definitive inference test.
7. **End-to-end training parity**: a tiny GPT (e.g. `V=32, d=32, L=2, Hq=4,
   Dh=8, d_ff=64, T=16`) trained N steps — forward logits, gradients, and
   post-AdamW params match CPU↔Metal (the M4 pattern, looser tol for drift).
8. **Overfit sanity**: the tiny GPT memorizes a small sequence (loss → ~0),
   run on both backends.

ASan/UBSan clean on all new kernels (the latent-bug gate from
[`plan_metal.md`](plan_metal.md)).

---

## 11. Inference runtime (host)

A small generation driver (example `examples/gpt/`):
- build + compile prefill and decode graphs once;
- run prefill on the prompt → logits + populated KV cache;
- loop: sample next token (greedy/temperature/top-k — host-side on CPU-read
  logits), upload it as the decode input, bump `past_len`, run decode graph;
- stop on EOS or length.

Sampling is host C (CPU), reading logits via `gd_tensor_copy_to_cpu`. No new
backend ops needed for sampling in v1.

---

## 12. Phasing

P0–P3 are ordered; later phases depend on earlier. Each op lands with its
CPU+Metal kernels and parity tests before the next phase.

- [x] **G0 — Primitive ops**: `embedding`(+bwd), `transpose`(+bwd), `gelu`(+bwd).
  CPU+Metal+autograd+parity. (No attention yet.)
  Landed: `_GD_OP_{GELU,GELU_BWD,TRANSPOSE,EMBEDDING,EMBEDDING_BWD}`; attrs
  `gelu_tanh`, `perm[]`/`perm_ndim`; `_gd_infer_transpose`/`_gd_infer_embedding`;
  public `gd_gelu`/`gd_transpose`/`gd_embedding`; autograd rules (gelu' exact+tanh,
  transpose via inverse-perm, embedding scatter-add); CPU kernels; Metal kernels
  (`gd_erff` A&S approximation since MSL lacks `erf`; transpose = 4-byte word
  permute; embedding gather + single-thread scatter, I32 ids). Tests:
  `tests/test_metal_gpt.c` (forward + gradient CPU↔Metal parity at 1e-4) and
  finite-difference gradchecks in `tests/test_autograd.c`. ASan-clean.
- [x] **G1 — RoPE**: `gd_rope`(+bwd), position-id handling, cos/sin computation.
  Parity + gradcheck. Head-major `[B,T,H,Dh]` layout.
  Landed: `_GD_OP_ROPE`/`_GD_OP_ROPE_BWD`; attrs `rope_theta`/`rope_n_dims`/
  `rope_interleaved`; `_gd_infer_rope` (positions index the product of dims
  before [heads, head_dim]); public `gd_rope` + `gd_rope_config` (theta, n_dims,
  NeoX half-split vs GPT-J interleaved). One kernel serves forward and backward
  via `sin_sign` (+1/-1); backward is the transpose (negative-angle) rotation.
  CPU kernel (fp64 angle math) + Metal kernel (I32 positions). Tests:
  forward parity (both layouts) + gradient parity in `test_metal_gpt.c`, and a
  finite-difference gradcheck in `test_autograd.c`. ASan-clean.
- [x] **G2 — SDPA dense+causal**: reference kernel (CPU + Metal `GPU_SAFE`),
  forward+backward, parity; GQA support. Trainable.
  Landed: `_GD_OP_SDPA`/`_GD_OP_SDPA_BWD`; attrs `attn_scale`/`n_q_heads`/
  `n_kv_heads`/`head_dim`/`causal`/`sliding_window`; `_gd_infer_sdpa` (4D
  head-major, Hq%Hkv==0). Public `gd_sdpa` + `gd_sdpa_config`. **Genuine
  multi-output IR node**: new `_gd_graph_emit_multi` records one node with N
  produced values/virtual tensors; `cpu_run_node` relaxed to 0..3 outputs; the
  SDPA backward node yields dq/dk/dv from a single node. Reference forward =
  two-pass stable softmax (no T² score matrix); reference backward recomputes
  softmax stats (O(T²)). Metal backward = two kernels under one node (dq per
  query row, dk/dv per kv row — no atomics, no cross-thread accumulation),
  dispatched in one encoder via a name-keyed pipeline lookup. CPU + Metal,
  grouped-query attention, causal/sliding-window predicate. Tests: forward
  parity (dense+causal, GQA) + dq/dk/dv gradient parity in `test_metal_gpt.c`;
  finite-difference gradchecks (dense + causal, GQA) in `test_autograd.c`.
  ASan-clean. (sdpa-vs-unfused-subgraph cross-check subsumed by the
  finite-difference gradcheck, an independent oracle.)

  Deferred to G3+: tiled FlashAttention forward (online softmax) and FA-2
  backward; v1 backward is O(T²)-recompute per (b,h) and the dk/dv kernel is
  O(T²) per kv slot — correctness-first.

  **Additive attention bias (landed).** `gd_sdpa` takes an optional `bias`
  tensor added to the scaled scores before softmax, broadcast over
  `[B, Hq, Tq, Tk]` (any axis may be 1). This is the standard SDPA float-mask /
  attention-bias input and covers **padding masks** (0 / large-negative), **ALiBi**
  (per-head linear bias), and arbitrary relative-position bias, composing with
  the causal/window fast path. Because bias is additive-constant in the scores,
  the dq/dk/dv formulas are unchanged — only the score computation gains a
  `+ bias` term (forward and the backward stat-recompute). Bias is **constant /
  non-differentiable in v1** (autograd treats it like positions/targets; learned
  relative bias grad is a later extension). Validated by a finite-difference
  gradcheck of q/k/v with a partial-masking bias, plus CPU↔Metal forward+grad
  parity (`sdpa_bias`). This addresses the "no custom attention mask" gap; the
  block-sparse path (G4) remains the structured-sparsity complement.
- [ ] **G3 — SDPA tiled (FlashAttention-style)**: Metal `GPU_FUSED` forward,
  online softmax, parity vs reference; optional logsumexp output.
- [ ] **G4 — Block-sparse SDPA**: `block_mask` value + host pattern builders
  (causal, sliding-window, custom); kernel visits allowed blocks; parity vs
  dense-masked reference. (Forward-first; sparse backward deferred.)
- [ ] **G5 — GPT model assembly**: `nn.h` builders (attention block, SwiGLU/GELU
  MLP, full `gd_gpt`), weight tying, RMSNorm pre-norm. Tiny-GPT training parity
  (CPU↔Metal) + overfit sanity.
- [ ] **G6 — KV cache + decode**: `kv_append`, prefill/decode graphs,
  `past_len` runtime scalar, host generation driver; prefill+decode == full
  forward parity. `examples/gpt/` greedy generation on CPU and Metal.

Acceptance per phase mirrors §10. `make check` stays green; Metal stays
optional/graceful where unavailable.

---

## 13. Open questions / decisions to confirm

1. **Multi-output IR nodes.** `sdpa_bwd` yields `dq,dk,dv`. The current graph
   favors single-output nodes (`_gd_graph_emit`) with in-place/`emit_to`
   variants; the optimizer-style multi-input in-place exists. **Proposal:** emit
   three nodes (dq, dk, dv) that each recompute or share a cheap precomputed
   stat, or add a genuine multi-output emit. Decision needed before G2 backward.
2. **`past_len` plumbing.** Tiny per-step int upload vs a dedicated
   `kv_append`-returned length vs an sdpa param patched in place. **Proposal:** a
   1-element int32 tensor `k_len` uploaded each decode step; sdpa reads it.
   Confirm at G6.
3. **Block mask representation.** CSR-of-blocks int32 `[n_qb, max_kb]` (chosen)
   vs bitmap `[n_qb, n_kb]`. CSR is better for high sparsity; bitmap simpler.
   Confirm at G4.
4. **Transpose op generality.** Full `permute` vs just the 2-axis swaps
   attention needs. **Proposal:** general `permute` (perm array) — small extra
   cost, broadly useful.
5. **RoPE variant default.** NeoX half-split (default) vs GPT-J interleaved;
   expose both via `interleaved` flag (chosen).
6. **SDPA backward memory.** Reference O(T²) recompute (v1) vs tiled FA-2
   backward (later). Confirmed: reference first.

---

## 14. Out of scope (explicitly deferred)

- Sparse-attention backward (training with block sparsity); v1 sparse is
  forward/inference-only.
- bf16/fp16/fp8 and quantized attention/KV kernels (fp32 first).
- Paged/virtualized KV cache and cross-request batching/serving.
- Fused QKV+RoPE+SDPA+out-proj single kernel (kept as a later fusion rung; the
  `sdpa` node is the seam where it will land).
- Tensor/sequence/pipeline parallelism; multi-device.
- Speculative/medusa decoding, MoE, alternative positional schemes (ALiBi,
  learned), encoder-decoder/cross-attention.

## 15. Risks

- **Attention numerics drift** (online softmax, GPU fp32 accum vs CPU fp64):
  mitigate with stable algorithms, 1e-4 tolerance, and the multi-implementation
  parity ladder (§10.4).
- **Static-shape vs growing KV**: the `k_len` scalar + `max_seq` allocation is
  the crux; G6's prefill+decode==full-forward test is the gate.
- **Block-sparse correctness leaks** (future tokens visible): the kernel always
  applies the fine-grained causal/window predicate within visited blocks, so a
  too-coarse block list can never leak; tested in §10.5.
- **Multi-output backward** ergonomics (sdpa): resolve decision §13.1 before G2.
- **Transpose on virtual tensors**: resolved by adding a real `transpose` op and
  preferring the transpose-free head-major path (§8).
```
