# gradients.c — Prefix-Causal SDPA for VLM Attention

Status: implemented v1

Goal: support VLM-style self-attention where a fixed prefix block (image tokens)
attends bidirectionally within the prefix, while suffix tokens (text) attend
causally to prefix + previous/current suffix tokens. Do this efficiently in the
existing tiled Metal SDPA path, not via dense additive masks.

Target layout:

```text
sequence = [ image/prefix tokens ][ text/suffix tokens ]
T        = prefix_len + text_len
```

Mask semantics:

```text
qpos = query position in full KV sequence
j    = key position

if qpos < prefix_len:
    allow j < prefix_len          # image sees image, not text
else:
    allow j <= qpos               # text sees all image + prior/current text
```

This is the common LLaVA/Qwen-VL-style prefix mask: image/image is bidirectional,
text/text is causal, text can see all image tokens.

References:
- [`docs/plan_gpt.md`](plan_gpt.md) §6/G3 for current SDPA kernels.
- [`docs/plan_block_sparse_sdpa_metal.md`](plan_block_sparse_sdpa_metal.md) for
  causal block-skip and future general sparse masks.

---

## 1. Current state

`gd_sdpa` currently supports:

```c
typedef struct gd_sdpa_config {
    float scale;        /* 0 => 1/sqrt(head_dim) */
    bool  causal;
    int   sliding_window; /* 0 => none */
} gd_sdpa_config;
```

and optional additive bias:

```text
bias [B, Hq, Tq, Tk], broadcastable
```

Correct prefix-causal attention can be expressed today with `causal=false` and a
large-negative additive bias:

```text
bias[i,j] = allowed(i,j) ? 0 : -1e9
```

but this is not efficient: kernels still scan dense `Tq*Tk` pairs. Fully-masked
text keys for image queries still burn compute and memory bandwidth. Bias storage
can be broadcast/small, but compute stays dense.

Desired change: make prefix-causal a native SDPA mask mode so tiled kernels skip
fully-dead blocks, like causal block-skip.

---

## 2. Public API

Minimal API extension: append `prefix_len` to `gd_sdpa_config`.

```c
typedef struct gd_sdpa_config {
    float scale;          /* 0 => 1/sqrt(head_dim) */
    bool  causal;
    int   sliding_window; /* 0 => none */
    int   prefix_len;     /* >0 with causal=true => prefix-causal */
} gd_sdpa_config;
```

Semantics:

- `causal=false, prefix_len=0`: dense/bidirectional.
- `causal=true, prefix_len=0`: standard causal.
- `causal=true, prefix_len>0`: prefix-causal.
- `causal=false, prefix_len>0`: invalid in v1. Return `GD_ERR_INVALID_ARGUMENT`.

`prefix_len` is counted in key/value sequence coordinates. For normal training
prefill, `Tq == Tk == T`, so query `i` has `qpos=i`. For decode/paged-style
graphs where `Tq < Tk`, keep the existing causal offset convention:

```text
qpos = i + (Tk - Tq)
```

Validation:

```text
0 <= prefix_len <= Tk
if Tq == Tk: prefix_len <= Tq
if Tq < Tk: prefix_len may be <= Tk; queries may all be suffix queries
```

Sliding-window composition:

- v1: allow `sliding_window=0` only when `prefix_len>0`.
- later: text/text window can apply only to suffix keys, while prefix keys remain
  globally visible:

```text
if qpos >= prefix_len && j >= prefix_len:
    require qpos - j < sliding_window
prefix keys j < prefix_len remain visible to all suffix queries
```

Additive bias still composes with prefix-causal: bias is added only for positions
that pass the structural mask.

---

## 3. Internal attrs / params

Add to `_gd_op_attrs` in `src/graph/graph_internal.h`:

```c
int prefix_len; /* SDPA: bidirectional prefix length for causal mask */
```

Add to `gd_metal_sdpa_params` in `src/backends/metal/metal_kernel_types.h`:

```c
int prefix_len;
```

Set in `gd_sdpa()` emission:

```c
attrs.prefix_len = config ? config->prefix_len : 0;
```

Set in `fill_sdpa_params()`:

```c
p->prefix_len = node->attrs.prefix_len;
```

Existing graphs with zero-initialized config retain old behavior.

---

## 4. Core predicate

Replace current predicate:

```c
static inline bool gd_sdpa_allowed(int i, int j, int Tq, int Tk,
                                   int causal, int window)
```

with prefix-aware form:

```c
static inline bool gd_sdpa_allowed(int i, int j, int Tq, int Tk,
                                   int causal, int window, int prefix_len)
{
    int qpos = i + (Tk - Tq);

    if (causal) {
        if (prefix_len > 0) {
            if (qpos < prefix_len) {
                if (j >= prefix_len) return false;
            } else {
                if (j > qpos) return false;
            }
        } else {
            if (j > qpos) return false;
        }
    }

    if (window > 0) {
        if (prefix_len > 0) {
            /* v2 semantics: keep prefix globally visible, window suffix only. */
            if (qpos >= prefix_len && j >= prefix_len && (qpos - j) >= window) {
                return false;
            }
        } else if ((qpos - j) >= window) {
            return false;
        }
    }
    return true;
}
```

For v1, reject `window>0 && prefix_len>0`, so kernel branch exists but is not
used until tests land.

CPU_REF and Metal must use identical predicate semantics.

---

## 5. Block-skip bounds

The tiled Metal kernels already use coarse range helpers to avoid fully masked
blocks. Make them prefix-aware.

Definitions:

```text
BQ = GD_METAL_SDPA_BQ
BK = GD_METAL_SDPA_BK
q0 = first query index in query block
qmax = min(q0 + BQ - 1, Tq - 1)
off = Tk - Tq
q0pos = q0 + off
qmaxpos = qmax + off
```

### 5.1 Forward / stats / dq: key upper bound

Current causal bound returns exclusive key limit for a query block. Prefix-causal
bound:

```c
static inline int gd_sdpa_kb_end_prefix(int q0, int bq,
                                        int Tq, int Tk,
                                        int causal, int prefix_len)
{
    if (!causal) return Tk;

    int qmax = q0 + bq - 1;
    if (qmax > Tq - 1) qmax = Tq - 1;
    int qmaxpos = qmax + (Tk - Tq);

    if (prefix_len > 0) {
        if (qmaxpos < prefix_len) {
            return min(prefix_len, Tk);      /* prefix query block */
        }
        return min(qmaxpos + 1, Tk);         /* suffix or mixed block */
    }

    return min(qmaxpos + 1, Tk);             /* standard causal */
}
```

Mixed block crossing `prefix_len` may scan extra suffix keys for prefix queries;
fine-grained `gd_sdpa_allowed()` masks them. This keeps barriers uniform and code
simple. Only one boundary block pays this small overhead.

### 5.2 dkv: query lower bound

For `dk/dv`, each key block visits query blocks that may attend to it. Prefix
keys are visible to all queries, so key blocks starting before `prefix_len` must
start at query block 0. Text key blocks can skip query blocks whose global query
position is below the key.

```c
static inline int gd_sdpa_qb_start_prefix(int k0, int qblk,
                                          int Tk, int Tq,
                                          int causal, int prefix_len)
{
    if (!causal) return 0;
    if (prefix_len > 0 && k0 < prefix_len) return 0;

    int qmin = k0 - (Tk - Tq);
    if (qmin <= 0) return 0;
    if (qmin > Tq) return Tq;
    return (qmin / qblk) * qblk;
}
```

Again, per-element predicate handles boundary blocks exactly.

### 5.3 Sliding-window skip

V1 can defer prefix+window. If implemented later, block bounds become:

```text
prefix keys: always visible to suffix queries
suffix keys: apply causal/window bounds within suffix region
```

Need both lower and upper key bounds per query block, not only upper. This is
covered by block-sparse plan; do not block prefix-causal v1 on it.

---

## 6. Metal kernels touched

Update predicate calls and block bounds in:

- `gd_sdpa_tiled`
- `gd_sdpa_splitk`
- `gd_sdpa_combine` only needs params layout; no predicate
- `gd_sdpa` reference fallback
- `gd_sdpa_bwd_stats`
- `gd_sdpa_bwd_dq`
- `gd_sdpa_bwd_dkv`
- split-K backward kernels:
  - `gd_sdpa_bwd_stats_dq_split`
  - `gd_sdpa_bwd_stats_dq_combine` only params/layout
  - `gd_sdpa_bwd_dkv_split`
  - `gd_sdpa_bwd_dkv_reduce` only params/layout

Important barrier rule: do not branch per-thread around `threadgroup_barrier()`.
Use block-uniform loop bounds (`kb_end`, `qb_start`) and keep per-element masks
inside visited blocks.

---

## 7. Split-K notes

Forward split-K partitions key range. Prefix-causal changes each query block's
valid key upper bound, but split ranges can remain global:

```text
split s scans [s*split_len, min((s+1)*split_len, kb_end))
```

If split start is >= `kb_end`, write an empty partial:

```text
m = -inf, l = 0, acc = 0
```

Combine already handles empty partials if written consistently.

Backward split-K:

- stats/dq split uses same per-query `kb_end`.
- dkv split uses prefix-aware `qb_start`.
- partial reduce kernels do not need mask logic beyond new params layout.

This avoids allocating per-query-block schedules and keeps compile-time scratch
unchanged.

---

## 8. CPU_REF changes

CPU kernels in `src/backends/cpu_ref/cpu_kernels.c` use `sdpa_allowed()` for both
forward and backward. Add `prefix_len` arg to:

- `_gd_cpu_k_sdpa`
- `_gd_cpu_k_sdpa_bwd`

and call sites in `src/backends/cpu_ref/cpu_ref.c`.

CPU_REF stays dense and slow. It is correctness oracle, not perf path.

---

## 9. Shape / validation

In `gd_sdpa()` or shape inference:

```text
prefix_len >= 0
prefix_len <= k.size[1]
if prefix_len > 0:
    causal == true
    sliding_window == 0       # v1 restriction
```

Bias validation unchanged:

```text
bias broadcastable to [B,Hq,Tq,Tk]
```

Bias with prefix-causal means:

```text
if structural mask disallows (i,j): ignore position entirely
else score += bias(i,j)
```

---

## 10. Tests

Add CPU op tests:

1. Small prefix mask forward:

```text
B=1, T=5, prefix_len=2, H=1, Dh=4
```

Compare native prefix-causal vs dense bias mask with `causal=false`.

2. Backward grad parity:

```text
q/k/v require grad
prefix_len in {1, 2, T-1}
compare native prefix-causal vs dense bias mask
```

3. CPU↔Metal graph compare:

```text
T=16, prefix_len=5
T=130, prefix_len=33        # crosses multiple BQ blocks
T=600, prefix_len=257       # split-K path
```

4. Boundary cases:

```text
prefix_len=0     => identical to causal
prefix_len=1
prefix_len=T     => prefix-only bidirectional block; no text
prefix_len=T-1   => one text token
```

5. VLM-layout behavior test:

```text
image query cannot attend text key
text query can attend all image keys
text query cannot attend future text key
```

Use hand-constructed logits/value rows where forbidden positions would visibly
change output if included.

---

## 11. Benchmark plan

Canonical benchmark variants:

```bash
GD_METAL_MPS=1 GD_DEVICE=metal GD_BENCH_B=8 GD_BENCH_T=1024 make gpt-bench
```

Add VLM-specific benchmark knobs later:

```text
GD_BENCH_PREFIX=256
GD_BENCH_MASK=prefix
```

Compare three modes:

1. Dense causal baseline (`prefix_len=0`, `causal=true`).
2. Prefix-causal native (`prefix_len=M`, `causal=true`).
3. Dense bias mask (`causal=false`, bias [1,1,T,T]).

Expected attention pair counts for `T=M+N`:

```text
dense bidir       T*T
causal            T*(T+1)/2
prefix-causal     M*M + N*M + N*(N+1)/2
```

Example `M=256, N=768, T=1024`:

```text
dense bidir       1,048,576 pairs
causal              524,800 pairs
prefix-causal       557,440 pairs
```

So native prefix-causal should be close to causal cost, not dense cost:

```text
prefix-causal / causal ≈ 1.06x pairs
prefix-causal / dense  ≈ 0.53x pairs
```

Acceptance target:

```text
native prefix-causal attention time <= 1.15x standard causal
native prefix-causal attention time <= 0.65x dense bias mask
```

for `M=256,N=768,Dh=64,H=5,B=8`, allowing launch/shape noise.

---

## 12. Phases

### P0 — CPU + IR surface

- Add `prefix_len` to config, attrs, CPU kernels.
- Validate inputs.
- Add CPU tests vs dense bias mask.

### P1 — Metal single-pass tiled path

- Add `prefix_len` to Metal params.
- Update `gd_sdpa_allowed` and block bounds.
- Validate `gd_sdpa_tiled`, fallback `gd_sdpa`, and non-split backward kernels.

### P2 — Metal split-K path

- Update forward split-K and backward split-K kernels.
- Validate long-context tests (`T>=600`, prefix crossing split boundary).

### P3 — GPT/VLM integration hooks

- Add optional `prefix_len` path in model builders once VLM model API exists.
- Add benchmark knobs for prefix masks.

### P4 — Optional prefix+window

- Let suffix tokens use sliding window over suffix while keeping prefix globally
  visible.
- Needs lower-bound key skip in forward/stats/dq plus extra tests.

---

## 13. Non-goals

- No general arbitrary masks in this task. Dense additive bias remains correctness
  fallback; block-sparse schedule belongs to `plan_block_sparse_sdpa_metal.md`.
- No learned bias gradients. Existing SDPA bias is constant/non-differentiable.
- No concat/split API for VLM tokens. This plan only specifies attention mask
  semantics once q/k/v are already built over `[prefix][suffix]` sequence.
- No packed/paged KV-cache design. Existing `Tq<Tk` causal offset is preserved.

---

## 14. Summary

Current additive bias can make VLM prefix attention correct, but it is dense.
Native `prefix_len` in SDPA gives the exact VLM mask while preserving block-skip:
image queries scan only image keys; text queries scan prefix + causal text. For
common `M=256,N=768`, work is ~6% above causal and ~47% below dense bidirectional
attention.
