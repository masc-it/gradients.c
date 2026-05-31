# gradients.c — Metal GPU_FUSED (kernel fusion)

Status: draft v0.2

Goal: cut the bandwidth-bound, dispatch-heavy "tail" of the GPT step by fusing
adjacent IR nodes into single Metal kernels. Fusion is a **Metal-backend lowering
pass** (Option A): the canonical IR stays op-granular (CPU_REF runs it node by
node as the oracle), and only Metal's `compile` recognizes a pattern and emits a
fused kernel. No general planner — **targeted peephole fusions** only, added one
at a time, each with an unfused fallback.

This doc is scoped to *writing the first fused kernels*: §4 is the concrete list
of what the codebase needs before any fused kernel can land; §5 is the phased
kernel work.

References:
- [`docs/design_spec.md`](design_spec.md) §17 (backend lowering), §18 (fusion/IR,
  "compiler may avoid materializing virtual tensors").
- [`plan_gemm_perf_tuning_metal.md`](plan_gemm_perf_tuning_metal.md) §1.1 (tail
  breakdown / Amdahl).
- `src/graph/graph_internal.h` (`_gd_node`, `_gd_value`, `producer_node_id`);
  `src/backends/metal/metal_backend.m` (`compile` / `execute` / `encode_node`);
  `src/nn/nn.c` (`attention_block` / `mlp_block` — the op chains below).

---

## 1. Why (measured)

Canonical step (12.39M GPT, B=4, T=256, 6 layers, `GD_PROFILE=trace`, ~483 ms):
GEMM ~121 ms (25%), attention ~181 ms (38%, at its fp32 floor), **tail ~180 ms
(37%)**. The tail is bandwidth-bound elementwise/reduction ops + the LM-head/CE
region, spread over ~500 dispatches. Fusion cuts both device round-trips and
dispatch count.

The GPT op chains we fuse against (`nn.c`):

```
attn: n=rms_norm(x,ln1) -> linear(wq)|linear(wk)|linear(wv) (all read n)
      -> rope(q),rope(k) -> sdpa -> linear(wo) -> add(x,.)
mlp:  n=rms_norm(h,ln2) -> linear(w_gate)|linear(w_up) (both read n)
      -> silu(gate) -> mul(.,up) -> linear(w_down) -> add(h,.)
head: rms_norm(x,ln_f) -> linear(w_lmhead) -> cross_entropy
```

`reshape`s are already metadata-only `_GD_OP_COPY` and zero-copy (P7) — not
targets.

---

## 2. The governing constraint: activations are pinned by backward

In a joint fwd+bwd graph, backward nodes re-read forward activations, so a
forward intermediate is usually **not** single-consumer and cannot be dropped.
The legality test for *not materializing* a value `v` is: **every consumer of `v`
(forward and backward) is inside the fused group.** Three outcomes per pattern:

- **Drop free** — `v` is forward-only (no backward reads it). Best case.
- **Recompute** — drop `v`, recompute it in the fused backward from inputs that
  are still live. Trades a little ALU for the store/reload.
- **Must-store** — `v` has a live consumer outside the group (e.g. the residual
  stream); fusion only saves a *reload*, not the store. Partial win.

Consequence: consumers analysis must run over the **whole** graph, and the unit
of fusion for training is often the (forward + matching backward) pair.

---

## 3. Design (Option A recap)

- IR unchanged and shared; CPU_REF runs the unfused nodes = oracle.
- Metal `compile` runs a small **peephole recognizer**; a match binds a group of
  nodes to one fused pipeline, marks the absorbed node(s), and (optionally) drops
  the buffer for the fusion-internal value.
- `execute` iterates the node list, skips absorbed nodes, and the group's head
  node encodes the fused kernel.
- Parity is guaranteed at **group boundaries** (graph outputs + any value with an
  outside consumer); fusion-internal values may not exist on Metal.

> Out of scope here (separate, model-layout track): **fused QKV / gate-up
> projection** — concatenating `Wq|Wk|Wv` (and `Wgate|Wup`) into one weight so the
> three/two sibling GEMMs that read `n` become one larger GEMM. That is the
> single biggest GPT structural win, but it is a *weight-layout change at model
> construction* (`gd_gpt_create` / `gd_gpt_forward`), not a fused kernel, and
> needs none of the machinery below. Track it in the GPT model plan, do it first.

---

## 4. What we need before the first fused kernel (the prerequisites)

This is the minimal infrastructure (phase F0). None of it is pattern-specific.

**Compile-time (in `metal compile`, before per-node pipeline resolution):**
1. **Consumers map.** Reverse of `producer_node_id`: for each value, the list of
   consuming `(node, input-slot)`. Build once by scanning `graph->nodes`. Needed
   for legality and for "is this value still live outside the group".
2. **Legality predicate.** `fusable(producer, consumer)` =
   - `consumer` reads exactly the `producer` output(s) we intend to absorb, and
   - the absorbed value's consumers are all inside the group (per §2), and
   - same device/dtype, and shapes the fused kernel supports.
   Otherwise: no fusion (unfused path, always correct).
3. **Peephole recognizer.** For the chosen pattern, walk nodes and apply the
   predicate; emit a *fused step* = `{head node id, absorbed node ids[], fused
   kind, external input value ids, surviving output value ids}`.
4. **Executable annotation.** Add to `_gd_executable`:
   - `uint8_t *node_absorbed` (execute skips these), and
   - per-head fusion info: `int *node_fused_kind` (0 = none) plus, if the fused
     kernel needs extra pipelines, reuse `node_pso2/pso3`.
5. **(Optional, deferrable) non-allocation.** Skip the storage buffer for a
   dropped fusion-internal value. *Not required for the first cut* — leaving the
   buffer allocated-but-untouched is correct; the bandwidth win already comes from
   the fused kernel not reading/writing it. Add this once correctness is proven.

**Execute-time:**
6. `execute` loop: `if (node_absorbed[i]) continue;` and, for a fused head,
   dispatch the fused `encode_*` instead of the per-op one.

**Kernel-side (per pattern):**
7. New `.metal` kernel(s) (fwd, and bwd or recompute), shared param struct in
   `metal_kernel_types.h` if needed, registered in the op→kernel map or
   `g_metal_extra_kernels`, and an `encode_fused_*` binding inputs/outputs.

**Validation / observability (must-have, not optional):**
8. **Parity**: a test graph that triggers the pattern, checked with
   `gd_graph_compare(CPU, METAL)` at boundaries (1e-4). Plus a direct unit test:
   fused-kernel output vs the unfused composition.
9. **"Fusion fired" assertion.** Because the fallback is silent, add a way to
   confirm the fusion actually engaged (e.g. a profiler event / debug counter, or
   the fused step shows up under a fused op name in `GD_PROFILE=trace`). A test
   that expects the fused kernel must fail if it silently fell back.

F0 ships with the recognizer disabled (no patterns) to prove the plumbing is a
no-op on correctness and perf before any kernel exists.

---

## 5. Phases

- [x] F0 — **Fusion infrastructure (identity pass).** Landed the executable
  plumbing: `_gd_executable.node_absorbed` (per-node skip flag, calloc'd/freed),
  the `metal_plan_fusions(self, graph, exe)` hook called at end of `compile`
  (no patterns yet → leaves `node_absorbed` all-zero), and the skip check
  (`if (node_absorbed[i]) continue;`) in **both** execute paths (normal and the
  `GD_PROFILE=trace` per-node path). Validated: `make check` green, ASan-clean,
  T=256 step 408 ms / 2499 tok/s unchanged — a true no-op, as required.
  Deferred to the first pattern (F1): the consumers map, the legality predicate,
  `node_fused_kind` + the fused-encode dispatch branch, and the fusion-fired
  assertion (nothing to observe until a pattern exists).
- [x] F1 — **SwiGLU `silu`+`mul`** (infra shakedown). Landed: `gd_silu_mul`
  kernel (writes `hh = silu(gate)·up` **and** `act = silu(gate)`), the consumers
  analysis (`value_min_consumer`), the `try_fuse_silu_mul` legality predicate,
  `node_fused_src` + the fused-encode dispatch in both execute paths, and a
  white-box `_gd_metal_fusions_applied()` counter + a parity/fired test
  (`test_swiglu_fusion`). **Kept `act` materialized** (so the unfused backward is
  unchanged) rather than recompute — simplest correct first kernel.
  Validated: 6 fusions in the 6-layer GPT, CPU↔Metal parity 1e-4, GPT train
  parity, ASan-clean. **Perf: neutral / within noise** — `silu` fwd ~3 ms -> ~0
  (absorbed into `mul`), 6 fewer dispatches, step 408 ms unchanged (~0.4%). The
  value is the proven pipeline; the perf payoff is F2. Dropping `act` (recompute
  in a fused backward) is deferred — it needs a fused backward and buys little.
- [ ] F2 — **`matmul`/`linear`→`cross_entropy`** (LM head). Investigated; scope is
  bigger than first framed (it is a cut-cross-entropy kernel). Key findings:
  - The logits claim "forward-only" is **false in training**: `logits` is
    consumed by both `cross_entropy` (fwd) **and** `cross_entropy_bwd` (which
    recomputes softmax). So a forward-only fusion is illegal in a training graph
    (the legality gate refuses; it would only fire for eval/inference graphs).
  - The real training win is eliminating **`dlogits[N,V]`** in the backward
    (~32 ms at T=256; ~131 MB / much larger at T=512,B=8). That needs the head
    grad matmuls (`d_normed = dlogits@wte`, `d_wte = dlogits^T@normed`) to
    **recompute `dlogits` in-tile** from materialized `logits` + per-row softmax
    stats — i.e. a cut-cross-entropy backward. `d_wte` is the hard part
    (cross-token reduction; can't drop `dlogits` without a tiled recompute GEMM).
  - A naive row-oriented fused forward (one threadgroup/row, GEMV + online
    softmax) would **regress** the forward, because it discards the efficient
    tiled reg-blocked GEMM. A non-regressing fused forward must itself be tiled.
  Conclusion: F2 is a substantial, specialized, tiled kernel project (fwd + the
  harder cut-CE bwd), not an F1-style increment. Split into F2a/F2b below.
- [ ] F2a — *(optional, eval-only)* tiled fused `matmul`→`cross_entropy` forward
  that streams the loss without materializing logits. Legality gates it to graphs
  where `logits` has a single consumer (the CE) → fires for eval, safely falls
  back in training. Must be tiled to avoid regressing the GEMM.
- [ ] F2b — **cut-cross-entropy backward** (the training win): per-row softmax
  stats + recompute `dlogits` inside the two head-grad matmul tiles, dropping the
  `[N,V]` `dlogits` materialization. The high-value, high-effort core of F2.
- [ ] F3 — *(later, if warranted)* RoPE folded into SDPA q/k staging.
- [ ] F4 — *(later, partial)* residual `add` + `rms_norm` reload-merge.

Each phase: parity (`gd_graph_compare` boundaries, 1e-4) + GPT train parity +
ASan + fusion-fired assertion; report before/after `GD_PROFILE=trace` per-op/step
ms, dispatch count, and end-to-end step (B=4, T=256 and T=1024).

---

## 6. Contract notes

- Metal-only; IR + CPU_REF unchanged; no new public ops.
- Conservative legality + always-correct unfused fallback.
- Fused reductions (CE/softmax, any norm) accumulate in fp32 to match CPU_REF at
  1e-4.
- Trace stays legible: a fused step reports under a fused op name.

---

## 7. Reporting rule

Record per phase: before/after per-op/step `gpu_ms` + dispatch count + end-to-end
step (T=256 and T=1024); non-trivial choices (legality conditions, store-vs-
recompute, accumulation dtype); learnings (where the win did/didn't show,
lifetime constraints hit); validation (boundary parity, GPT train parity, ASan,
fusion fired).

---

## 8. Benchmark table

| Phase | T | tail ms | dispatches | step (real) | tokens/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 256 | ~180 | ~500 | 408 ms | 2510 | post GEMM+attention work |
| baseline | 1024 | ~190 | ~500 | 2965 ms | 1381 | |

(Fill as F0–F4 land.)

---

## 9. Risks

- **Activation lifetime** (§2): most forward intermediates can't be dropped.
  Mitigation: whole-graph consumers analysis; target forward-only (CE) first;
  recompute (SwiGLU); accept partial wins (residual+norm).
- **Silent fallback hides regressions.** Mitigation: §4.9 fusion-fired assertion;
  tests fail if the pattern didn't engage.
- **Shrunken parity surface.** Fusion-internal values aren't materialized.
  Mitigation: boundary compare + per-fused-op unit tests vs unfused composition.
- **Numerics.** fp32 accumulation in fused reductions to hold 1e-4.
- **Compiler surface / mis-fusion.** Mitigation: one hardcoded pattern at a time,
  conservative predicate, unfused fallback always correct; F0 proves the no-op.
- **Amdahl.** Tail is ~37%; halving it is ~1.2× end-to-end. Report end-to-end.
