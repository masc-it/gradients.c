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

- [ ] F0 — **Fusion infrastructure** (§4 items 1–4, 6, 8–9) as an identity pass:
  consumers map, legality predicate scaffold, `node_absorbed`/`node_fused_kind`
  in the executable, step-aware `execute`, and the parity + fusion-fired test
  harness. No patterns yet → must not change any result or timing.
- [ ] F1 — **First kernel: SwiGLU `silu`+`mul`** (infra shakedown, small kernel).
  `act=silu(gate)` is single-consumer (`mul`); fuse forward into one kernel
  `hh = silu(gate)·up`. Backward needs `silu(gate)` and `up` → **recompute
  `silu(gate)`** in the (modified) backward so `act` is never materialized.
  Proves the whole pipeline end-to-end with low kernel risk.
- [ ] F2 — **Headline kernel: `linear`→`cross_entropy`** (LM head). Logits are
  forward-only (no backward reads them) → drop the `[B,T,V]` materialization;
  forward streams the loss, backward recomputes `softmax` to form `dlogits`.
  Biggest single tail item (~50 ms incl. bwd).
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
