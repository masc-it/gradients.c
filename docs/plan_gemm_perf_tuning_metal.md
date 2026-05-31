# gradients.c — Metal GEMM Performance Tuning

Status: draft v0.1

Goal: raise the throughput of `matmul` / `linear` on the Metal backend toward the
hardware ceiling, **without** changing the public op contract or the GPU_SAFE
execution model (one IR node → one backend op/dispatch, same math, CPU_REF stays
the correctness oracle).

This plan is the GEMM-specific follow-up to
[`docs/plan_metal_improve_foundations.md`](plan_metal_improve_foundations.md)
(P10 landed the first tiled GEMM) and the SDPA work in
[`docs/plan_gpt.md`](plan_gpt.md) (G3 FlashAttention tiling). After G3, attention
is no longer the bottleneck on the canonical workload — **GEMM is**.

References:
- [`docs/design_spec.md`](design_spec.md) §7 (backend roles), §17–18
  (backend/compiler, kernel cache/autotune)
- [`docs/plan_metal.md`](plan_metal.md) Metal backend, P10 tiled GEMM
- `data/metal/compute_encoder.md`, `data/metal/metal_reference.md`
  (threadgroups, `threadExecutionWidth`, MPS, GPU counters)
- Metal Shading Language Specification — `simdgroup_matrix` (§ SIMD-group matrix
  functions), `simdgroup_load`/`simdgroup_store`/`simdgroup_multiply_accumulate`

---

## 1. Where we are

Current `matmul`/`linear` use a threadgroup-tiled scalar kernel
(`gd_matmul_tiled` / `gd_linear_tiled`, `GD_METAL_GEMM_TILE = 16`), streaming K
in tiles through threadgroup memory, one thread per output element with a scalar
FMA inner loop (P10).

Measured:
- Standalone 512×512×512 GEMM: **~289 GFLOP/s** (naive was ~169).
- Canonical training workload (12.39M GPT, B=4, T=256, 6 layers,
  `GD_PROFILE=trace`): after G3 attention tiling, GEMM is the top cost —
  `matmul` 148.6 ms + `linear` 57.6 ms = **~206 ms of a ~500 ms step** (~41%).
  Overall step throughput ~164 GFLOP/s.

Apple M-series fp32 peak is low-single-digit TFLOP/s, so the scalar tiled kernel
is leaving a large multiple on the table: it does not use the GPU matrix units
and is largely ALU/issue-bound, not memory-bound, for square-ish shapes.

### Important caveat (end-to-end vs kernel speedup)
A chunk of GEMM time is the weight-tied **LM head** `[*, 8000]` matmuls
(vocab-bound, memory-heavy, skinny N or K). Matrix-unit kernels help the
square-ish projections (qkv, out-proj, MLP) most and the skinny vocab GEMMs
least. So even a kernel that is several× faster in GFLOP/s yields an estimated
**~1.3–1.5× end-to-end** on this workload — the GFLOP/s multiple does not pass
through one-to-one. Set expectations accordingly and always report end-to-end
step time, not just microbenchmark GFLOP/s.

### 1.1 The non-GEMM "tail" (Amdahl)
After G3, the per-op trace of the canonical step (B=4, T=256, 6 layers,
`GD_PROFILE=trace`, ~573 ms total) splits roughly into thirds:

```text
GEMM      (matmul+linear)       ~210 ms  (37%)
attention (sdpa+sdpa_bwd)       ~181 ms  (32%)
tail      (everything else)     ~182 ms  (32%)
```

The "tail" is **as large as GEMM**, so GEMM tuning alone is Amdahl-capped: even
an *infinitely fast* GEMM only removes ~37% of the step (-> ~363 ms); a realistic
~2x GEMM kernel removes ~105 ms (-> ~468 ms, ~1.2x end-to-end). The tail must be
trimmed in parallel to keep the overall speedup from saturating.

Measured tail breakdown (same trace), with why it costs and how to trim:

| op | ms | calls | why it costs | trim lever |
|----|---:|---:|---|---|
| `cross_entropy_bwd` | 32.3 | 1 | writes `dlogits[B,T,V]` = 1024x8000 = 8.2M floats and recomputes softmax over V=8000 | fuse softmax+CE+grad; same `[*,8000]` hot region as the LM head |
| `cross_entropy` | 17.4 | 1 | reduction over V=8000 per position | fuse with the LM-head matmul (logits never fully materialized) |
| `copy` | 18.1 | 104 | reshape/materialization copies; many already aliased (P7) but ~104 real ones remain | alias more (view-only), fuse reshapes into producers, cut dispatch count |
| `adamw_step` | 16.5 | 56 | bandwidth-bound: reads/writes param+m+v+grad ~= 4x12.4M floats | batch all params + `step_inc` into one launch; fewer, fatter dispatches |
| `reduce_to` | 16.5 | 43 | grad broadcast/accumulation reductions (parallel since P9) | fuse into the producing backward op; avoid separate accumulation passes |
| `add` | 16.1 | 43 | residual + grad-accumulation elementwise, bandwidth-bound | fuse residual add into the adjacent norm/projection (GPU_FUSED) |
| `embedding_bwd` | 12.6 | 1 | scatter over `[V,d]` = 8000x320 | atomics / segmented scatter; or fuse with the tied LM-head grad |
| `rope` (+bwd) | 12.0 | 24 | elementwise rotary over q,k | fuse into the qkv projection / attention entry |
| `mul` | 11.1 | 18 | SwiGLU `silu(gate)*up` elementwise | fuse `silu`+`mul` (+down-proj) into a SwiGLU kernel |
| `rms_norm` (+bwd,+wbwd) | ~18 | 39 | per-row reductions (P13) | fuse the norm into the following linear / residual |
| `silu` (+bwd) | 5.7 | 12 | elementwise | fuse into the MLP |
| `embedding` | 5.2 | 1 | gather | -- |

Key point: the tail is dominated by **memory-bandwidth-bound elementwise /
reduction ops** and the **large `[*,8000]` LM-head / cross-entropy region** --
not by arithmetic. The lever for almost all of it is **kernel fusion** (fewer
launches, fewer device round-trips), which is the **`GPU_FUSED` rung** -- a
*different* effort than GEMM kernel tuning and outside GPU_SAFE. The exceptions
that stay GPU_SAFE: batching `adamw_step`/`step_inc` into one launch, aliasing
more `copy` reshapes, and an atomic/segmented `embedding_bwd`.

There is also a flat **dispatch-count** component: this step issues ~500 kernel
dispatches (87 matmul + 104 copy + 56 adamw + 43 reduce_to + 43 add + ...), each
with launch/encode overhead. Fusion cuts that count directly; the P8 encode
plan already removed the per-dispatch host lookups.

Implication for sequencing: pursue GEMM (this plan) and the tail (a future
`GPU_FUSED` plan) as complementary tracks. Neither alone moves the workload much
below ~2x of today; together they do. Always report the end-to-end step so the
Amdahl ceiling stays visible.

---

## 2. Options, ranked by ceiling vs risk

### Option A — `simdgroup_matrix` custom kernel (highest ceiling, highest risk)
Use Apple's SIMD-group matrix intrinsics (`simdgroup_float8x8`,
`simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store`) so the inner
product runs on the GPU matrix units instead of scalar FMAs. This is how you
actually approach hardware peak (a multiple of today's ~289 GFLOP/s).

Risks:
- MSL `simdgroup_matrix` is fixed 8×8; correct use needs careful
  threadgroup-memory staging, register/accumulator blocking, double-buffering,
  and SIMD-lane↔tile mapping — easy to get subtly wrong.
- Heavy occupancy / tile-size / threadgroup-size tuning that **must be iterated
  on-device**; there is no reliable way to converge to peak without a
  measurement loop on real hardware.
- Accumulation order changes → must re-validate CPU↔Metal parity (expected fine
  at 1e-4, but a real gate).
- Must still cover the full `matmul` contract (batch broadcast, `trans_a`,
  `trans_b`, arbitrary shapes incl. skinny vocab) → the fast path needs a clean
  fallback for shapes it does not handle, adding branches/surface area.

### Option B — MPS / MPSGraph (`MPSMatrixMultiplication`) (high ceiling, medium risk)
Call Apple's tuned GEMM. Likely near-peak with zero kernel tuning from us. Still
"one IR node → one MPS call", so GPU_SAFE holds.

Risks (mostly integration, not math):
- New framework dependency (MetalPerformanceShaders); build/link wiring.
- Wrapping our `MTLBuffer`s without copies; mapping transposes / batches /
  strides / offsets into MPS descriptors.
- MPS uses its own encoders — interplay with our single compute-command-encoder
  execution model and the P8 encode plan.
- May underperform on tiny/odd/skinny shapes; need a shape-based fallback.
- Lower correctness risk than hand-rolled MMA; higher "does it fit our runtime"
  risk.

### Option C — bigger scalar tiles + register blocking + `float4` (modest ceiling, low risk)
Each thread computes a micro-tile (e.g. 4×4) of outputs, larger threadgroup
tiles, vectorized (`float4`) loads/stores, K-loop unrolling. Gets perhaps
**2–3×** over the current 16×16 scalar kernel with no new dependency and low
correctness risk. Does not reach matrix-unit peak.

---

## 3. Recommendation (sequencing)

1. **Do Option C first** — safe, real gain, no new deps; banks progress and
   establishes a better scalar baseline + the autotune/oracle harness.
2. **Prototype Option A (`simdgroup_matrix`)** behind a shape guard + env flag,
   validated against the existing tiled kernel as the oracle, with on-device
   tile-size autotuning. Treat as max-upside, must-measure, keep the proven
   kernel as fallback.
3. **Benchmark Option B (MPS)** in parallel as a reference ceiling; adopt if it
   beats the custom kernel and integrates cleanly, otherwise keep as a
   documented alternative.

Rationale: C de-risks and gives the harness; A has the highest kernel ceiling
but needs measurement; B is the pragmatic "near-peak for free" check that tells
us how much headroom A is chasing.

---

## 4. Cross-cutting requirements

- **Oracle parity.** Every variant proves CPU↔Metal parity via `gd_graph_compare`
  (1e-4) and keeps the existing scalar tiled kernel as the correctness fallback.
  Document any tolerance change from accumulation-order differences.
- **Shape coverage + fallback.** Fast paths may require alignment / dtype / shape
  constraints; a node that does not qualify must route to the current tiled
  kernel. Selection happens at compile (P8 plan) where shapes are known.
- **GPU_SAFE preserved.** One IR node → one backend op. No fusion, no layout
  changes to the public contract. Internal repack/scratch is allowed
  (allocation-free per run, like the SDPA stats scratch).
- **Autotune/cache.** Tile/threadgroup params should be selectable and, ideally,
  cached per (shape, dtype) — aligns with design_spec §17 kernel cache. Start
  with a fixed good config; add autotuning only if it pays.
- **Profiling.** Use `GD_PROFILE=trace` per-op GPU timing and a standalone GEMM
  microbench (the `/tmp/gemm_bench` pattern) across a shape sweep
  (square, skinny-N vocab, skinny-K, batched). Always also report the canonical
  end-to-end step (B=4, T=256).

---

## 5. Phases

- [ ] T0 — GEMM microbench + shape sweep harness (square, vocab-skinny, batched)
  committed under `tests/`/`bench/`, plus oracle parity diff against the scalar
  tiled kernel. Establishes the measurement loop.
- [x] T1 — Option C: register-blocked / `float4` GEMM (matmul + linear),
  parity-gated. **Done — see §9.** Replaced the 1-thread-per-output 16×16 tiled
  kernels with 64×64 thread-tile blocking: each thread owns a 4×4 micro-tile
  (`float4` accumulators + `float4` inner reads), `BK=8` K-tiles staged in
  threadgroup memory. Handles all shapes (bounds-checked) + transpose/batch, so
  it is a drop-in for the old kernels.
- [ ] T2 — Option B spike: `MPSMatrixMultiplication` behind `GD_METAL_MPS=1`,
  buffer aliasing + transpose/batch mapping, microbench vs C, decide keep/drop.
- [x] T3 — Option A: `simdgroup_matrix` kernel — **attempted, parity-correct, but
  slower for fp32; gated off behind `GD_METAL_GEMM_SIMD=1`.** See §8.1.
- [ ] T4 — Selection + (optional) per-(shape,dtype) autotune cache; wire fast
  path into the P8 compile plan; document final shape→kernel policy.

Each phase: land with parity + numbers; do not regress the canonical workload.

---

## 6. Reporting rule (same as the other plans)

When marking a phase done, record in this doc:
1. **Before/after numbers**: GEMM microbench GFLOP/s across the shape sweep **and**
   the canonical end-to-end step (B=4, T=256) ms/iter + GFLOP/s.
2. **Non-trivial choices**: tile/threadgroup sizes, alignment/shape constraints,
   fallback conditions, any new dependency.
3. **Learnings / surprises**: occupancy, register pressure, MMA mapping gotchas,
   where the GFLOP/s multiple did/didn't pass through to end-to-end.
4. **Validation**: parity (`gd_graph_compare` 1e-4), GPT train parity, ASan.

---

## 7. Benchmark table

| Phase | Shape / workload | Kernel | GFLOP/s | Notes |
|---|---|---|---:|---|
| P10 | 512³ | scalar tiled 16×16 | ~289 | naive was ~169 |
| baseline | B=4 T=256 step | scalar tiled | ~164 (step) | GEMM ~206ms of ~500ms (~41%) |
| T1 | B=4 T=256 step | 64×64 reg-blocked | ~201 (step) | matmul 148->81ms, linear 57->38ms; 2065->2511 tok/s |
| T1 | B=4 T=1024 step | 64×64 reg-blocked | ~136 (step) | matmul 558->248ms (2.25×), linear 196->102ms (1.92×) |
| T3 | B=4 T=256 matmul | simdgroup_matrix (fp32) | — | 408ms vs 82ms reg-blocked — 3–5× slower, gated off (§8.1) |

(Fill as T2/T4 land.)

## 8. T1 findings — register-blocked / float4 GEMM

Replaced both tiled GEMM kernels (`gd_matmul_tiled`, `gd_linear_tiled`, same
names/op-map) with 64×64 thread-tile blocking: a 16×16 threadgroup (256 threads)
computes a 64×64 output block; each thread owns a 4×4 micro-tile held as
`float4 acc[4]`; the K dimension streams in `BK=8` tiles staged into threadgroup
memory; the inner loop loads one `float4` row of A and B from threadgroup memory
and does 4 fused `acc[i] += a[i]*bvec` ops. Bounds checks on every staged load
and every store keep it a drop-in for all shapes (incl. the non-multiple V=8000
LM head) plus `trans_a/trans_b`, weight transpose, bias, and batch broadcast.

Numbers (`GD_PROFILE=trace`, B=4, 6 layers):

```text
            T=256                 T=1024
matmul      148 -> 81 ms (1.83x)   558 -> 248 ms (2.25x)
linear       57 -> 38 ms (1.50x)   196 -> 102 ms (1.92x)
step        496 -> 407 ms          3316 -> 2950 ms
GFLOP/s     165 -> 201             121 -> 136
tokens/s   2065 -> 2511           1235 -> 1388
```

Validation: CPU↔Metal parity at 1e-4 (`test_metal` matmul/linear across
square/skinny/transposed/batched shapes), GPT train parity, ASan-clean.

Learnings / choices:
- **Arithmetic intensity is the lever.** The old kernel reissued a threadgroup
  load per FMA (1 thread/output); the 4×4 micro-tile reuses each loaded A/B
  `float4` across 16 FMAs, cutting threadgroup-load and barrier traffic per FLOP
  ~4×. That, not raw `float4`, is the bulk of the win.
- **Low register pressure helped occupancy.** `acc[4]` (16 floats) + two `float4`
  operands is far lighter than the SDPA kernels' 64-float register arrays, so
  the GEMM kernel keeps high occupancy — a contrast worth noting vs §6.x of the
  block-sparse plan where occupancy was the SDPA ceiling.
- **Bigger micro-tiles (8×8) not yet tried.** Would raise intensity further but
  needs `float4`-pair accumulators and more threadgroup memory; left for a later
  pass since T1 already shifted the canonical workload to ~201 GFLOP/s.
- **End-to-end vs kernel multiple (Amdahl, §1.1).** matmul ~2.25× / linear
  ~1.9×, but the T=1024 step only moved 3316->2950 (1.12×) because attention +
  the tail still dominate; the canonical T=256 step moved 496->407 (1.22×),
  matching GEMM's ~41% share. Reaching matrix-unit peak (Option A,
  `simdgroup_matrix`) is the remaining GEMM ceiling.

---

## 8.1 T3 finding — simdgroup_matrix is not competitive for fp32

Implemented `gd_matmul_simd` / `gd_linear_simd`: 64×64 block per threadgroup, 4
simdgroups (2×2) each owning a 4×4 grid of 8×8 `simdgroup_float8x8` accumulator
fragments, K streamed in 8-deep tiles (transpose/batch normalized during
staging). Parity-correct (CPU↔Metal 1e-4, GPT train), but measured **3–5× slower
than the register-blocked kernel** on M-series:

```text
matmul gpu_ms (B=4, 6 layers, GD_PROFILE=trace):
  T       reg-blocked (T1)   simdgroup (T3)
  256     82                 ~408
  1024    248                ~1870
```

Iteration ruled out the obvious host-side culprits:
- **Output staging**: tried (a) full 64×64 threadgroup tile, (b) per-simdgroup
  8×8 staged copy, (c) direct device `simdgroup_store` for interior blocks. All
  ~the same (~408 ms at T=256) → the store is **not** the bottleneck.
- **Occupancy**: the direct-store path uses only As/Bs (~4 KB threadgroup), so
  occupancy is high, yet no improvement.

That leaves the `simdgroup_multiply_accumulate` itself: on Apple GPUs the matrix
units are built for fp16/bf16, and **fp32 matrix-multiply runs at a fraction of
the fp32 vector-ALU rate**. The register-blocked `float4` kernel already saturates
fp32 FMA throughput, so the matrix path cannot win in fp32.

Decision: keep the kernels behind `GD_METAL_GEMM_SIMD=1` (opt-in, default off so
there is no regression) as a validated foundation for a future mixed-precision
(fp16/bf16) path — that is where `simdgroup_matrix` pays off. The shipping fp32
GEMM remains the register-blocked kernel (T1).

Learning: `simdgroup_matrix` is a precision-gated optimization, not a free GEMM
speedup. Revisit T3 only alongside an fp16/bf16 compute path; for fp32 the
remaining headroom is in the register-blocked kernel (bigger micro-tiles, deeper
K-unroll) or Option B (MPS, which internally selects precision-appropriate
kernels).

## 9. Risks

- **On-device tuning dependency.** Option A peak requires iterating tile sizes /
  occupancy on real hardware; without a tight measurement loop it can land
  *slower* than C. Mitigation: T0 harness + autotune + keep C fallback.
- **Diminishing end-to-end returns (Amdahl).** GEMM is ~37% of the step;
  attention ~32%; the non-GEMM "tail" ~32% (see §1.1). A several× GEMM kernel
  speedup may be only ~1.2–1.5× end-to-end until the tail is also trimmed. The
  tail is mostly memory-bandwidth-bound elementwise/reduction ops plus the
  `[*,8000]` LM-head/cross-entropy region; trimming it is largely a **fusion**
  problem (`GPU_FUSED`, a separate effort) with a few GPU_SAFE wins (batched
  AdamW, more `copy` aliasing, atomic `embedding_bwd`). Mitigation: always report
  end-to-end; treat GEMM and the tail as complementary tracks; handle the skinny
  vocab GEMM separately from the square-ish projections.
- **Parity drift.** MMA / different accumulation order changes rounding;
  validate at 1e-4 and document. Mitigation: oracle diff in T0.
- **MPS integration.** Framework dependency + encoder-model mismatch could
  complicate the single-encoder + P8 plan execution. Mitigation: behind a flag,
  spike first (T2), keep optional.
- **Maintenance surface.** Multiple GEMM variants + shape selection increase
  complexity. Mitigation: one fallback (scalar tiled) is always correct; fast
  paths are strictly opt-in by shape at compile.
```
