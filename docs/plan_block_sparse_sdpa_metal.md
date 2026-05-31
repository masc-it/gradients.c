# gradients.c — Block-Sparse / Causal-Skip SDPA (Metal)

Status: draft v0.1

Goal: cut the O(T²) attention work for masked attention (causal, sliding-window,
and general block-sparse) by **skipping fully-masked key blocks** in the tiled
Metal SDPA kernels, instead of iterating every key block and masking
per-element. Start with the cheap, high-value **causal block-skip** (~2× on
causal attention), then generalize to sliding-window and an explicit
block-sparse schedule.

This is the Metal-kernel realization of [`docs/plan_gpt.md`](plan_gpt.md) §5
(block-sparse mask model) and the G4 phase, building directly on the G3
FlashAttention tiling that already landed
(`gd_sdpa_tiled` / `gd_sdpa_bwd_{stats,dq,dkv}`).

References:
- [`docs/plan_gpt.md`](plan_gpt.md) §5 (block mask model), §6 (SDPA kernels),
  G2 (dense+causal reference), G3 (tiled), G4 (block-sparse).
- [`docs/plan_metal_improve_foundations.md`](plan_metal_improve_foundations.md)
  (profiling harness, `GD_PROFILE=trace`, canonical workload).
- [`docs/plan_gemm_perf_tuning_metal.md`](plan_gemm_perf_tuning_metal.md) §1.1
  (Amdahl split; which workloads are attention- vs GEMM-bound).

---

## 1. Motivation (measured)

Per-op trace of the canonical training step (12.39M GPT, 6 layers,
`GD_PROFILE=trace`), at two sequence lengths:

```text
                 T=256            T=1024
sdpa  (fwd)      48 ms            476 ms
sdpa_bwd         128 ms           1324 ms
attention total  181 ms (32%)     1801 ms (60%)
GEMM             210 ms (37%)     717 ms (24%)
tail             182 ms (32%)     486 ms (16%)
step (real)      500 ms           3734 ms
throughput       164 GFLOP/s      107 GFLOP/s
tokens/s         2055             1104
```

Attention is O(T²); GEMM and the tail are O(T). So at long context attention
dominates (60% at T=1024) and tokens/s falls. The G3 tiled kernels fixed the
*constant factor* (memory traffic) but not the asymptotics.

**The waste:** the current tiled kernels (`gd_sdpa_tiled`,
`gd_sdpa_bwd_{stats,dq,dkv}`) loop over **all** key blocks `kb in [0, Tk)` and
apply `gd_sdpa_allowed(i, j, ...)` per element. For **causal** attention, a query
block near position `q0` can never attend to key blocks that start beyond its
last query's causal horizon — roughly **half** the (query-block, key-block)
pairs are entirely masked yet still loaded and processed. Skipping those blocks
is a ~2× win on causal attention with no change to results.

---

## 2. Key idea: block-level skip + per-element predicate

The tiled kernels already iterate key blocks; add a cheap **block-range test**
that skips a key block when *no* (query in this query-block, key in this
key-block) pair is allowed, while keeping the existing fine-grained
`gd_sdpa_allowed` per-element test inside any block that is *partially* allowed
(the diagonal / window-edge blocks). This is exactly the safety property from
`plan_gpt.md` §5: the coarse block list may be conservative, but the per-element
predicate guarantees no future/out-of-window key ever contributes.

Definitions (query block `qb` covers queries `[q0, q0+BQ)`, key block `kb`
covers keys `[k0, k0+BK)`; causal offset `off = Tk - Tq`, so query `i` may attend
to key `j` iff `j <= i + off` (and, for windows, `i + off - j < w`)):

- **Causal skip**: skip key block if `k0 > (q0 + BQ - 1) + off` (its first key is
  beyond the last query's horizon → fully masked). Equivalently iterate
  `kb` only up to the block containing `q0 + BQ - 1 + off`.
- **Sliding-window skip**: additionally skip key blocks entirely *below* the
  window: skip if `(q0 + off) - (k0 + BK - 1) >= w` (the whole block is older
  than the window for every query in the block).
- **Block-sparse (general)**: a per-query-block list of allowed key-block ids
  (the `block_mask` graph value from `plan_gpt.md` §5); the kernel visits only
  listed blocks. Causal/window are the degenerate built-in cases.

Partial blocks (those that straddle the diagonal/window edge) are still visited
and rely on the per-element predicate — correctness is unchanged; only fully
dead blocks are dropped.

---

## 3. Where it applies

All four tiled kernels iterate key (or query) blocks and benefit:

- `gd_sdpa_tiled` (forward): per query-block, skip dead key blocks.
- `gd_sdpa_bwd_stats` (per query-block): same key-block skip.
- `gd_sdpa_bwd_dq` (per query-block): same key-block skip.
- `gd_sdpa_bwd_dkv` (per **key**-block): symmetric — skip dead **query** blocks
  for this key block (causal: queries with `i + off < k0` cannot attend → skip
  query blocks entirely below the key block).

The dkv kernel is the largest single attention cost (≈1324 ms at T=1024), so its
query-block skip matters most.

---

## 4. GPU_SAFE / contract notes

- Stays GPU_SAFE for causal/window: one IR node → one backend op, same math; the
  kernel just visits fewer blocks. No new op, no fusion.
- General block-sparse adds a `block_mask` **graph value** (int32
  `[n_q_blocks, max_kb]`, `-1` padded), per `plan_gpt.md` §5 — an extra input to
  the `sdpa` node, host-built. That is the G4 surface; it is still one node.
- CPU_REF stays the oracle (dense masked reference). Block-skip must produce
  bit-for-block-identical results to the dense-masked path within tolerance
  (it computes the exact same allowed set).
- Block sizes reuse the G3 tile constants (`GD_METAL_SDPA_BQ`, `…_BK`) so the
  skip math lines up with the staging tiles. Expose `attn_block_q/attn_block_k`
  attrs later if independent block sizes are wanted (`plan_gpt.md` §3.2).

---

## 5. Phases

- [x] B1 — **Causal block-skip** in all four tiled kernels (forward + stats + dq
  + dkv). Bound the block loop / `continue` on dead blocks; keep the per-element
  predicate for partial blocks. Parity vs dense-causal reference; measure at
  T=256 / 1024. **Implemented and correct, but wall-time-neutral for causal
  self-attention — see §6.1 for the finding.** The infrastructure (uniform block
  bounds `gd_sdpa_kb_end` / `gd_sdpa_qb_start`) is the foundation B2/B1.5 build on.
- [ ] **B1.5 — Split-K / flash-decoding (the actual causal wall-time lever).**
  See §6.1: block-skip removes only the *light* groups; the critical path (heavy
  blocks scanning the full key range) is unchanged and saturates the GPU. Split
  each query block's key range across several threadgroups, each computing a
  partial online-softmax `(m, l, acc)` over a key slice, then a combine pass
  merges them. Shortens the per-group critical path from `Tk` to `Tk/nsplit`.
  Higher complexity (needs a merge step + scratch); closer to GPU_FUSED. This is
  the prerequisite for any causal forward wall-time win on this hardware.
- [ ] B2 — **Sliding-window block-skip** (skip blocks above the diagonal *and*
  below the window). **Expected to win where B1 does not**: a window caps *every*
  query block's key scan to `w`, so it shortens the critical path itself (unlike
  causal, which leaves the last block scanning all `Tk`). Parity vs windowed
  reference.
- [ ] B3 — **General block-sparse**: `block_mask` graph value + host pattern
  builders in `nn.h` (`causal`, `sliding_window(w)`, `block_sparse(custom)` e.g.
  BigBird window+global+random); forward + the per-query-block kernels visit the
  listed key blocks. (This is `plan_gpt.md` G4, forward-first.)
- [ ] B4 — **Sparse backward** (dq/dkv over the block list). Deferred in
  `plan_gpt.md` (sparse is forward/inference-first); land after B3 if training
  with sparsity is needed.

Each phase: land with CPU↔Metal parity (`gd_graph_compare` 1e-4) + GPT train
parity + ASan, and report numbers at multiple T.

---

## 6. Expected payoff

- **Causal (B1)**: reduces *total* attention compute ~2× (the strictly-lower
  triangle is skipped) but — measured — **does not reduce wall-time** on this
  GPU/workload (see §6.1). It is energy/throughput-neutral-to-slightly-positive
  and a correctness-preserving foundation for B2/B1.5.
- **Sliding-window (B2)**: from O(T²) to O(T·w). Unlike causal, this *does* cut
  wall-time because it shortens the critical path (every block scans ≤ `w` keys).
- **Split-K (B1.5)**: shortens the causal critical path `Tk → Tk/nsplit`; the
  first lever that actually speeds up causal forward here.
- **Block-sparse (B3/B4)**: scales with the chosen sparsity pattern.

Note the Amdahl framing from the GEMM plan §1.1: the end-to-end multiple is
always smaller than the attention-kernel multiple; report both.

### 6.1 Finding — block-skip is critical-path-bound, not throughput-bound

B1 was implemented in all four tiled kernels and validated (multi-block causal
and causal+window parity at 1e-4, `test_sdpa_multiblock`, T=130 spanning 3 query
/ 9 key blocks). But the canonical bench showed **no wall-time change**:

```text
T=1024, B=4, 6 layers (GD_PROFILE=trace), forward sdpa gpu_ms:
  dense (no skip)          ~476 ms   (256 block-units of work)
  causal block-skip (B1)   ~477 ms   (136 block-units, ~0.53x)
  uniform key-cap to 512   ~305 ms   (perf probe, shorter MAX scan)
  uniform key-cap to 16    ~16 ms    (perf probe)
T=1024, B=16, 1 layer:
  causal block-skip        226 ms
  dense                    235 ms    (~4% apart despite 0.53x work)
```

Reducing *total* work 256 → 136 units gave **zero** speedup; only reducing the
*maximum* per-group key scan (uniform cap) helped. Conclusion: the makespan is
bound by the **critical path**, not throughput. For causal self-attention the
last query block (and, in `dkv`, the first key block) must scan the full key/
query range; those ~`B·Hq` heavy threadgroups alone saturate the GPU for the
whole kernel duration, and the light groups the skip removes were overlapping
them for free. Raising parallelism (B=16, 1280 groups) did not change this — the
heavy groups still dominate.

Implications:
1. **Causal forward/backward wall-time needs key-dimension splitting** (B1.5
   split-K / flash-decoding), not block-skip.
2. **Block-skip still pays off** for (a) sliding-window / block-sparse, where it
   shortens the critical path because *every* block's scan is bounded, and (b)
   throughput-bound regimes (huge grids, energy-limited) where total work
   dominates. Hence B1 is kept as the foundation, not reverted.
3. The CPU_REF reference still loops the dense triangle; CPU is unaffected.

Learning to carry forward: on Apple GPUs, an attention kernel with low
per-threadgroup occupancy (64 threads, large register arrays) is critical-path
bound, so the optimization that matters is *shortening the longest single
threadgroup's scan*, not *reducing aggregate work*.

---

## 7. Reporting rule

When marking a phase done, record:
1. **Before/after numbers**: `sdpa` and `sdpa_bwd` `gpu_ms` (`GD_PROFILE=trace`)
   at T=256 / 1024 / 2048, plus the end-to-end step (B=4) ms/iter, GFLOP/s,
   tokens/s.
2. **Non-trivial choices**: block-bound math, partial-block handling, block sizes,
   `block_mask` representation (CSR-of-blocks vs bitmap — `plan_gpt.md` §13.3).
3. **Learnings / surprises**: occupancy at long T, divergence from skipped
   blocks, any tolerance change.
4. **Validation**: dense-causal/windowed parity, block-mask-equals-causal
   identity test (`plan_gpt.md` §10.5), GPT train parity, ASan.

---

## 8. Benchmark table

| Phase | T | sdpa fwd | sdpa_bwd | step (real) | tokens/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| G3 (dense tiled) | 256 | 48 ms | 128 ms | 500 ms | 2055 | attention 32% |
| G3 (dense tiled) | 1024 | 476 ms | 1324 ms | 3734 ms | 1104 | attention 60% |
| B1 (causal skip) | 256 | 50 ms | 128 ms | 496 ms | 2068 | no change — critical-path bound (§6.1) |
| B1 (causal skip) | 1024 | 477 ms | 1306 ms | 3708 ms | 1110 | no change — critical-path bound (§6.1) |

B1 is correct (parity green) and neutral on wall-time for causal; the wall-time
levers are B2 (window) and B1.5 (split-K). (Fill as B1.5–B4 land.)

---

## 9. Risks

- **Correctness leak (future tokens visible).** A too-aggressive block skip could
  drop a partially-allowed block. Mitigation: only skip blocks proven *fully*
  dead by the block-range test; always keep the per-element `gd_sdpa_allowed`
  predicate inside visited blocks (the `plan_gpt.md` §5 safety property). Gate
  with a "block_mask == causal must equal dense-causal exactly" test.
- **Thread divergence / occupancy.** Skipping blocks changes per-threadgroup work
  imbalance (query blocks late in the sequence visit more key blocks under
  causal). Mitigation: measure; the imbalance is inherent to causal and still a
  net win; consider load-balancing schedules only if profiling demands it.
- **dkv symmetry bug.** The dkv kernel skips *query* blocks (not key blocks);
  off-by-one in the transposed bound is easy. Mitigation: derive both bounds from
  the single `gd_sdpa_allowed` predicate and unit-test against the dense path.
- **Block size coupling.** Reusing the G3 staging tile size as the skip
  granularity ties two concerns; fine for now, revisit if independent
  `attn_block_q/k` are exposed.
- **Backward sparsity scope.** Sparse backward (B4) is heavier and deferred in
  `plan_gpt.md`; do not block B1–B3 on it.
```
