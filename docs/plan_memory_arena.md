# gradients.c — Metal Scratch Arena Plan

Status: implemented v1 (2026-05-31)

Goal: replace per-node Metal scratch buffers with one executable-level shared
scratch arena sized to the maximum scratch need of any node. Nodes execute
sequentially in the current Metal backend, so internal scratch lifetimes do not
overlap. This should cut peak memory substantially, especially for long-context
SDPA split-K training, without changing graph semantics or public APIs.

References:
- [`docs/plan_metal_improve_foundations.md`](plan_metal_improve_foundations.md)
- [`docs/plan_block_sparse_sdpa_metal.md`](plan_block_sparse_sdpa_metal.md)
- `src/backends/metal/metal_backend.m`
- `src/backends/metal/metal_kernel_types.h`

---

## 1. Motivation

Today `_gd_executable` stores `node_scratch[j]` per node. Scratch is allocated at
compile time for ops that need temporary device storage:

- `sdpa` split-K forward partials
- `sdpa_bwd` split-K stats / dq / dkv partials
- `cross_entropy` per-position loss scratch
- `rms_norm_wbwd` row-block partials

For GPT, every transformer layer has its own SDPA and SDPA_BWD nodes, so each
node gets its own large scratch buffer even though only one node runs at a time.
This duplicates hundreds of MB at T=512 and multiple GB at T=1024+.

Current backend execution model:

```text
one executable run
  one command buffer
    encode node 0 dispatches
    encode node 1 dispatches
    ...
    encode node N dispatches
```

Dispatches in one command buffer/encoder are ordered. MPS nodes may close and
reopen the compute encoder, but command-buffer order is still preserved. Thus
scratch used by node `j` is dead before node `j+1` starts.

---

## 2. Estimated memory win

For the current 6-layer GPT workload (`H=5`, `Dh=64`, split policy
`S=ceil(T/128)`, capped), SDPA scratch dominates.

### B=8, T=512, S=4

```text
sdpa fwd scratch per node:
  B*H*T*S*(Dh+2)*4 ~= 21.6 MB
  6 nodes ~= 129.8 MB

sdpa_bwd scratch per node:
  stats + stats_part + dq_part + dkv_part
  ~= 85.1 MB
  6 nodes ~= 510.7 MB

total per-node SDPA scratch ~= 640 MB
shared arena max             ~= 85 MB
estimated save               ~= 555 MB
```

### B=8, T=1024, S=8

```text
sdpa fwd scratch per node ~= 86.5 MB
6 nodes                  ~= 519 MB

sdpa_bwd scratch per node ~= 340 MB
6 nodes                   ~= 2.04 GB

total per-node SDPA scratch ~= 2.56 GB
shared arena max             ~= 340 MB
estimated save               ~= 2.2 GB
```

CE/RMS scratch is small compared with SDPA, but should use the same arena for
simplicity and to remove all per-node scratch storage.

---

## 3. Design

Replace per-node scratch storage with per-node scratch size metadata plus one
arena:

```c
struct _gd_executable {
    ...
    size_t *node_scratch_bytes;
    gd_storage *scratch_arena;
    size_t scratch_arena_bytes;
};
```

Compile phase:

1. For each node, compute scratch byte need exactly as today.
2. Store it in `node_scratch_bytes[j]`.
3. Track `max_scratch_bytes`.
4. After node planning, allocate one Metal `gd_storage` of `max_scratch_bytes` if
   nonzero.
5. No per-node scratch buffer allocation.

Encode phase:

1. If `node_scratch_bytes[i] > 0`, bind `scratch_arena` as that node's scratch
   buffer.
2. Existing per-op internal offsets stay unchanged because every op sees scratch
   starting at byte offset 0.
3. If a node needs no scratch, pass `nil` as today.

Important: the arena is **not** graph-visible storage. It never backs tensor
values and is never downloaded. It is private executable scratch.

---

## 4. Correctness / safety

Safe under current backend assumptions:

- Nodes execute sequentially in graph order.
- Each scratch-using op writes its scratch before reading it within the same node:
  - CE writes per-position loss, then reduce reads it.
  - RMS wbwd writes row-block partials, then reduce reads them.
  - SDPA split-K writes partials, then combine/reduce reads them.
- No scratch contents are needed after the node finishes.
- CPU_REF remains oracle; graph values unchanged.

Trace mode remains safe: it runs one node per command buffer and waits after each
node, so reuse is even stricter.

Caveat for future work: if backend later overlaps nodes across multiple command
buffers/queues, a single arena is no longer sufficient. Then scratch must become
a lifetime-aware allocator/ring with command-buffer fences. Not needed now.

---

## 5. Implementation phases

### A1 — Arena metadata + allocation

- Add `node_scratch_bytes`, `scratch_arena`, `scratch_arena_bytes` to
  `_gd_executable`.
- Free arena in executable teardown.
- Replace `node_scratch[j]` allocation sites with byte-size computation:
  - `_GD_OP_RMS_NORM_WBWD`
  - `_GD_OP_CROSS_ENTROPY`
  - `_GD_OP_SDPA`
  - `_GD_OP_SDPA_BWD`
- Allocate arena once after planning.

### A2 — Encode path switch

- Replace:

```objc
id<MTLBuffer> scratch = exe->node_scratch[i] ? ... : nil;
```

with:

```objc
id<MTLBuffer> scratch = exe->node_scratch_bytes[i] > 0 ? arena : nil;
```

- Keep all op encoders unchanged.
- Keep SDPA region offsets unchanged (`stats_off`, `dq_part_off`, etc. remain
  offsets inside the arena for that node invocation).

### A3 — Profiling / diagnostics

Add optional profile event or debug log:

```text
scratch_arena_bytes
scratch_per_node_total_bytes
scratch_saved_bytes = total - arena
```

This makes memory savings visible without external tools.

### A4 — Validation

Run:

```sh
make build
make check
GD_METAL_MPS=1 make check
make docs-check
GD_METAL_MPS=1 ASan test_metal_gpt
```

Bench memory and speed:

```sh
GD_METAL_MPS=1 GD_DEVICE=metal GD_BENCH_T=512 GD_BENCH_B=8 make gpt-bench
GD_METAL_MPS=1 GD_DEVICE=metal GD_BENCH_T=1024 GD_BENCH_B=8 make gpt-bench
```

Expected speed: neutral to slight positive if avoiding memory pressure. Primary
win is peak memory/headroom.

---

## 6. Risks

- **Hidden scratch lifetime bug.** If any op reads scratch written by a prior
  node, arena reuse would break it. Current code has no such dependency; scratch
  is passed only to current node encoders.
- **Wrong arena size.** Underestimating size causes out-of-bounds GPU writes.
  Mitigation: preserve exact existing byte formulas and assert requested bytes <=
  arena bytes before encode.
- **Trace/profile accounting changes.** Trace mode remains correct, but memory
  allocation count changes. Document it.
- **Future parallel execution.** Single arena assumes sequential node execution.
  If parallel scheduling lands, replace with interval allocator/ring.

---

## 7. Implementation result (2026-05-31)

Implemented A1/A2:

- `_gd_executable` now stores `node_scratch_bytes[j]` plus one
  `scratch_arena`/`scratch_arena_bytes`.
- Compile computes each scratch byte need, tracks the maximum, and allocates one
  Metal arena.
- Encode binds the arena for any scratch-using node and checks
  `node_scratch_bytes[j] <= scratch_arena_bytes`.
- Removed per-node scratch `gd_storage` allocations.
- Op encoders and scratch internal offsets are unchanged.

A3 profiling counters were skipped for v1; external memory tools can confirm peak
allocation drop, and the compile formulas are unchanged from the old per-node
buffers.

Validation:

```text
make check                         green
GD_METAL_MPS=1 make check          green
make docs-check                    green
GD_METAL_MPS=1 ASan test_metal_gpt green
```

Release timing is neutral, as expected for a memory-capacity optimization:

```text
GD_METAL_MPS=1 B=4,T=256: 191.97 ms best, 5334 tok/s
GD_METAL_MPS=1 B=8,T=512: 1108.55 ms best, 3695 tok/s
Default Metal B=8,T=512:   1309.97 ms best, 3127 tok/s
GD_METAL_MPS=1 B=4,T=1024 smoke: 1965.41 ms, 2084 tok/s
```

## 8. Decision

Ship. This is low-risk production cleanup with large memory savings:

- ~555 MB saved for B=8,T=512,6L
- ~2.2 GB saved for B=8,T=1024,6L

Not a primary throughput optimization, but it unlocks larger contexts/batches and
reduces memory pressure. It stays GPU_SAFE: no graph changes, no math changes,
no public API changes.
