# SDPA Metal performance optimization

Status: causal SDPA specialization plus SIMD dK/dV lane reduction complete.

## Goal

Reduce GPT training step time before longer runs. Current target is Metal GPT
training at 8--20M params, sequence length 256--2048, batch size 4--8, with
CPU reference correctness preserved.

Priority order:

1. `sdpa_bwd` split path.
2. `sdpa` forward split path.
3. Re-profile GEMM/MPS only after attention is less dominant.

## Baseline trace matrix

Command shape:

```sh
GD_PROFILE=trace \
GD_GPT_MLP_POWLU=1 \
GD_BENCH_FUSED_LMCE=1 \
GD_METAL_MPS=1 \
GD_DEVICE=metal \
GD_BENCH_ITERS=1 \
GD_BENCH_WARMUP=0 \
GD_BENCH_B=<B> \
GD_BENCH_T=<T> \
make gpt-bench
```

Trace serializes one node per command buffer, so use op shares/order more than
absolute wall time.

| shape | traced step | `sdpa` | `sdpa_bwd` | attention share | peak live GPU alloc |
|---|---:|---:|---:|---:|---:|
| `B=8 T=256` | `441.8 ms` | `62.8 ms` | `118.7 ms` | `41%` | `1.17 GB` |
| `B=8 T=512` | `926.7 ms` | `174.9 ms` | `390.4 ms` | `61%` | `1.96 GB` |
| `B=8 T=1024` | `2754.4 ms` | `589.7 ms` | `1502.1 ms` | `76%` | `3.54 GB` |
| `B=4 T=2048` | `4700.2 ms` | `1071.1 ms` | `2885.0 ms` | `84%` | `3.44 GB` |

## Experiments

### Split-count tuning

Added runtime knobs for profiling only:

- `GD_METAL_SDPA_SPLIT_MIN`
- `GD_METAL_SDPA_SPLIT_MAX`

Tuning around the current default was noisy. Smaller/larger split counts shifted
single-run results but did not show a reliable broad win across `T=512/1024/2048`.
Keep defaults for now; knobs stay useful for future A/B runs.

### Rejected scratch reduction

Tested replacing split `dK/dV` partial scratch + reduce with zero + float atomic
accumulation. It reduced total allocated bytes for `B=8 T=1024` by about `134 MB`
(`3810 MB -> 3677 MB`) but slowed traced `sdpa_bwd` (`~1475 ms -> ~1583 ms`) and
did not reduce peak live allocation. Rejected: speed matters more and scratch peak
was not the limiting allocation.

### Implemented: causal/no-bias specialization

GPT training uses causal attention with no additive bias, no sliding window, and
no prefix region. Generic kernels still branch through mask and bias helpers for
every score. Added specialized Metal kernels for this hot path:

- `gd_sdpa_tiled_causal`
- `gd_sdpa_splitk_causal`
- `gd_sdpa_bwd_stats_dq_split_causal`
- `gd_sdpa_bwd_dkv_split_causal`

Host dispatch selects them only when:

```c
causal && window == 0 && prefix_len == 0 && !has_bias
```

All other SDPA modes keep the generic kernels. `GD_METAL_SDPA_CAUSAL_FAST=0`
disables the specialization for A/B testing. Backward causal kernels live in
`src/ops/sdpa/sdpa_bwd_causal.metal` to keep shader files under the size limit.

## Post-change trace matrix

| shape | traced step | `sdpa` | `sdpa_bwd` | attention share | note |
|---|---:|---:|---:|---:|---|
| `B=8 T=256` | `438.1 ms` | `63.5 ms` | `115.4 ms` | `41%` | small bwd win |
| `B=8 T=512` | `915.8 ms` | `167.1 ms` | `378.3 ms` | `60%` | fwd+bwd win |
| `B=8 T=1024` | `2606.7 ms` | `579.4 ms` | `1406.3 ms` | `76%` | bwd main win |
| `B=4 T=2048` | `4544.4 ms` | `1089.1 ms` | `2789.5 ms` | `85%` | bwd win; fwd noise/regression small |

Causal kernels exposed that the old command-buffer chunk size (`8`) can fall
back onto the slower sustained path after several long-context training steps.
Lowered default `GD_METAL_CMD_CHUNK` to `4`, which keeps the new kernels stable.
`GD_METAL_CMD_CHUNK=0` still forces the old monolithic path for A/B tests.

Original requested release run (`B=8 T=1024`, 2 warmup + 10 measured) improved
from previous post-chunk baseline:

| metric | before | after |
|---|---:|---:|
| mean ms/iter | `2460.2` | `2391.7` |
| best ms/iter | `2403.5` | `2380.0` |
| best tokens/s | `3408` | `3442` |
| best GFLOP/s | `234.9` | `237.2` |

## Follow-up: dK/dV lane reduction

Temporary subpass cuts showed `sdpa_bwd` at `B=8 T=1024` was approximately:

| cumulative cut | traced `sdpa_bwd` |
|---|---:|
| stats+dq split only | `614 ms` |
| stats+dq split+combine | `637 ms` |
| + dK/dV split | `1422 ms` |
| full op | `1426 ms` |

So remaining backward hot spot was `gd_sdpa_bwd_dkv_split_causal`; final reduce
was negligible. The old causal dK/dV kernel reduced each 8-lane dot product via
threadgroup scratch (`ss_part`, `dp_part`, `pjsh`, `dssh`) and two extra barriers
per query tile. Replaced that with 8-lane SIMD shuffle-xor reductions:

```text
ss = sum_8_lanes(q * k)
dp = sum_8_lanes(dO * v)
```

All lanes now compute the same `pj`/`ds` scalars directly and update their owned
channel slice, eliminating the dot partial scratch and two barriers. Compile-time
`#error` guards the assumption that `GD_SDPA_DKV_LANES == 8`.

Trace matrix after SIMD dK/dV reduction:

| shape | traced step | `sdpa` | `sdpa_bwd` | attention share |
|---|---:|---:|---:|---:|
| `B=8 T=256` | `419.4 ms` | `63.9 ms` | `93.8 ms` | `38%` |
| `B=8 T=512` | `840.2 ms` | `172.9 ms` | `298.3 ms` | `56%` |
| `B=8 T=1024` | `2277.7 ms` | `574.1 ms` | `1081.0 ms` | `73%` |
| `B=4 T=2048` | `3871.9 ms` | `1105.1 ms` | `2095.1 ms` | `83%` |

Requested release run (`B=8 T=1024`, 2 warmup + 10 measured) now:

| metric | after causal specialization | after SIMD dK/dV |
|---|---:|---:|
| mean ms/iter | `2391.7` | `2059.9` |
| best ms/iter | `2380.0` | `2051.4` |
| best tokens/s | `3442` | `3993` |
| best GFLOP/s | `237.2` | `275.2` |

## Next work

1. Forward `sdpa` is now the largest attention subpass after dK/dV; optimize
   causal split-K forward or reassess fwd specialization at `T=2048`.
2. Re-profile wider 12--20M configs; GEMM share may rise after the backward win.
3. Revisit scratch arena / memory reuse separately; atomic dK/dV is not the path.
4. Only after attention drops below ~50% of step, retune residual elementwise
   tail.
