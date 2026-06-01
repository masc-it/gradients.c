# Performance optimization pass 1

Status: baseline captured; command-buffer chunking, SDPA kernels, and planned MPS GEMM dispatch landed.

## Goal

Before longer GPT training runs, identify current bottlenecks for the Metal
training path and choose optimization order. Target models are roughly 8–20M
parameters, sequence length 256–2048, batch size tuned to 16GB unified memory
(typically 4–8 for tests).

## Baseline command

```sh
GD_GPT_MLP_POWLU=1 \
GD_BENCH_FUSED_LMCE=1 \
GD_METAL_MPS=1 \
GD_DEVICE=metal \
GD_BENCH_ITERS=10 \
GD_BENCH_WARMUP=2 \
GD_BENCH_T=1024 \
GD_BENCH_B=8 \
make gpt-bench
```

Same command was also run with `GD_PROFILE=summary` and `GD_PROFILE=trace` plus
`GD_PROFILE_BACKEND=metal`.

## Baseline result: B=8, T=1024

Model/config:

- params: 8.34M
- `d_model=256`, `layers=6`, `heads=4`, `kv_heads=4`, `head_dim=64`
- `d_ff=1024`, `vocab=8000`
- MLP: PowLU, `m=3`
- training: forward + backward + AdamW + fused LM cross entropy + grad clipping

Release run with `GD_PROFILE=summary`:

- mean: `3325.8 ms/iter`
- best: `3275.1 ms/iter`
- best throughput: `2501 tok/s`
- mean compute: `169.7 GFLOP/s`
- peak live GPU allocation: `3.54 GB`
- max RSS: `~3.84 GB`
- host encode: `~14.8 ms / 12 runs`, about `1.2 ms/iter`

Host encode is not the bottleneck. Most wall time is in Metal command-buffer
execution/wait.

## Trace attribution: B=8, T=1024

Trace mode serializes one node per command buffer to attribute host-measured GPU
work by op. It changes scheduling, so use percentages/order more than absolute
wall time.

Per-step approximate attribution from `GD_PROFILE=trace` over 12 total runs
(2 warmup + 10 measured):

| op/group | ms/step | share |
|---|---:|---:|
| `sdpa_bwd` | `1480` | `55%` |
| `sdpa` | `556` | `21%` |
| `matmul` | `252` | `9%` |
| `linear` | `124` | `5%` |
| `lm_cross_entropy` + `lm_cross_entropy_bwd` | `89` | `3%` |
| other ops combined | `~179` | `7%` |

Detailed per-step trace:

| op | ms/step | share |
|---|---:|---:|
| `sdpa_bwd` | `1480.3` | `55.2%` |
| `sdpa` | `556.1` | `20.7%` |
| `matmul` | `252.3` | `9.4%` |
| `linear` | `124.1` | `4.6%` |
| `lm_cross_entropy` | `50.6` | `1.9%` |
| `lm_cross_entropy_bwd` | `38.2` | `1.4%` |
| `add` | `33.5` | `1.2%` |
| `powlu_bwd` | `27.8` | `1.0%` |
| `rope_bwd` | `20.5` | `0.8%` |
| `adamw_step` | `16.7` | `0.6%` |
| `copy` | `15.7` | `0.6%` |
| `reduce_to` | `14.9` | `0.6%` |
| `rms_norm` | `13.8` | `0.5%` |
| `rope` | `13.1` | `0.5%` |
| `powlu` | `9.1` | `0.3%` |
| `rms_norm_wbwd` | `8.5` | `0.3%` |
| other tiny ops | `<3 each` | `<0.1% each` |

Main bottleneck is attention. `sdpa + sdpa_bwd` accounts for about `76%` of
traced step at `T=1024`; `sdpa_bwd` alone is the largest item by far.

## Sequence-length sensitivity checks

Additional one-iteration trace checks:

| shape | traced step | attention share | GEMM share | LMCE share |
|---|---:|---:|---:|---:|
| `B=8 T=512` | `981 ms` | `61%` | `21%` | `5%` |
| `B=8 T=1024` | `2681 ms` | `76%` | `14%` | `3%` |
| `B=4 T=2048` | `4716 ms` | `86%` | `8%` | `2%` |

Attention dominance grows quickly with sequence length. For 1024–2048 context,
non-attention kernel work has limited upside unless attention improves first.

## Resolved anomaly: trace faster than normal

`GD_PROFILE=trace` measured about `2.68s/step` for `B=8 T=1024`, while the old
normal summary/wall path measured about `3.33s/step`. Trace mode should normally
be slower because it serializes one command buffer per node.

Root cause: the normal Metal executor submitted the whole graph as one giant
command buffer. Long GPT graphs, especially long-context SDPA with repeated large
scratch-buffer read/write hazards, hit a Metal scheduler/hazard-tracking cliff.
Trace mode accidentally avoided this by using many small command buffers.

Evidence:

| experiment | result | conclusion |
|---|---:|---|
| old monolithic path, `B=8 T=1024` | `~3.35s` | reproduces slow path |
| trace path, `B=8 T=1024` | `~2.68s` | faster because it chunks by node |
| command-buffer chunk every 8 graph nodes | `~2.43s mean`, `2.41s best` | fixes cliff and beats trace |
| compute-encoder split every 8, same command buffer | `~3.24s` | encoder boundaries are not enough |
| no MPS / no fused LMCE, monolithic | `~3.41s` | not caused by MPS or LMCE |
| no MPS / no fused LMCE, trace | `~2.70s` | reproduces without MPS |
| no MPS / no fused LMCE, command chunk 8 | `~2.48s` | chunking still fixes it |
| `B=4 T=2048`, monolithic | `~6.02s` | larger context worsens cliff |
| `B=4 T=2048`, command chunk 8 | `~4.52s` | chunking fixes long-context case too |

Implemented fix: normal Metal execution now submits graphs as a stream of small
command buffers. Initial default was `8` encoded graph nodes; the SDPA causal
specialization pass later lowered the default to `4` for better sustained
long-context training. `GD_METAL_CMD_CHUNK=0` forces the old
single-command-buffer path for A/B testing.

Post-fix baseline for original command (`B=8 T=1024`, 2 warmup + 10 measured):

- mean: `2460.2 ms/iter`
- best: `2403.5 ms/iter`
- best throughput: `3408 tok/s`
- best compute: `234.9 GFLOP/s`
- profiler wait event reports `items=56` command buffers per step

This resolves the trace-vs-normal contradiction: normal chunked execution is now
faster than trace, as expected.

## Optimization order

1. Keep command-buffer chunking enabled by default; use `GD_METAL_CMD_CHUNK=0`
   only for regression/debug A/B tests.
2. Optimize `sdpa_bwd` Metal kernel (see `plan_sdpa_perf_optim.md`).
3. Optimize `sdpa` Metal forward kernel (see `plan_sdpa_perf_optim.md`).
4. Re-profile GEMM/MPS after attention improvements, especially for wider
   20M-ish configs or shorter sequence lengths.
5. Keep fused LMCE as-is for now. At vocab 8k it is not current bottleneck.

## Follow-up: planned MPS GEMM dispatch

After SDPA lane-kernel work, GEMM became visible enough to re-profile. The
compiler already built `GDMPSGemmPlan` objects for contiguous no-bias `linear`
and supported `matmul` nodes when `GD_METAL_MPS=1`, but the encode path still
always dispatched the portable Metal GEMM kernel. Wired `linear` and `matmul`
encode to use the preplanned MPS matrix multiplication path when present, while
keeping the existing Metal kernel fallback for bias, broadcasted, offset, or
non-contiguous cases.

Trace deltas (`GD_PROFILE=trace`, one measured iteration):

| shape | step before | step after | `linear` before -> after | `matmul` before -> after |
|---|---:|---:|---:|---:|
| `d=256 B=8 T=512` | `614.0 ms` | `519.5 ms` | `70.9 -> 44.6 ms` | `126.3 -> 64.4 ms` |
| `d=256 B=8 T=1024` | `1559.8 ms` | `1347.0 ms` | `125.4 -> 68.4 ms` | `227.6 -> 102.7 ms` |
| `d=384 B=4 T=1024` | `1297.2 ms` | `1083.7 ms` | `135.3 -> 68.2 ms` | `247.9 -> 105.9 ms` |

Requested release run (`d=256 B=8 T=1024`, 2 warmup + 10 measured):

| metric | before MPS dispatch | after MPS dispatch |
|---|---:|---:|
| mean ms/iter | `1323.6` | `1133.0` |
| best ms/iter | `1320.0` | `1131.3` |
| best tokens/s | `6206` | `7241` |
| best GFLOP/s | `427.7` | `499.0` |

Long-context release check (`d=256 B=4 T=2048`, 1 warmup + 4 measured):

- mean: `2000.7 ms/iter`
- best: `1995.0 ms/iter`
- best tokens/s: `4106`
- best GFLOP/s: `360.5`

## Current conclusion

For planned GPT training workloads, efficient long-context attention remains the
largest gate, but MPS GEMM dispatch removed the next major non-attention gap.
At `T=1024`, attention is still the majority of traced time; at `T=512` and
wider `T=1024`, GEMM is no longer the obvious next target. Next high-upside work
should return to attention or improve trainer production features.
