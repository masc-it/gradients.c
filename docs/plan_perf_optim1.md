# Performance optimization pass 1

Status: profiling baseline captured.

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

## Notable anomaly

`GD_PROFILE=trace` measured about `2.68s/step` for `B=8 T=1024`, while normal
summary/wall mode measured about `3.33s/step`. Trace mode normally should be
slower because it serializes one command buffer per node. This suggests the
single large command buffer path, MPS interop, encoder switching, or scheduling
may add overhead/stalls not visible in trace attribution.

This deserves investigation, but it does not change bottleneck order: both trace
and normal runs are dominated by GPU execution, and trace points clearly to SDPA.

## Optimization order

1. Optimize `sdpa_bwd` Metal kernel.
2. Optimize `sdpa` Metal forward kernel.
3. Investigate normal-vs-trace scheduling gap:
   - huge single command buffer behavior
   - MPS encode boundaries / compute encoder close-resume cost
   - possible long dependency chains or command-buffer occupancy issue
4. Re-profile GEMM/MPS after attention improvements, especially for wider
   20M-ish configs or shorter sequence lengths.
5. Keep fused LMCE as-is for now. At vocab 8k it is not current bottleneck.

## Current conclusion

For planned GPT training workloads, efficient long-context attention is the gate.
At `T=1024`, attention is roughly three quarters of step time; at `T=2048`, it is
closer to seven eighths. Further training runs should wait on SDPA forward/backward
optimization or at least a conscious decision to accept current throughput.
