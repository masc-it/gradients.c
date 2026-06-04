# v2 QA stress and performance report

Date: 2026-06-04
Scope: `docs/design_spec.md` foundation/runtime contract plus public F16 GEMM forward/backward through `gd_matmul` and `gd_matmul_backward`.
Device used for sample performance run: Apple M1 Pro, unified memory, Metal metallib backend.

## Probe coverage added / updated

- `src/ops/matmul/fwd.py` builds/runs a tiny C harness linked against `libgradients.a`, executes actual `gd_matmul`, and compares returned F16 outputs against PyTorch for classic transformer GEMM shapes.
- `src/ops/matmul/bwd.py` builds/runs a tiny C harness linked against `libgradients.a`, executes actual `gd_matmul_backward`, and compares `grad_x = grad_out @ W^T` / `grad_w = X^T @ grad_out` against PyTorch for the same shapes.
- `probes/v2_matmul_training_perf_probe.c` measures public API F16 `gd_matmul` plus optional `gd_matmul_backward` on transformer-like workloads with hidden sizes 128/256/512, MLP ratio 4, and 4 attention heads.
- `make gemm-perf-probe` builds a separate `-O3 -DNDEBUG` library/metallib under `build-perf`, runs the public GEMM performance probe with `GRADIENTS_METALLIB` pointed at that build.
- `probes/run_v2_qa_stress.sh` builds a separate optimized performance library/metallib in `build-qa-probes-perf`, runs the public GEMM performance probe, and can still run the standalone raw MPS baseline (`v2_metal_arena_probe`) for comparison.

## Commands run

```sh
make build
GD_MATMUL_FWD_PROFILE=classic uv run src/ops/matmul/fwd.py
GD_MATMUL_BWD_PROFILE=classic uv run src/ops/matmul/bwd.py
GD_QA_PERF_BACKWARD=1 GD_QA_PERF_PROFILE=all GD_QA_PERF_WARMUP=5 \
  GD_QA_PERF_ITERS=20 GD_QA_PERF_PIPELINE_ITERS=20 make gemm-perf-probe
```

## QA results

- `src/ops/matmul/fwd.py`: **PASS** — PyTorch loop reference and actual `gd_matmul` outputs matched PyTorch for:
  - fallback small shape `(4,7)x(7,6)`
  - hidden 128/256/512 `qkv`, `proj`, `mlp_up`, `mlp_down`
  - hidden 128/256/512 single-head attention score/value GEMM shapes
- `src/ops/matmul/bwd.py`: **PASS** — actual `gd_matmul_backward` matched PyTorch F16-rounded `grad_x` and `grad_w` for the same fallback and classic transformer shapes.
- `v2_matmul_training_perf_probe`: **PASS** — public forward+backward matmul-only performance measured with no warnings.

## GEMM kernel tuning summary

Current production path uses the offline Metal `gradients.metallib` and register-tiled simdgroup-matrix F16 GEMM kernels for aligned classic shapes. Forward uses NN GEMM; matmul backward adds high-performance NT/TN kernels:

- forward: `Y = X @ W`
- backward input gradient: `grad_x = grad_out @ W^T` via NT GEMM, without materializing `W^T`
- backward weight gradient: `grad_w = X^T @ grad_out` via TN GEMM, without materializing `X^T`

Selected tuned path:

- per-simdgroup output tile: `32 x 32`
- simdgroups per threadgroup: `4`, covering up to `32 x 128` output per threadgroup
- threads per threadgroup: `128`
- K step: `8`
- forward loads: direct device `simdgroup_load`
- backward transposed loads: direct per-lane register-fragment loads for NT/TN sources; no transpose/copy kernels
- compute: `4 x 4` FP32 simdgroup-matrix accumulators per simdgroup with serpentine multiply order
- store: direct per-lane `thread_elements()` epilogue to F16, with optional bias for linear forward
- fallback: scalar threadgroup-memory tiled kernels for non-multiple/bounds cases, including NT/TN backward

Tried and rejected in this tuning pass:

- one-simdgroup `64 x 64` register tile (`NBLK=8`): too much register pressure; slower across profiles.
- `8` simdgroups per threadgroup: slower than `4` on the transformer profiles.
- MLX-style staged `32 x 32` / `4`-simdgroup threadgroup-memory tile: correct, but slower than direct register-tiled loads on these shapes.

## Public API metallib GEMM forward+backward performance (`-O3`, wall time)

Default perf mode runs thirteen forward F16 matmuls and then `gd_matmul_backward` for each, yielding 39 GEMM calls per scoped arena step. Wall time includes public API validation, scratch/data arena allocation, command-buffer submit, and synchronization. It exercises the production `gradients.metallib` path, not MPS descriptor/object creation.

| profile | tokens | hidden | heads | fwd+bwd flops/step | sync step | sync TFLOP/s | pipelined step | pipelined TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h128x4h4 | 512 | 128 | 4 | 1.208G | 1.879 ms | 0.64 | 1.239 ms | 0.98 |
| h256x4h4 | 512 | 256 | 4 | 4.027G | 2.927 ms | 1.38 | 2.283 ms | 1.76 |
| h512x4h4 | 512 | 512 | 4 | 14.496G | 6.878 ms | 2.11 | 6.173 ms | 2.35 |

Best observed public matmul forward+backward throughput in this run: **2.35 TFLOP/s** pipelined on `h512x4h4`.

Forward-only mode remains available with `GD_QA_PERF_BACKWARD=0`; the same build measured about **2.31 TFLOP/s** pipelined on `h512x4h4` in this run.

## Verdict

`gd_matmul_backward` is now implemented and correctness-checked against PyTorch. The Metal backend uses metallib-backed NN/NT/TN register-tiled simdgroup GEMM kernels and avoids hot-path MPS/object allocation and transpose materialization.

The stack is still **not complete high-performance training** because:

1. `gd_linear_backward` still returns `GD_ERR_NOT_IMPLEMENTED`.
2. The measured path is matmul-only; it excludes autograd graph traversal, optimizer updates, AMP scaler work, loss kernels, softmax, activations, and normalization kernels.
3. Public matmul remains F16, 2D, row-strided only. There is no public batched/transpose matmul API for full attention workloads.
4. Current timings are wall-clock and include public API and queue effects; per-kernel GPU timestamps are still needed for finer tuning.

Recommended next steps:

1. Implement `gd_linear_backward`, including a high-performance bias-gradient reduction.
2. Add public transposed/batched GEMM forms so attention and backward can be expressed directly by callers.
3. Add per-kernel GPU timestamp instrumentation in the Metal backend for tuning NN/NT/TN separately.
4. Continue tile-shape tuning once backward/attention workloads are represented directly.
