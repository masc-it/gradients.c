# v2 QA stress and performance report

Date: 2026-06-04
Scope: public F16 linear/GEMM forward+backward through `gd_linear`, `gd_linear_backward`, `gd_matmul`, and `gd_matmul_backward`.
Device used for sample performance run: Apple M1 Pro, unified memory, Metal metallib backend.

## Probe coverage added / updated

- `src/ops/matmul/fwd.py` and `src/ops/matmul/bwd.py` execute actual `gd_matmul` / `gd_matmul_backward` and compare against PyTorch for fallback and classic transformer GEMM shapes.
- `src/ops/linear/fwd.py` executes actual `gd_linear` with both `bias=NULL` and fused bias, comparing against PyTorch.
- `src/ops/linear/bwd.py` executes actual `gd_linear_backward` with and without bias, comparing `grad_x`, `grad_w`, and optional `grad_bias` against PyTorch.
- `probes/v2_matmul_training_perf_probe.c` measures public API F16 fused-bias linears plus attention matmuls, with optional backward, on hidden sizes 128/256/512, MLP ratio 4, and 4 attention heads.

## Commands run

```sh
make check
GD_MATMUL_FWD_PROFILE=classic uv run src/ops/matmul/fwd.py
GD_MATMUL_BWD_PROFILE=classic uv run src/ops/matmul/bwd.py
GD_LINEAR_FWD_PROFILE=classic uv run src/ops/linear/fwd.py
GD_LINEAR_BWD_PROFILE=classic uv run src/ops/linear/bwd.py
GD_QA_PERF_BACKWARD=1 GD_QA_PERF_PROFILE=all GD_QA_PERF_WARMUP=5 \
  GD_QA_PERF_ITERS=20 GD_QA_PERF_PIPELINE_ITERS=20 make gemm-perf-probe
```

## QA results

- Matmul forward/backward: **PASS** against PyTorch F16-rounded references.
- Linear forward: **PASS** for optional-bias and fused-bias paths.
- Linear backward: **PASS** for `grad_x = grad_out @ W^T`, `grad_w = X^T @ grad_out`, and `grad_bias = sum_rows(grad_out)`.
- `make check`: **PASS**.
- Performance probe: **PASS** with no warnings.

## Kernel summary

Current production path uses the offline Metal `gradients.metallib` and avoids runtime shader compilation, MPS descriptors, transpose materialization, and hot-path ObjC allocation.

- Forward linear: `Y = X @ W + b` using NN register-tiled simdgroup GEMM with optional fused bias epilogue.
- Backward input gradient: `grad_x = grad_out @ W^T` using NT GEMM.
- Backward weight gradient: `grad_w = X^T @ grad_out` using TN GEMM.
- Backward bias gradient: one optimized F16 row-reduction kernel with FP32 accumulation and one simdgroup per output column.
- Fallback tiled NN/NT/TN kernels remain for non-aligned/small shapes.

Selected GEMM tile:

- per-simdgroup output tile: `32 x 32`
- simdgroups per threadgroup: `4`, covering up to `32 x 128` output per threadgroup
- threads per threadgroup: `128`
- K step: `8`
- FP32 simdgroup-matrix accumulation, F16 store

## Public API fused-linear + attention-GEMM fwd+bwd performance (`-O3`, wall time)

Default perf mode runs five fused-bias dense linears (`qkv`, `proj`, `gate`, `up`, `down`) plus eight attention matmuls per step. Backward adds the corresponding 26 GEMMs and five bias-gradient reductions, for 39 GEMM calls plus 5 reductions. Wall time includes public API validation, arena allocation, command-buffer submit, and synchronization.

| profile | tokens | hidden | heads | fwd+bwd flops/step | sync step | sync TFLOP/s | pipelined step | pipelined TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h128x4h4 | 512 | 128 | 4 | 1.208G | 1.818 ms | 0.66 | 1.322 ms | 0.91 |
| h256x4h4 | 512 | 256 | 4 | 4.027G | 2.986 ms | 1.35 | 2.555 ms | 1.58 |
| h512x4h4 | 512 | 512 | 4 | 14.496G | 7.056 ms | 2.05 | 6.810 ms | 2.13 |

Best observed public fused-linear/matmul forward+backward throughput in this run: **2.13 TFLOP/s** pipelined on `h512x4h4`.

## Verdict

`gd_linear` now supports optional bias and uses a fused GEMM epilogue when bias is present. `gd_linear_backward` is implemented with high-performance NT/TN GEMM plus a dedicated bias-gradient reduction.

Remaining work for full training performance:

1. Add public batched/transpose GEMM forms for direct attention and backward shapes.
2. Add per-kernel GPU timestamp instrumentation to tune NN/NT/TN/reduction separately.
3. Add/optimize non-GEMM training kernels: softmax, losses, activations, normalization, optimizer, and AMP scaler paths.
