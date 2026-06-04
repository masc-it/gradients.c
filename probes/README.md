# v2 probes

Temporary standalone programs for validating v2 design choices before consolidation.

## v2_foundation_probe

Checks foundation invariants without v1 headers or real kernels:

- params/state/scratch/data arenas
- aligned arena spans, OOM status, reset generations, sealed params
- no heap allocation in scoped hot path
- scratch/data ring slots + fake async fences
- `gd_begin` / `gd_end` lifecycle
- concrete tensors, storage allocation vs view offsets, non-contiguous `slice`, explicit `contiguous`
- `Module`, `ModuleList`, `ModuleDict`
- named parameter traversal, tied-weight dedup, param groups
- checkpoint manifest shape for model/optimizer/scaler/LR scheduler

Run:

```sh
mkdir -p build/probes
cc -std=c11 -Wall -Wextra -Werror -pedantic \
  probes/v2_foundation_probe.c -o build/probes/v2_foundation_probe
build/probes/v2_foundation_probe
```

Sanitizer run:

```sh
cc -std=c11 -Wall -Wextra -Werror -pedantic \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  probes/v2_foundation_probe.c -o build/probes/v2_foundation_probe_asan
build/probes/v2_foundation_probe_asan
```

## v2_matmul_training_perf_probe

Links against `libgradients.a` and measures the public `gd_linear`,
`gd_linear_backward`, `gd_matmul`, and `gd_matmul_backward` paths on
transformer-like workloads. This exercises the production Metal
`gradients.metallib` F16 register-tiled simdgroup GEMM kernels and linear bias
reductions through the scoped public API. It includes a small initialized
correctness smoke test, then runs optimized profiles that mimic training-step
GEMM pressure:

- `h128x4h4`: `tokens=512 hidden=128 heads=4 intermediate=512`
- `h256x4h4`: `tokens=512 hidden=256 heads=4 intermediate=1024`
- `h512x4h4`: `tokens=512 hidden=512 heads=4 intermediate=2048`

Each forward profile submits thirteen F16 GEMM-like ops in a scoped arena step:
fused-bias dense projection/MLP linears (`qkv`, `proj`, `gate`, `up`, `down`)
plus two attention matmul shapes per head (`QK^T`, `AV`) for four heads. By
default the probe also runs the corresponding backward op after each forward op,
adding `grad_out @ W^T`, `X^T @ grad_out`, and five bias-gradient reductions.
Wall time intentionally includes public API validation, scratch/data arena
allocation, command-buffer submit, and wait. The resulting benchmark is still
GEMM/linear-only: it does not include autograd graph traversal, optimizer, AMP
scaler, softmax, losses, normalization, or activation kernels.

Run directly with an optimized library/metallib build:

```sh
make gemm-perf-probe
```

Or run via the stress script:

```sh
sh probes/run_v2_qa_stress.sh
```

Knobs:

```sh
GD_QA_PERF_PROFILE=all|h128x4h4|h256x4h4|h512x4h4
GD_QA_PERF_WARMUP=5
GD_QA_PERF_ITERS=20
GD_QA_PERF_PIPELINE=1
GD_QA_PERF_PIPELINE_ITERS=20
GD_QA_PERF_BACKWARD=1       # include gd_matmul_backward after each forward matmul
GD_QA_PERF_RING_SLOTS=3
RUN_PERF=0                  # skip public API perf probe
RUN_METAL_BASELINE=0        # skip standalone raw MPS baseline
PERF_BUILD_DIR=build-qa-probes-perf
```

The stress script builds the performance library separately with `-O3 -DNDEBUG`
(default overrideable via `PERF_CFLAGS`, `PERF_OBJCFLAGS`, and
`PERF_PROBE_CFLAGS`) so performance results are not accidentally taken from the
Makefile's debug-oriented default build.

## v2_elementwise_reduce_perf_probe

Links against `libgradients.a` and measures the public binary elementwise,
broadcasting, reduction, and scalar-loss autograd paths added for in-graph MSE:

- direct F16/F32 `gd_add`, `gd_sub`, `gd_mul`
- row/vector broadcast F16/F32 `gd_add` / `gd_mul` (`[tokens, hidden]` with `[hidden]`)
- all-elements `gd_reduce_sum` and `gd_reduce_mean`
- row-wise `gd_reduce_sum_axis(..., axis=1)` and `gd_reduce_mean_axis(..., axis=1)`
- row-wise `reduce_mean_axis -> backward` for reduced-axis broadcast gradients
- full `sub -> mul -> reduce_mean -> backward` MSE graph

Profiles use activation-sized tensors such as `4096x1024` and `8192x2048`.
Wall time includes public validation, scratch arena allocation, command-buffer
submit, and synchronization.

Run:

```sh
make elementwise-reduce-perf-probe
```

Knobs:

```sh
GD_QA_ELEM_PROFILE=all|h1024_f16|h2048_f16|h1024_f32
GD_QA_ELEM_WARMUP=3
GD_QA_ELEM_ITERS=10
PERF_BUILD_DIR=build-perf
```

## v2_metal_arena_probe

Checks device-memory mapping for v2 arenas on Metal:

- shared-only arena buffers
- suballocated tensor refs with Metal buffer offsets
- compute kernels reading/writing suballocated refs
- direct CPU/GPU access through shared storage
- F16 MPS matrix multiplication using arena suballocations
- F16 batched MPS matrix multiplication with explicit `matrices` / `matrixBytes`
- F16 strided/sliced MPS matrix multiplication with MPS matrix origins
- F16 MPS linear followed by custom Metal bias/GELU in the same command buffer
- F16 RMSNorm/LayerNorm row reductions with FP32 accumulators
- transformer-sized F16 MPS GEMM benchmark with deterministic random inputs/weights
- batched attention-like F16 MPS GEMM benchmark
- scratch/data ring slots backed by command-buffer fences
- state object reset waiting on in-flight command buffer

Run:

```sh
mkdir -p build/probes
clang -fobjc-arc -Wall -Wextra -Werror \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  probes/v2_metal_arena_probe.m -o build/probes/v2_metal_arena_probe
build/probes/v2_metal_arena_probe
```

Sanitizer run:

```sh
clang -fobjc-arc -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  probes/v2_metal_arena_probe.m -o build/probes/v2_metal_arena_probe_asan
ASAN_OPTIONS=abort_on_error=1 build/probes/v2_metal_arena_probe_asan
```

Benchmark knobs:

```sh
GD_PROBE_MPS_WARMUP=3 GD_PROBE_MPS_ITERS=10 build/probes/v2_metal_arena_probe
```

Allocator/layout knobs used by benchmark paths:

```sh
GD_PROBE_BENCH_PROFILE=256h4              # only hidden=256, heads=4 shapes
GD_PROBE_BENCH_HAZARD=tracked|untracked    # default tracked
GD_PROBE_ALLOC_ALIGN=256                   # power-of-two suballoc alignment
GD_PROBE_BENCH_PREFIX_PAD=13               # force nonzero suballoc offsets
GD_PROBE_ROW_PAD_BYTES=128                 # extra bytes per row stride
GD_PROBE_BATCH_PAD_ROWS=1                  # extra rows between batched matrices
GD_PROBE_SCRATCH_RING_SLOTS=3              # scratch ring depth, clamped 1..64
GD_PROBE_DATA_RING_SLOTS=3                 # data ring depth, clamped 1..64
GD_PROBE_SCRATCH_SLOT_BYTES=67108864       # bytes per scratch ring slot, min 4096
GD_PROBE_DATA_SLOT_BYTES=8388608           # bytes per data ring slot, min 1024
```

If no Metal device exists, probe prints skipped and exits 0.
