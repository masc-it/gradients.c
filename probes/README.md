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
