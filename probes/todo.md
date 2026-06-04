# v2 probe TODO

Stress workloads to validate v2 design before consolidation.

## Findings so far

- Metal tensor arenas are shared-only. Private storage/staging path removed.
- ML stress probes use F16 tensor storage; FP32 remains only for CPU reference math, accumulators, optimizer/sensitive future probes.
- Foundation memory contract now tracks aligned spans, reset generations, sealed params, descriptor/storage lifetimes, and no scoped hot-path heap allocation.
- Library tensor descriptor foundation now mirrors probe semantics: dtype/shape/strides, storage span + view offset, slice views, explicit contiguous output allocation, and stale ring generation validation.
- Library transfer foundation uses backend upload/download helpers over spans/tensors; Metal implementation touches shared storage inside backend only and waits relevant slot/persistent fences.
- Library tensor init helpers now use Metal command-scope kernels: `empty` is uninitialized, `zeros`/`ones` are backend fills, `rand`/`rand_uniform` are deterministic backend RNG fills.
- Library op capsules started with `matmul` and `linear`: forward/backward now use metallib-backed custom F16 GEMM kernels; linear has optional fused bias epilogue plus bias-gradient reduction.
- Fixed probe F32→F16 conversion rounding-carry bug; values near exponent boundaries like `0.124987` must round to `0.125`, not `0.0625`.
- Best 256h4 defaults: shared storage, tracked hazards, 256B suballoc alignment, compact MPS-recommended `rowBytes`, tight `matrixBytes`.
- Recommended 256h4 ring defaults: scratch 3 slots x 64 MiB, data 3 slots x 8 MiB.
- Ring depth/capacity affects OOM and wait frequency, not GEMM throughput directly.
- Padding row strides or batched matrix gaps gives no reliable win and can hurt small H=256 shapes.
- Generic MPS batched GEMM is acceptable for layout validation, but skinny attention-V shapes underutilize MPS; custom `sdpa_varlen` remains needed.
- Binary elementwise direct kernels on real activation tensors reach memory-bandwidth-like throughput through public API; generic broadcast was initially too slow, so row/vector broadcast now has a 2D specialized Metal path.
- All-elements reductions use multi-stage simdgroup contiguous kernels; current public API performance is acceptable for scalar losses and MSE graph stress.
- Axis reductions now use dedicated simdgroup Metal kernels and optimized reduced-axis broadcast for backward; continue watching non-last-axis strided cases as normalization workloads expand.

## Device memory / MPS layout

- [x] Batched MPS matmul
  - `[B,M,K] x [B,K,N] -> [B,M,N]`
  - use `MPSMatrixDescriptor` with `matrices` / `matrixBytes`
  - validate batched strides, matrix offsets, arena suballocations
  - includes batched attention-like F16 perf benchmark

- [x] Strided/sliced MPS matmul
  - rowBytes larger than compact row
  - use MPS matrix origins where possible
  - validate non-contiguous view semantics and MPS origin mapping
  - finding: `MPSMatrixMultiplication` origins use `MTLOriginMake(row, column, 0)`, not `(column, row, 0)`

## Transformer block pieces

- [x] Linear + bias + activation in one command buffer
  - MPS matmul then custom Metal bias/GELU kernel
  - validate MPS + custom kernels interleave without command boundary
  - finding: MPS encode followed by custom compute encoder in same command buffer preserves order and works with shared arena offsets
  - finding: device tensors should be F16; CPU reference computes in FP32 then rounds to F16 for comparison

- [ ] Fused-ish MLP block
  - `gate_up = X @ Wgu`
  - custom SiLU/gate split kernel
  - `down = hidden @ Wd`
  - validate multiple large scratch allocations in one scope

- [ ] QKV projection + split views
  - `X @ Wqkv -> [M,3D]`
  - metadata views for q/k/v slices
  - custom kernel reads q/k/v slice offsets without copies

- [x] RMSNorm / LayerNorm row reductions
  - D = 256 / 512 / 768
  - FP32 accumulators, F16 storage
  - validate row-wise reduction pattern and numeric stability
  - finding: 256-thread row reductions pass for D=256/512/768 with padded rowBytes and shared arena offsets

## VLM-specific layout

- [ ] Concat + suffix slice
  - image prefix + text suffix -> compact hidden
  - suffix slice view passed to LMCE-shape checker kernel
  - validate VLM layout without suffix contiguous copy

- [ ] Suffix CE / LMCE toy kernel
  - suffix-only logits/loss on small vocab
  - no full `[B,img_tokens+text_tokens,V]` materialization
  - validate prefix hidden receives zero LMCE gradient contribution

## Embedding / tied weights

- [ ] Embedding gather
  - token IDs in shared data
  - embedding weights in shared params
  - gather to scratch
  - validate integer data + random access into shared params

- [ ] Embedding backward scatter-add with duplicates
  - repeated token IDs
  - custom scatter-add grad accumulation
  - validate additive grad semantics for tied embeddings

## KV cache / generation state

- [ ] KV cache append + decode read
  - K/V cache in `state_arena`
  - write at by-value `start_pos`
  - read prefix/window positions
  - validate state side effects and no mutable cache-pos tensor

- [ ] State reset with real KV workload in flight
  - replace spin-only state fence test with KV append/read workload
  - validate command-buffer retention and state-object fences

- [ ] Device-side argmax / top-k sample
  - logits in scratch
  - next token in state/data
  - no CPU readback
  - validate generation loop output path

## Training control / persistent state

- [x] AdamW optimizer + AMP step path
  - FP32 moments in state arena
  - FP32 master weights for F16 params
  - loss-scale-aware grad unscale, finite check, overflow skip, dynamic scale update

- [ ] Multi-head loss sum
  - two scalar losses in scratch
  - weighted sum kernel
  - one total scalar for backward seed
  - validate multi-loss flow shape
