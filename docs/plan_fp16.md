# Plan: FP16 mixed precision for training and inference

Status: planned

This plan turns the earlier checklist in `docs/todo_fp16_mixed_precision.md` into an
implementation plan with production gates. Goal is higher Metal throughput for GPT
training and inference while keeping CPU reference correctness and exact F32 as the
default path.

## Non-negotiable invariants

- [ ] F32 remains default behavior and reference path.
- [ ] FP16 is opt-in until correctness, fallback, and perf gates pass.
- [ ] No pure-FP16 training mode. Training is mixed precision only.
- [ ] FP16 storage is allowed; numerically sensitive accumulation stays F32.
- [x] Leaf gradients consumed by optimizers are F32.
- [x] AdamW state and master params are F32 for F16 model params.
- [ ] Loss scalar passed to `gd_backward()` is F32.
- [ ] Metal F16 graphs must fail closed if unsupported, not silently CPU-fallback.
- [ ] CPU_REF supports every F16 op claimed as public/supported, for parity tests.
- [ ] Measured performance decides rollout; no speculative fast-path default.

## Precision contracts

### Inference contract

- [x] Model params: F16.
- [x] Activations: F16 unless op explicitly returns F32 stats/loss.
- [ ] GEMM/linear: F16 input/output storage, F32 accumulation or backend-verified
      equivalent precision.
- [ ] RMSNorm/reductions/softmax/SDPA: F16 load/store with F32 reduction, stats,
      score, and normalization math.
- [x] LM head / lmCE: F16 hidden/weight storage with F32 logits math and F32 loss
      when a loss is requested.
- [ ] KV cache path, once added, stores F16 K/V by default and computes attention
      stats in F32.

### Training contract

- [ ] Model params used by forward/backward: F16.
- [x] Optimizer-owned master params: F32.
- [x] AdamW `m`/`v`: F32.
- [ ] Forward activations: F16 where safe.
- [ ] Backward activation grads: F16 allowed for internal flow.
- [x] Parameter gradients: F32 before accumulation into leaf grad slots.
- [ ] Reductions, norms, softmax, SDPA stats, CE/lmCE: F32 math.
- [x] Loss scaling: dynamic scaler scales F32 loss, unscales F32 grads, checks
      finite, skips optimizer step on NaN/Inf.
- [x] Optimizer step updates F32 master, then casts master -> F16 model param.

## Current gradients.c design fit

| Area | Fits today | Required before serious FP16 work |
|---|---|---|
| dtype model | `GD_DTYPE_F16` exists and tensor desc carries dtype | implement reliable F16 CPU helpers, cast, copy, materialization, tests |
| compute policy | `gd_compute_policy {compute_dtype, accum_dtype}` exists; matmul/linear attrs carry it | define exact semantics; extend non-GEMM kernels to honor F32 accumulation |
| autograd grad slots | `_gd_tensor_ensure_grad()` creates F32 slots | accumulate/cast to F32 before leaf grad writes; avoid F16 dW precision loss |
| matmul/linear | MPS plan abstraction exists | accept F16 descriptors; verify/guarantee F32 accumulation; support F32 grad outputs |
| optimizer | AdamW state tensors are already F32 | add F32 master params for F16 params; remove F32-param-only restriction safely |
| SDPA | kernels already keep score/stats in float | add typed F16 load/store kernels for fwd/bwd; keep exact full causal default |
| CE/lmCE | fused lmCE avoids huge logits materialization | force scalar loss F32; support F16 hidden/weight/logits with F32 math |
| backend support | op support hooks exist | add dtype-aware support matrix and no-hidden-fallback enforcement |
| GPT API | high-level GPT config exists | add param dtype / AMP config plumbing; keep default F32 |

Design is compatible, but not ready to start by only adding F16 kernels. Needed
foundational changes first: typed storage access, dtype-aware support checks,
clear gradient dtype rules, optimizer master ownership, and F32 loss/scaler flow.

## Proposed public/API shape

### Minimal v1 APIs

- [x] Add `gd_gpt_config.param_dtype` with default `GD_DTYPE_F32`.
- [ ] Keep `gd_context_set_compute_policy(ctx, {F16, F32})` as v1 autocast knob.
- [x] Add `GD_BENCH_DTYPE=f32|f16` to `gpt_bench`.
- [ ] Add `GD_BENCH_AMP=1` for training with dynamic loss scaling.
- [x] Extend `gd_adamw_config` with master-param policy:

```c
typedef enum gd_master_param_policy {
    GD_MASTER_PARAM_AUTO = 0,      /* F16 params use F32 master */
    GD_MASTER_PARAM_DISABLED = 1,  /* F32 params only */
    GD_MASTER_PARAM_ALWAYS = 2     /* all params update through F32 master */
} gd_master_param_policy;
```

- [x] Add AMP scaler API in `optim.h`:

```c
typedef struct gd_amp_scaler gd_amp_scaler;

typedef struct gd_amp_scaler_config {
    float init_scale;
    float growth_factor;
    float backoff_factor;
    int growth_interval;
    float min_scale;
    float max_scale;
} gd_amp_scaler_config;
```

### Deferred APIs

- [ ] Full autocast push/pop stack.
- [ ] BF16 default mode.
- [ ] FP8 training.
- [ ] Quantized optimizer or 8-bit optimizer state.

## Phase 0: design gate and support matrix

Must finish before broad kernel work.

- [ ] Define per-op dtype support matrix for CPU_REF and Metal.
- [ ] Add compile-time graph validation that rejects unsupported dtype/op/backend
      combos with clear errors.
- [ ] Add env/test mode to fail on CPU fallback for Metal graphs.
- [ ] Document exact `gd_compute_policy` semantics:
  - [ ] `compute_dtype` selects preferred storage dtype for eligible outputs.
  - [ ] `accum_dtype` selects internal accumulator dtype for GEMM/reductions/stats.
  - [ ] F16 training requires `compute_dtype=F16, accum_dtype=F32`.
- [ ] Decide F16 param initialization path:
  - [ ] either pack F32 init buffer to F16 before `gd_tensor_copy_from_cpu()`, or
  - [ ] create F32 temp params then cast to F16 params inside model creation.
- [ ] Add profiler tags for dtype path: F32 Metal, F16 MPS, F16 custom, CPU fallback.
- [ ] Establish baselines on current F32 head for:
  - [ ] GPT train `B=8 T=1024`, full causal.
  - [ ] GPT train `B=8 T=1024`, window 256.
  - [ ] GPT train `B=4 T=2048`, full causal.
  - [ ] GPT train `B=4 T=2048`, window 256.
  - [ ] GPT inference/forward-only equivalent once harness exists.

## Phase 1: F16 dtype primitives and CPU reference

Goal: F16 tensors are real tensors, not raw buffers that only some Metal kernels
understand.

- [x] Add internal IEEE binary16 conversion helpers:
  - [x] `_gd_f32_to_f16_bits(float)`.
  - [x] `_gd_f16_bits_to_f32(uint16_t)`.
  - [x] deterministic round-to-nearest-even behavior.
  - [x] NaN/Inf/subnormal tests.
- [ ] Extend CPU_REF typed load/store helpers:
  - [x] F32 load/store.
  - [x] F16 load/store through F32 compute.
  - [ ] BF16 helpers may exist but stay non-default.
- [x] Implement CPU `gd_cast` for F32<->F16 and F16<->F16.
- [x] Implement Metal `gd_cast` for F32<->F16 and F16<->F16.
- [x] Fix `gd_copy` for dtype-sized copies, not fixed 32-bit raw copies.
- [x] Add tensor materialization tests:
  - [x] F16 tensor storage size is 2 bytes/elem.
  - [x] F32 -> F16 -> F32 tolerance.
  - [x] CPU/Metal cast parity.
  - [x] materialize virtual F16 tensor to CPU.
- [x] Add raw-copy docs: `gd_tensor_copy_from_cpu()` remains raw bytes; typed
      conversion uses `gd_cast` or new helper if added.

## Phase 2: loss dtype and scalar correctness

Goal: `gd_backward()` keeps scalar F32 contract even with F16 models.

- [x] Change CE meta so scalar loss output dtype is F32 regardless of logits dtype.
- [x] Change lmCE meta so output 0 loss is F32 regardless of hidden/weight dtype.
- [x] Keep lmCE auxiliary outputs (`m`, `l`, target logits or row stats) F32.
- [x] Update CPU CE/lmCE kernels to load F16 logits/hidden/weight and compute F32:
  - [x] CE F16 logits -> F32 loss.
  - [x] lmCE F16 hidden/weight -> F32 loss.
- [x] Update Metal CE/lmCE kernels to load F16 and compute F32:
  - [x] CE F16 logits -> F32 loss.
  - [x] lmCE F16 hidden/weight -> F32 loss.
- [ ] Add tests:
  - [x] F16 logits -> F32 CE loss.
  - [x] F16 hidden/weight -> F32 lmCE loss.
  - [ ] `gd_backward(ctx, loss)` accepts F16-model F32 loss.
  - [x] CE loss parity vs F32 baseline within tolerance.
  - [x] lmCE loss parity vs F32 baseline within tolerance.

## Phase 3: F16 MPS GEMM and fallback GEMM

Goal: unlock main FLOPS path first.

- [x] Extend `_gd_metal_plan_mps_gemm()` dtype mapping:
  - [x] F32 descriptors -> `MPSDataTypeFloat32` as today.
  - [x] F16 descriptors -> `MPSDataTypeFloat16`.
  - [x] row bytes and matrix bytes use `gd_dtype_sizeof()`.
  - [x] output descriptor may be F16 when backend supports it.
- [ ] Verify MPS behavior empirically:
  - [ ] F16 input/F16 output speed.
  - [ ] F16 input/F32 output support and speed.
  - [x] numerical error vs CPU F32 reference on small GEMMs.
  - [ ] whether accumulation precision is acceptable for training.
- [ ] If MPS cannot produce required F32 param grads efficiently, add custom Metal
      F16xF16->F32 GEMM for gradient outputs before claiming training perf.
- [x] Add dtype-aware MPS plan rejection with fallback only to supported Metal kernel.
- [ ] Add matmul/linear tests:
  - [x] F16 forward output parity vs F32 reference tolerance.
  - [ ] transposed weight path.
  - [ ] batched matmul path.
  - [ ] non-contiguous/offset rejection remains safe.
- [ ] Benchmark standalone GEMM sizes matching GPT:
  - [ ] `[B*T, d] x [d, 3d]` QKV.
  - [ ] `[B*T, d] x [d, d_ff]` MLP up/gate.
  - [ ] `[B*T, d_ff] x [d_ff, d]` MLP down.
  - [ ] lmCE chunk GEMMs.

## Phase 4: F16 forward inference kernel coverage

Goal: GPT inference graph runs fully on Metal with F16 params/activations and no
CPU fallback.

- [x] Elementwise kernels typed for F16:
  - [x] add/mul/scale.
  - [x] copy.
  - [x] ReLU/SiLU/GELU/PowLU forward.
- [x] Shape/data movement kernels typed for F16:
  - [x] transpose.
  - [x] reduce_to where needed.
  - [x] embedding forward.
- [x] RMSNorm forward:
  - [x] load F16 input/weight.
  - [x] compute sumsq/inv in F32.
  - [x] store F16 output.
- [x] RoPE forward:
  - [x] load/store F16.
  - [x] compute trig/rotate in F32.
- [x] Softmax forward if used outside SDPA:
  - [x] F32 max/sum.
  - [x] F16 output.
- [x] SDPA forward:
  - [x] full causal F16 path.
  - [x] sliding-window causal F16 path.
  - [x] generic bias/prefix path either F16-supported or rejected clearly.
  - [x] scores/stats/online softmax in F32.
  - [x] output F16.
  - Note: first F16 path uses generic tiled/split-K kernels; causal/window F16 specializations remain perf work.
- [x] Fused residual+RMSNorm path typed for F16 or disabled safely for F16.
- [x] Fused lmCE forward supports F16 hidden/weight with F32 stats/loss.
- [ ] Add GPT inference/forward-only harness if missing:
  - [x] `GD_BENCH_MODE=infer|forward|train` (current spelling: `fwd|train`).
  - [x] `GD_BENCH_DTYPE=f16`.
  - [ ] report fallback count.
- [ ] Acceptance for inference phase:
  - [ ] F16 GPT forward/loss parity vs F32 tolerance on tiny graph.
  - [ ] no CPU fallback on supported Metal graph.
  - [ ] measured speedup over F32 on at least one GPT target shape.

## Phase 5: autograd F32 gradient accumulation semantics

Goal: F16 training graph produces F32 leaf grads without losing precision first.

- [ ] Add helper to decide gradient accumulator dtype for a graph value:
  - [ ] external leaf requiring grad -> F32.
  - [ ] optimizer-consumed params -> F32.
  - [ ] internal activation -> value dtype unless op requires F32.
- [ ] Update `_gd_bwd_accumulate()`:
  - [ ] cast first contribution to accumulator dtype before storing.
  - [ ] cast later contributions before `gd_add()`.
  - [ ] ensure accumulated grad tensor dtype matches accumulator dtype.
- [ ] Update `_gd_bwd_accumulate_broadcast()`:
  - [x] reduce/cast to accumulator dtype for leaf params.
  - [x] preserve broadcast correctness.
- [x] Update final leaf write:
  - [x] copy if accumulated dtype equals grad slot dtype.
  - [x] cast if needed.
  - [x] assert leaf grad slot remains F32 for F16 params.
- [x] Matmul/linear backward policy:
  - [x] activation-input gradients may be F16 for memory/perf.
  - [x] parameter gradients must be F32.
  - [x] implement F16xF16->F32 dW path or explicit cast-to-F32 correctness path.
  - [x] avoid computing dW as F16 then widening; that loses training precision.
- [ ] Op-specific backward dtype fixes:
  - [x] RMSNorm backward: dx F16 allowed, dweight F32.
  - [x] embedding backward: dweight F32.
  - [x] lmCE backward: hidden grad F16 allowed, weight grad F32.
  - [x] SDPA backward: dq F16 allowed, dk/dv F16 for activations; param-facing grads F32 through projections.
  - [x] activations: preserve incoming grad dtype unless leaf accumulation requires F32.
- [ ] Tests:
  - [x] F16 param receives F32 grad slot.
  - [ ] two gradient paths accumulate into one F32 grad.
  - [x] broadcast grad accumulation casts/reduces correctly.
  - [x] matmul weight grad dtype is F32 under F16 training.
  - [ ] no accidental F16 leaf grad in GPT.
  - [x] embedding weight grad dtype is F32 under F16 training.

## Phase 6: AdamW master params

Goal: optimizer updates stable F32 master weights and refreshes F16 model params.

- [x] Extend optimizer slot:
  - [x] `param` model tensor, F32 or F16.
  - [x] optional `master` F32 tensor for F16 params.
  - [x] `m` F32.
  - [x] `v` F32.
- [x] Extend AdamW creation:
  - [x] F32 params use current in-place behavior.
  - [x] F16 params use F32 master under `GD_MASTER_PARAM_AUTO`.
  - [x] initialize master by casting param -> F32.
  - [x] deduplicate tied params correctly.
- [ ] Extend AdamW step:
  - [x] step consumes F32 grad.
  - [x] update master in F32.
  - [x] cast master -> model param after successful step.
  - [x] do not refresh model param when step is skipped by AMP.
- [x] Update CPU and Metal AdamW kernels if needed:
  - [x] F32 master path.
  - [x] F32 grads/state.
  - [x] F16 model param refresh kernel/cast.
- [ ] Tests:
  - [x] F16 param AdamW creates F32 master.
  - [x] master changes after step.
  - [x] F16 param refresh matches rounded master.
  - [x] F32 AdamW behavior unchanged.
  - [ ] tied embeddings share one master slot/update.

## Phase 7: AMP loss scaler, unscale, finite check, skip step

Goal: production-safe F16 training loop.

- [x] Implement `gd_amp_scaler` with defaults:
  - [x] initial scale `2^15` or `2^16`.
  - [x] growth factor `2.0`.
  - [x] backoff factor `0.5`.
  - [x] growth interval `1000` or `2000` finite steps.
  - [x] min scale `1.0`.
- [x] Implement `gd_amp_scaler_scale_loss()` using F32 loss.
- [x] Implement unscale + finite-check:
  - [x] in-place `grad *= 1 / scale` for F32 grads.
  - [x] detect NaN/Inf across all optimizer grads.
  - [x] Metal path reduces to GPU flag with one host read.
  - [x] CPU path deterministic.
- [ ] Integrate with optimizer:
  - [x] `gd_optimizer_step_amp()` wrapper.
  - [x] unscale before grad clipping.
  - [x] if found_inf: skip AdamW, skip master->param cast, backoff scale.
  - [x] if finite: AdamW, refresh F16 params, maybe grow scale.
- [ ] Tests:
  - [x] finite grads unscale correctly.
  - [ ] injected NaN skips step and backs off.
  - [x] injected Inf skips step and backs off.
  - [x] finite interval grows scale.
  - [x] zero grads after skipped step still works.

## Phase 8: F16 backward kernel coverage for GPT training

Goal: GPT F16+AMP training runs fully on Metal without unsupported fallback.

- [ ] Linear/matmul backward supports required dtype combinations:
  - [x] dX F16 output path.
  - [x] dW F32 output path.
  - [ ] tied LM weight grad F32.
- [x] RMSNorm backward typed F16/F32 as defined.
- [x] activation backward typed F16.
- [x] RoPE backward typed F16.
- [x] embedding backward accumulates F32 weight grad.
- [x] SDPA backward typed F16:
  - [x] full causal path.
  - [x] sliding-window path.
  - [x] F32 stats and softmax math.
  - [ ] validate dk/dv effects through following projection dW remain F32 at leaf.
- [x] lmCE backward typed F16 hidden + F32 weight grad.
- [x] `clip_grad_norm` consumes F32 grads and computes norm in F32.
- [ ] Add no-fallback GPT training test with small model.
- [ ] Add toy overfit test:
  - [ ] F16+AMP loss decreases.
  - [ ] no NaN under default scaler.
  - [ ] final loss close enough to F32 baseline for tiny run.

## Phase 9: integration with GPT and benchmarks

- [ ] Add GPT construction dtype:
  - [ ] default F32.
  - [ ] F16 params when requested.
  - [ ] F16 tied embedding/head handled correctly.
- [ ] Add benchmark flags:
  - [ ] `GD_BENCH_DTYPE=f32|f16`.
  - [ ] `GD_BENCH_AMP=0|1`.
  - [ ] `GD_BENCH_MODE=train|forward|infer`.
  - [ ] print precision policy and scaler state.
  - [ ] print CPU fallback count.
- [ ] Add profile breakdown by dtype path:
  - [ ] F16 MPS GEMM time.
  - [ ] F16 SDPA time.
  - [ ] AMP unscale/check time.
  - [ ] master refresh cast time.
- [ ] Benchmark target matrix:

| workload | mode | dtype | attention | required before checkoff |
|---|---|---|---|---|
| `B=8 T=1024` | train | F16+AMP | full causal | no fallback, loss finite, speedup measured |
| `B=8 T=1024` | train | F16+AMP | window 256 | no fallback, loss finite, speedup measured |
| `B=4 T=2048` | train | F16+AMP | full causal | no fallback, loss finite, speedup measured |
| `B=4 T=2048` | train | F16+AMP | window 256 | no fallback, loss finite, speedup measured |
| `B=8 T=1024` | infer | F16 | full/window | no fallback, speedup measured |
| `B=4 T=2048` | infer | F16 | full/window | no fallback, speedup measured |

## Phase 10: rollout gates

- [ ] Correctness gate:
  - [ ] `make test GD_ENABLE_METAL=0` passes.
  - [ ] `make test GD_ENABLE_METAL=1` passes.
  - [ ] F16-specific CPU tests pass.
  - [ ] F16-specific Metal tests pass.
  - [ ] no hidden fallback in F16 GPT supported graph.
- [ ] Numerical gate:
  - [ ] F16 inference loss/logits match F32 within documented tolerance.
  - [ ] F16+AMP toy MLP loss decreases.
  - [ ] F16+AMP toy GPT loss decreases.
  - [ ] Grad dtype assertions pass for all GPT params.
- [ ] Performance gate:
  - [ ] standalone F16 GEMM faster than F32 for GPT-relevant shapes.
  - [ ] F16 inference faster than F32 on target GPT shapes.
  - [ ] F16+AMP training faster than F32 after AMP overhead.
  - [ ] if speedup < 1.25x on target training shape, keep FP16 experimental.
- [ ] Production gate:
  - [ ] clear public docs and examples.
  - [ ] scaler state checkpointable once checkpointing exists.
  - [ ] failure modes have actionable errors.
  - [ ] default remains F32.

## Risk register

- [ ] MPS F16 accumulation may not match desired F32 accumulation.
  - Mitigation: empirical numeric tests; if insufficient, custom F16xF16->F32 GEMM
    for training gradients, MPS F16 only for inference/activation outputs.
- [ ] Leaf dW computed in F16 then widened would silently harm training.
  - Mitigation: dtype assertions on all param grads; op-specific dW F32 path.
- [ ] Hidden CPU fallback can erase perf wins.
  - Mitigation: no-fallback execution mode and profile counters.
- [ ] Loss scaling skip path can leave stale F16 params or stale grads.
  - Mitigation: optimizer wrapper owns step/refresh/zero-grad order; tests inject Inf.
- [ ] F16 exp/softmax overflow.
  - Mitigation: all max/sum/exp math in F32; F16 only at storage boundaries.
- [ ] Tied embeddings can get duplicate master params.
  - Mitigation: optimizer dedupe by tensor identity/storage; explicit tied-param test.
- [ ] CPU reference refactor can become broad.
  - Mitigation: support only claimed F16 ops first; unsupported F16 ops fail clearly.

## Definition of done

- [ ] F16 inference mode for GPT runs on Metal without CPU fallback and beats F32.
- [ ] F16+AMP training mode for GPT runs on Metal without CPU fallback and beats F32
      on at least one target shape.
- [ ] All params in F16 training have F32 grads, F32 AdamW states, and F32 master
      params.
- [ ] Dynamic loss scaling handles finite and non-finite steps correctly.
- [ ] CPU_REF remains a valid correctness oracle for all supported F16 ops.
- [ ] Docs list supported dtype/op/backend matrix and known exclusions.
- [ ] F32 default path perf and correctness do not regress.
