# TODO: FP16 mixed-precision training

Status: planned

Goal: support proper FP16 training on Metal without pretending pure-FP16 training is stable. Use FP16 for storage/throughput where useful, FP32 for accumulation, grads, optimizer state, and master weights.

## Summary

Proper FP16 training means **mixed precision**, not pure FP16:

- FP16 model params used by forward/backward kernels.
- FP32 accumulation for matmul, reductions, norms, softmax, CE, SDPA stats.
- FP32 gradients for leaf grad slots and optimizer input.
- FP32 AdamW state (`m`, `v`) and FP32 master params.
- Dynamic loss scaling with finite-check and skipped optimizer steps.
- FP32 master update then cast master params back to FP16 model params.

Current design has right hooks (`gd_dtype`, `gd_compute_policy`, F32 grad slots, F32 optimizer state), but implementation is not complete.

## Current state

Already present:

- `gd_dtype` includes `GD_DTYPE_F16` and `GD_DTYPE_BF16`.
- `gd_compute_policy { compute_dtype, accum_dtype }` exists.
- `gd_matmul_desc` / `gd_linear_desc` carry compute policy.
- `_gd_tensor_ensure_grad()` creates F32 grad slots.
- AdamW state tensors are F32.

Blocking gaps:

- CPU_REF kernels require F32.
- Metal kernels mostly use `float*` and do not dispatch typed F16 kernels.
- AdamW rejects non-F32 params: `adamw v1 supports F32 parameters only`.
- Grad accumulation can dtype-fail when F16 backward contrib meets F32 grad slot.
- CE/lmCE infer scalar loss with logits dtype; backward currently expects scalar F32 loss.
- Cast path does not fully support F32 <-> F16.
- No loss scaler API.
- No finite-check/unscale/skip-step path.
- No FP32 master weight ownership in optimizer.

## Target training flow

```text
params_f16        = model params used by forward/backward
master_params_f32 = optimizer-owned source of truth

loss_f32 = forward_loss(params_f16)
scaled_loss = loss_f32 * scale
backward(scaled_loss)

unscale all grads_f32: grad *= 1 / scale
found_inf = any_nonfinite(grads_f32)

if !found_inf:
    optional grad clipping on grads_f32
    AdamW(master_params_f32, grads_f32, m_f32, v_f32)
    cast master_params_f32 -> params_f16
    maybe grow scale
else:
    skip optimizer step
    scale *= backoff

zero grads
```

## Loss scaling policy

Add dynamic scaler with defaults:

- initial scale: `2^15` or `2^16`
- growth factor: `2.0`
- backoff factor: `0.5`
- growth interval: `1000` or `2000` finite steps
- min scale: `1.0`
- max scale: configurable, default no practical cap beyond finite float

Needed API sketch:

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

gd_status gd_amp_scaler_create(gd_context *ctx,
                               const gd_amp_scaler_config *config,
                               gd_amp_scaler **out);
void gd_amp_scaler_destroy(gd_amp_scaler *scaler);

float gd_amp_scaler_scale(const gd_amp_scaler *scaler);

gd_status gd_amp_scale_loss(gd_context *ctx,
                            gd_amp_scaler *scaler,
                            gd_tensor *loss_f32,
                            gd_tensor **scaled_loss_out);

gd_status gd_amp_unscale_grads(gd_context *ctx,
                               gd_amp_scaler *scaler,
                               gd_tensor **params,
                               int n_params,
                               bool *found_inf_out);

gd_status gd_amp_update(gd_amp_scaler *scaler, bool found_inf);
```

Optimizer integration can later wrap this:

```c
gd_status gd_optimizer_step_amp(gd_context *ctx,
                                gd_optimizer *optimizer,
                                gd_amp_scaler *scaler,
                                bool *stepped_out);
```

## Required implementation phases

### Phase 1: dtype/cast correctness

- Implement F32 -> F16 and F16 -> F32 casts in CPU_REF and Metal.
- Make `gd_cast` support float-to-float casts.
- Add tests:
  - F32 -> F16 -> F32 tolerance.
  - F16 tensor storage size and materialization.
  - Metal/CPU parity for casts.

### Phase 2: scalar loss must be F32

- Change `cross_entropy` and `lm_cross_entropy` inference to return scalar F32 regardless of logits/hidden dtype.
- Ensure CE/lmCE kernels load logits/hidden in storage dtype but compute loss in FP32.
- Ensure `gd_backward(ctx, loss)` still accepts scalar F32.
- Add tests:
  - F16 logits -> F32 loss.
  - F16 hidden/weight lmCE -> F32 loss.

### Phase 3: F16 Metal kernels with FP32 accumulation

Add typed Metal kernels or dtype-specialized variants:

- elementwise: F16 load/store, usually compute float for nonlinear ops.
- matmul/linear: F16 inputs, FP32 accum, F16 output.
- reductions: FP32 accumulation.
- RMSNorm: FP32 mean-square/reduction, F16 output.
- softmax: FP32 max/sum, F16 output.
- CE/lmCE: FP32 logits math/loss.
- SDPA: FP32 score/stats/softmax accumulation, F16 output.

Use backend dispatch by dtype. Avoid hidden CPU fallback.

### Phase 4: F32 gradient accumulation

Current leaf grad slots are F32; keep that.

Needed changes:

- When backward contribution dtype != target grad dtype, insert cast to F32 before `accumulate()` / `accumulate_broadcast()` writes leaf grads.
- Internal backward tensors may be F16 for memory, but leaf grads and optimizer grads must become F32.
- Matmul backward should use compute policy with FP32 accumulation.

Tests:

- F16 param receives F32 grad slot.
- Multiple gradient paths accumulate into F32 grad.
- Broadcast grad accumulation casts/reduces correctly.

### Phase 5: AdamW master params

Extend optimizer so FP16 params are allowed:

- `param`: model param, may be F16.
- `master`: optimizer-owned F32 tensor same shape as param.
- `m/v`: F32 tensors.
- Step uses `master` and F32 grad.
- After finite step, cast `master` -> `param`.

Config sketch:

```c
typedef enum gd_master_param_policy {
    GD_MASTER_PARAM_NONE,      /* F32 params only */
    GD_MASTER_PARAM_F32_COPY   /* for F16/BF16 params */
} gd_master_param_policy;
```

Behavior:

- F32 params may update in-place as today.
- F16 params require F32 master copy.
- Optimizer creation initializes master from param cast to F32.

### Phase 6: unscale + finite check kernels

Add ops/kernels:

- `UNSCALE_GRAD`: in-place `grad *= inv_scale`.
- `CHECK_FINITE`: reduce tensor(s) to found_inf flag.
- Optionally fuse unscale + check finite.

Metal should reduce per tensor to small scratch/flag, then host reads one flag or next kernel consumes it.

Tests:

- finite grads unscale correctly.
- NaN/Inf detected.
- optimizer step skipped on found_inf.
- scale growth/backoff works.

### Phase 7: compute policy/autocast ergonomics

Use existing `gd_context_set_compute_policy()`:

```c
gd_compute_policy fp16_policy = {
    .compute_dtype = GD_DTYPE_F16,
    .accum_dtype = GD_DTYPE_F32,
};
gd_context_set_compute_policy(ctx, fp16_policy);
```

Potential later API:

```c
gd_status gd_autocast_push(gd_context *ctx, gd_compute_policy policy);
gd_status gd_autocast_pop(gd_context *ctx);
```

For v1, context-level policy is enough if tests are explicit.

## Metal/M1 Pro notes

- FP16 is worth targeting on M1 Pro.
- BF16 is not primary target on M1 Pro: MPSMatrix rejects BF16, CPU lacks `FEAT_BF16`, and observed BF16 matmul paths are slower than optimized F16.
- Use F16 + FP32 accum as default mixed-precision target.
- Keep BF16 optional/benchmark-gated for M2+ later.

## Acceptance criteria

- F16 model forward loss decreases on small MLP and GPT toy run.
- Leaf grads for F16 params are F32.
- AdamW with F16 params updates F32 master and refreshes F16 params.
- Dynamic loss scaling skips update on injected Inf/NaN and backs off scale.
- Scale grows after configured finite interval.
- Metal backend runs without CPU fallback for supported F16 training graph.
- F16 mixed training matches F32 baseline within expected tolerance on tiny tests.

## Non-goals for first pass

- FP8 training.
- BF16 as default on M1/M1 Pro.
- Distributed/sharded optimizer.
- Perfect kernel peak performance before correctness.
- Full autocast op whitelist/blacklist system.
