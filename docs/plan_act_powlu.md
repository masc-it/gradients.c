# gradients.c — PowLU Activation Plan

Status: implemented through P3 (2026-05-31); P4 profiling pending

Paper path: `data/papers/2605.25704/document.md`

Goal: replace GPT MLP's SwiGLU activation path with **PowLU** (Power Linear Unit),
with a scalar CPU_REF oracle and optimized Metal forward/backward kernels.

Non-goals: fp16/bf16/FP8 kernels, MoE routing, fused gate/up/down GEMM, or removing
all legacy SwiGLU tests in the first patch. Keep SwiGLU available until PowLU parity
and training tests are stable.

Paper basis:
- SwiGLU behaves like `x^2` for large positive inputs, which amplifies outliers and
  hurts low-precision training stability (paper L5, L15).
- PowLU uses a rational power term and sigmoid to preserve non-linearity while
  limiting large-positive growth (paper L25, L57-L63).
- Training form is `PowLU(x1, x2) = x1 * f(x2)` where `x1` and `x2` are two linear
  projections; for `x2 > 0`, `f(x2)=x2^(m/(sqrt(x2)+1))*sigmoid(x2)`, and for
  `x2 <= 0`, `f(x2)=SiLU(x2)` (paper L59).
- Paper uses `m=3.0`; valid monotonic range is `0 < m < 10` (paper L57, L67,
  L73, L85, L240-L242).
- Expected benefit is lower forward/backward outliers and fewer loss spikes while
  retaining similar benchmark performance (paper L31, L119, L148, L181-L185,
  L199-L221).

---

## 1. Semantics

Define fused two-input PowLU as:

```text
out = x1 * f(x2)
```

where `x1` is the value projection (`up`) and `x2` is the gate projection
(`gate`). This matches the paper's `x1 * f(x2)` and maps cleanly onto the current
SwiGLU path:

```text
old: up * silu(gate)
new: up * powlu_gate(gate, m)
```

Gate function:

```text
sigmoid(z) = 1 / (1 + exp(-z))

f(z, m) =
  z^a * sigmoid(z),  a = m / (sqrt(z) + 1),  z > 0
  z * sigmoid(z),                              z <= 0
```

Full single-input expression in the paper is equivalent to `x * f(x)`; our MLP
uses the two-projection gated form.

Default hyperparameter:

```text
m = 3.0f
```

Validation policy:

```text
0 < m < 10
```

Reject `m <= 0`, `m >= 10`, NaN, or Inf in public API/model config.

---

## 2. Backward math

For `out = x1 * f(x2)` and upstream `go`:

```text
dx1 = go * f(x2)
dx2 = go * x1 * f'(x2)
```

For `z <= 0`:

```text
s = sigmoid(z)
f(z)  = z * s
f'(z) = s * (1 + z * (1 - s))      # SiLU derivative
```

For `z > 0`:

```text
r = sqrt(z)
a = m / (r + 1)
g = pow(z, a)
s = sigmoid(z)

# da/dz
da = -m / (2 * r * (r + 1)^2)

# d/dz pow(z,a(z)) = pow(z,a) * (a/z + da*log(z))
f'(z) = g * s * (a/z + da*log(z) + (1 - s))
```

Numerical guard:

- Branch on `z <= 0` before `sqrt/log/pow`.
- For `z > 0`, compute in f32 and use `log(max(z, 0x1p-126f))` / `sqrt(max(z,
  0.0f))` in kernels to avoid accidental `log(0)` if compiler speculates.
- Do not clamp normal positive inputs; CPU and Metal must match branch behavior.
- At `z == 0`, use negative/SiLU branch; output and derivative are both 0, matching
  paper differentiability.

---

## 3. Public API / config

### 3.1 Ops API

Add to `include/gradients/ops.h`:

```c
gd_status gd_powlu(gd_context *ctx,
                   gd_tensor *x1,      /* value/up projection */
                   gd_tensor *x2,      /* gate projection */
                   float m,
                   gd_tensor **out);
```

Rules:

- `x1` and `x2` same dtype/device.
- Floating-point only in v1 (F32 in existing CPU/Metal runtime).
- Shapes must be equal in v1. No broadcasting for optimized path.
- Output shape equals `x1`.

Do **not** add a unary public `gd_powlu_gate` in v1. The fused two-input op is the
model contract and avoids materializing the gate activation.

### 3.2 GPT config

Update `include/gradients/nn.h`:

```c
typedef enum gd_gpt_mlp_kind {
    GD_GPT_MLP_POWLU = 0, /* down(powlu(up(x), gate(x), m)) */
    GD_GPT_MLP_SWIGLU,    /* legacy */
    GD_GPT_MLP_GELU
} gd_gpt_mlp_kind;

float powlu_m; /* 0 => 3.0 */
```

Compatibility choice:

- New default in examples/benches/tests: `GD_GPT_MLP_POWLU`.
- Keep `GD_GPT_MLP_SWIGLU` for ablation and old tests.
- Decision: accept enum ABI churn and make `GD_GPT_MLP_POWLU=0` the new default;
  explicit `GD_GPT_MLP_SWIGLU` remains the legacy ablation path.

### 3.3 Model assembly

Change `src/nn/nn.c` MLP block:

```text
n = rms_norm(h)
gate = linear(n, w_gate)
if POWLU:
    up = linear(n, w_up)
    hh = gd_powlu(up, gate, powlu_m)
elif SWIGLU:
    act = gd_silu(gate)
    up = linear(n, w_up)
    hh = gd_mul(act, up)
elif GELU:
    hh = gd_gelu(gate)
down = linear(hh, w_down)
out = h + down
```

Parameter layout remains identical to SwiGLU (`w_gate`, `w_up`, `w_down`), so
checkpoints can be migrated by changing only config metadata.

---

## 4. IR additions

Files:

- `src/graph/graph_internal.h`
- `src/graph/graph.c`
- `src/graph/dump.c` if op attrs need printing

Add ops:

```c
_GD_OP_POWLU,
_GD_OP_POWLU_BWD,
```

Add attr:

```c
float powlu_m;
```

`_gd_op_kind_name()`:

```text
powlu
powlu_bwd
```

`_GD_OP_POWLU_BWD` is multi-output with two outputs: `dx1`, `dx2`.

---

## 5. Shape inference / op schema

Files:

- `src/ops/ops_internal.h`
- `src/ops/shape.c`
- `src/ops/op_schema.c`

Add:

```c
gd_status _gd_infer_powlu(gd_tensor *x1, gd_tensor *x2, gd_tensor_desc *out);
```

Checks:

- non-null args
- floating dtype
- same dtype/device
- same rank and sizes
- `m` finite and `0 < m < 10`

Emit:

```c
_gd_graph_emit(graph, _GD_OP_POWLU, inputs, 2, &attrs, &desc, out)
```

---

## 6. Autograd

File: `src/autograd/autograd.c`

In `backward_node`:

```text
case _GD_OP_POWLU:
  inputs = [x1, x2, go]
  out_descs = [desc(x1), desc(x2)]
  emit_multi(_GD_OP_POWLU_BWD, inputs, 3, attrs, out_descs, 2, grads)
  accumulate(input0, grads[0])
  accumulate(input1, grads[1])
```

Why multi-output: avoids two separate kernels and recomputes `f(x2)` / `f'(x2)`
only once on Metal.

---

## 7. CPU_REF implementation (naive oracle)

Files:

- `src/backends/cpu_ref/cpu_backend.h`
- `src/backends/cpu_ref/cpu_kernels.c`
- `src/backends/cpu_ref/cpu_ref.c`

Add scalar helpers:

```c
static inline float sigmoidf_stable(float x)
{
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

static inline float powlu_gate(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0f) return z * s;
    float r = sqrtf(z);
    float a = m / (r + 1.0f);
    return powf(z, a) * s;
}

static inline float powlu_gate_grad(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0f) return s * (1.0f + z * (1.0f - s));
    float r = sqrtf(z);
    float rp1 = r + 1.0f;
    float a = m / rp1;
    float g = powf(z, a);
    float da = -m / (2.0f * r * rp1 * rp1);
    return g * s * (a / z + da * logf(z) + (1.0f - s));
}
```

Kernels:

```c
gd_status _gd_cpu_k_powlu(desc, out, x1, x2, m)
gd_status _gd_cpu_k_powlu_bwd(desc, dx1, dx2, x1, x2, go, m)
```

Loops are plain scalar `for i in numel`.

Correctness priority > CPU speed.

---

## 8. Metal implementation (optimized fwd + bwd)

Files:

- `src/backends/metal/metal_kernel_types.h`
- `src/backends/metal/kernels.metal`
- `src/backends/metal/metal_backend.m`

### 8.1 Params

Add:

```c
typedef struct gd_metal_powlu_params {
    int numel;
    float m;
} gd_metal_powlu_params;
```

### 8.2 Kernels

Forward kernel:

```metal
kernel void gd_powlu(device const float *x1 [[buffer(0)]],
                     device const float *x2 [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant gd_metal_powlu_params &p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
```

Backward kernel:

```metal
kernel void gd_powlu_bwd(device const float *x1 [[buffer(0)]],
                         device const float *x2 [[buffer(1)]],
                         device const float *go [[buffer(2)]],
                         device float *dx1 [[buffer(3)]],
                         device float *dx2 [[buffer(4)]],
                         constant gd_metal_powlu_params &p [[buffer(5)]],
                         uint gid [[thread_position_in_grid]])
```

Implementation details:

- One thread per element, equal-shape contiguous tensors only.
- Inline `gd_sigmoid`, `gd_powlu_gate`, `gd_powlu_gate_grad` in MSL.
- Compute `gate = f(x2)` once in bwd; write:
  - `dx1 = go * gate`
  - `dx2 = go * x1 * gate_grad`
- Use `fast::`? No in v1. Use standard `exp`, `sqrt`, `pow`, `log` for parity.
- Consider a later fast path specialized for `m == 3.0f` only if profiling shows
  `pow/log` dominate.

### 8.3 Backend wiring

In `g_metal_kernels`:

```text
{_GD_OP_POWLU, "gd_powlu"}
{_GD_OP_POWLU_BWD, "gd_powlu_bwd"}
```

`metal_supports_node`:

- F32 only.
- Equal contiguous shapes.
- Reject broadcast.

`encode_node`:

- Add cases for `_GD_OP_POWLU` and `_GD_OP_POWLU_BWD`.
- For bwd, bind two output buffers from multi-output node.

Profiling name: `powlu`, `powlu_bwd`.

---

## 9. Fusion policy

Do **not** lower PowLU as `powlu_gate + mul` in IR. Make `gd_powlu` the canonical
op for the MLP. This gives:

- no materialized activation `f(gate)`
- one forward dispatch instead of unary+mul
- one backward dispatch instead of mul_bwd + silu_bwd
- direct access to both projections for gradient

Existing `gd_silu_mul` fusion can remain for legacy SwiGLU only. After PowLU is
default, mark `try_fuse_silu_mul` as legacy in comments.

Later optional fusion:

```text
linear(gate) + linear(up) + powlu + linear(down)
```

Not in this plan; MPS/custom GEMM boundaries make this high risk.

---

## 10. Tests

### 10.1 CPU tests

Files:

- `tests/test_ops.c`
- `tests/test_autograd.c`

Add:

- shape/dtype validation for `gd_powlu`
- finite output cases around:
  - negative large (`-20`)
  - negative small (`-1e-4`)
  - zero
  - positive small (`1e-8`, `1e-4`)
  - normal (`0.5`, `1`, `3`)
  - large positive (`20`)
- gradcheck for both inputs with `m=3.0`
- gradcheck with at least one non-default `m` (e.g. `m=2.0`)

### 10.2 Metal parity

Files:

- `tests/test_metal.c` for primitive forward/backward parity
- `tests/test_metal_gpt.c` for GPT primitive/model parity
- `tests/test_metal_gpt_train.c` for train-step parity

Add:

- forward CPU↔Metal parity for `gd_powlu`
- backward CPU↔Metal parity for `dx1`, `dx2`
- direct CPU finite-difference vs analytical gradients
- GPT forward/train parity with `GD_GPT_MLP_POWLU`
- legacy SwiGLU test remains until PowLU benches are green

Tolerance:

```text
1e-4 baseline; allow 2e-4 only if pow/log order differs near branch boundary.
```

### 10.3 Benchmarks

Files:

- `tests/gpt_bench.c`
- maybe `bench/` microbench

Add env/config switch:

```text
GD_BENCH_MLP=powlu|swiglu|gelu
GD_POWLU_M=3.0
```

Report:

- `powlu` + `powlu_bwd` GPU ms
- total MLP tail ms
- full step ms / tokens/s
- compare vs legacy `silu+mul` fusion

Expected: PowLU may be slower than SiLU because of `pow/log/sqrt`, but should
reduce activation/gradient outliers. Performance target is acceptable overhead
with stability gain; optimize only after correctness and train parity.

---

## 11. Implementation phases

### P0 — Paper + math lock

- [x] Fetch paper via `arxiv-md`.
- [x] Record formula, default `m=3`, valid range, stability motivation.
- [ ] Add this plan.

### P1 — Public API + CPU oracle

- [x] Add `_GD_OP_POWLU`, `_GD_OP_POWLU_BWD`, attr `powlu_m`.
- [x] Add `gd_powlu` public API and shape inference.
- [x] Add CPU forward/backward kernels.
- [x] Add autograd multi-output rule.
- [x] Add CPU op tests + gradcheck.

Gate:

```sh
make check
```

### P2 — Metal kernels

- [x] Add `gd_powlu` and `gd_powlu_bwd` MSL kernels.
- [x] Add Metal params / pipeline registration / encode wiring.
- [x] Add Metal primitive fwd+bwd parity tests.

Gate:

```sh
GD_METAL_MPS=1 make check
```

Also run without MPS; PowLU itself must not depend on MPS.

### P3 — GPT replacement

- [x] Add `GD_GPT_MLP_POWLU` and `powlu_m` config.
- [x] Update GPT MLP block to call `gd_powlu(up, gate, m)`.
- [x] Switch GPT tests/bench default from SwiGLU to PowLU.
- [x] Keep legacy `GD_GPT_MLP_SWIGLU` as ablation path.

Gate:

```sh
./build/tests/test_metal_gpt
./build/tests/test_metal_gpt_train
GD_METAL_MPS=1 ./build/tests/test_metal_gpt_train
```

### P4 — Profiling + docs

- [ ] Add `GD_PROFILE=trace` numbers for PowLU vs SwiGLU at B/T bench points.
- [ ] Record outlier telemetry if available: max/percentiles for MLP activation
  output and gate gradient.
- [ ] Update `docs/plan_gpt.md` and `docs/metal_gpu_fuse.md` references from
  default SwiGLU to PowLU.

---

## 12. Validation checklist

Implementation evidence (2026-05-31):
- `make test` passed.
- `GD_METAL_MPS=1 make check` passed.
- Release smoke, Metal GPT bench B=1/T=128/d=256/6L: PowLU best 78.52 ms / 1630 tok/s; legacy SwiGLU best 79.20 ms / 1616 tok/s.
- Trace smoke confirms fused kernels fire: `powlu count=6 gpu_ms=3.664`, `powlu_bwd count=6 gpu_ms=3.191`.

Must pass before marking implemented:

```sh
make check
GD_METAL_MPS=1 make check
make docs-check
```

Required evidence:

- CPU finite-difference gradcheck for both PowLU inputs.
- CPU↔Metal forward parity.
- CPU↔Metal backward parity for `dx1`, `dx2`.
- GPT forward parity with PowLU.
- GPT train parity with PowLU.
- ASan clean for PowLU tests.
- Trace confirms `powlu`/`powlu_bwd` kernels fire (not decomposed fallback).

---

## 13. Risks / open decisions

1. **Enum ABI.** Resolved: `GD_GPT_MLP_POWLU = 0` is the new default; legacy
   `GD_GPT_MLP_SWIGLU` remains available for explicit ablation.
2. **`pow/log` cost.** Metal PowLU may be slower than SiLU. Correctness first;
   optimize later with `m=3` specialization or polynomial/log2 approximations only
   if traces justify it.
3. **Near-zero derivative.** Formula has `log(z)` and `1/sqrt(z)` for `z>0`.
   Branch exactly at `z <= 0`; add targeted tests around zero.
4. **Paper is MoE-heavy.** This repo's GPT is dense MLP. Formula still applies,
   but stability gains need local measurement.
5. **SwiGLU removal.** Do not delete SwiGLU until PowLU has parity, train tests,
   and at least one bench profile. Keep ablation path.
