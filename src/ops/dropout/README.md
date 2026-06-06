# dropout

Inverted dropout op capsule.

## Public API

```c
gd_status gd_dropout(gd_context *ctx,
                     const gd_tensor *x,
                     float p,
                     bool training,
                     uint64_t seed,
                     gd_tensor *out);

gd_status gd_dropout_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              float p,
                              uint64_t seed,
                              gd_tensor *grad_x);
```

## Contract

- Supports contiguous `f16` and `f32` tensors.
- `p` must satisfy `0 <= p < 1`.
- Training forward uses inverted dropout:
  `out[i] = mask[i] ? x[i] / (1 - p) : 0`.
- The mask is deterministic for `(seed, linear_index)` and is independent of shape rank.
- `training=false` and `p=0` are identity paths that return an alias of `x`.
- Direct public backward recomputes the stateless mask from `(p, seed)`.
- Autograd backward uses the compact `u8` mask saved by forward, avoiding a second RNG pass.

## Metal implementation

- Op-local kernels live in `metal_dropout.metal`.
- Separate F16/F32 PSOs for:
  - forward with compact mask write,
  - direct backward with mask recompute,
  - saved-mask autograd backward.
- Kernels process four contiguous elements per thread with scalar unrolling.

## Validation

```sh
make check
uv run src/ops/dropout/fwd.py
uv run src/ops/dropout/bwd.py
GD_DROPOUT_PERF_PROFILE=smoke make op-perf OP=dropout
```

Perf controls:

```sh
GD_DROPOUT_PERF_PROFILE=all|smoke|<case-name>
GD_DROPOUT_PERF_WARMUP=10
GD_DROPOUT_PERF_ITERS=100
```

## Checklist

- [x] Public API generated in `include/gradients/ops_generated.h`
- [x] Forward validation/allocation/recording in `core_dropout.c`
- [x] Direct public backward in `core_dropout.c`
- [x] Saved-mask autograd backward in `autograd_dropout.c`
- [x] Backend dispatch in `metal_dropout.m`
- [x] Op-local Metal ABI/kernel implementation in `metal_dropout_types.h` / `metal_dropout.metal`
- [x] Forward PyTorch formula harness in `fwd.py`
- [x] Backward PyTorch formula harness in `bwd.py`
- [x] C tests under `tests/`
- [x] Op-local perf probe in `perf_test.c`
