# sigmoid

Elementwise sigmoid op capsule.

## Public API

- `gd_sigmoid(ctx, x, out)`
- `gd_sigmoid_backward(ctx, x, grad_out, grad_x)`

## Contract

- Supports contiguous `f16` and `f32` tensors.
- Output dtype and shape match input dtype and shape.
- Direct public backward recomputes `sigmoid(x)` from `x`.
- Autograd backward uses the saved forward output from the tape:
  `grad_x = grad_out * y * (1 - y)`, avoiding an extra exp pass in the training path.

## Metal implementation

- Op-local kernels live in `metal_sigmoid.metal`.
- F16 and F32 have separate forward/direct-backward PSOs.
- Autograd has separate saved-output backward PSOs for F16 and F32.
- Kernels process four contiguous elements per thread with scalar unrolling so sliced-but-contiguous offsets do not require vector alignment.

## Validation

```sh
make check
uv run src/ops/sigmoid/fwd.py
uv run src/ops/sigmoid/bwd.py
GD_SIGMOID_PERF_PROFILE=smoke make op-perf OP=sigmoid
```

Perf controls:

```sh
GD_SIGMOID_PERF_PROFILE=all|smoke|<case-name>
GD_SIGMOID_PERF_WARMUP=10
GD_SIGMOID_PERF_ITERS=100
```

## Checklist

- [x] Public API generated in `include/gradients/ops_generated.h`
- [x] Forward validation/allocation/recording in `core_sigmoid.c`
- [x] Direct public backward in `core_sigmoid.c`
- [x] Saved-output autograd backward in `autograd_sigmoid.c`
- [x] Backend dispatch in `metal_sigmoid.m`
- [x] Op-local Metal ABI/kernel implementation in `metal_sigmoid_types.h` / `metal_sigmoid.metal`
- [x] Forward PyTorch harness in `fwd.py`
- [x] Backward PyTorch harness in `bwd.py`
- [x] C tests under `tests/`
- [x] Op-local perf probe in `perf_test.c`
