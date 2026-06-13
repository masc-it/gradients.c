# tanh

Elementwise hyperbolic tangent op capsule.

## Public API

- `gd_tanh(ctx, x, out)`
- `gd_tanh_backward(ctx, x, grad_out, grad_x)`

## Contracts

- Supports contiguous `f16` and `f32` tensors.
- Output has the same shape and dtype as input.
- Forward computes `out = tanh(x)` in FP32 for F16 inputs, then stores back to F16.
- Direct backward computes `grad_x = grad_out * (1 - tanh(x)^2)`.
- Autograd uses the saved forward output: `grad_x = grad_out * (1 - out^2)`, avoiding a second tanh dispatch/recompute.

## Files

- `core_tanh.c` — validation/allocation, public forward/backward, saved-output backward helper.
- `autograd_tanh.c` — saved-output autograd rule.
- `metal_tanh.m` — Metal host validation/dispatch, dtype-specialized PSO selection.
- `metal_tanh_types.h` / `metal_tanh.metal` — op-local ABI and kernels.
- `fwd.py` / `bwd.py` — PyTorch parity harnesses.
- `perf_test.c` — op-local performance probe (`make op-perf OP=tanh`).

## Validation

```sh
uv run src/ops/tanh/fwd.py
uv run src/ops/tanh/bwd.py
make op-perf OP=tanh
```
