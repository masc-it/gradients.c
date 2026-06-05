# mse

Fused mean-squared-error loss capsule.

```c
gd_status gd_mse(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out);

gd_status gd_mse_backward(gd_context *ctx,
                          const gd_tensor *x,
                          const gd_tensor *y,
                          const gd_tensor *grad_out,
                          gd_tensor *grad_x,
                          gd_tensor *grad_y);
```

Contract:

- `x` and `y` must be contiguous tensors with identical shape and dtype.
- Supported input dtypes: `GD_DTYPE_F16`, `GD_DTYPE_F32`.
- Forward output is a scalar `GD_DTYPE_F32` loss.
- F16 forward accumulation is FP32.
- Backward requires scalar F32 `grad_out` and writes gradients in the input dtype.

Metal implementation:

- Forward computes per-chunk FP32 squared-error sums, then reduces partials to
  the scalar mean for large tensors.
- Backward is fused elementwise: `±grad_out * 2/N * (x - y)`.
- Kernels are dtype-specialized (`f16`, `f32`) and live entirely in this op
  capsule.

Validation:

```sh
make test
uv run src/ops/mse/fwd.py
uv run src/ops/mse/bwd.py
GD_MSE_PERF_WARMUP=10 GD_MSE_PERF_ITERS=100 make op-perf OP=mse
```
