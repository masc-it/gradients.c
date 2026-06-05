# huber

Fused mean Huber loss capsule with PyTorch-compatible default `delta=1.0`.

```c
gd_status gd_huber(gd_context *ctx,
                   const gd_tensor *x,
                   const gd_tensor *y,
                   gd_tensor *out);

gd_status gd_huber_backward(gd_context *ctx,
                            const gd_tensor *x,
                            const gd_tensor *y,
                            const gd_tensor *grad_out,
                            gd_tensor *grad_x,
                            gd_tensor *grad_y);
```

Contract:

- `x` and `y` must be contiguous tensors with identical shape and dtype.
- Supported input dtypes: `GD_DTYPE_F16`, `GD_DTYPE_F32`.
- Forward output is a scalar `GD_DTYPE_F32` mean loss.
- F16 forward accumulation is FP32.
- Backward requires scalar F32 `grad_out` and writes gradients in the input dtype.

Formula:

```text
d = x - y
loss_i = 0.5 * d^2                    if |d| <= 1
       = |d| - 0.5                    otherwise
out = mean(loss_i)
grad_x_i = grad_out * clamp(d, -1, 1) / N
grad_y_i = -grad_x_i
```

Validation:

```sh
make test
uv run src/ops/huber/fwd.py
uv run src/ops/huber/bwd.py
GD_HUBER_PERF_WARMUP=10 GD_HUBER_PERF_ITERS=100 make op-perf OP=huber
```
