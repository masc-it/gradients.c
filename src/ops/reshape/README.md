# reshape

`gd_reshape` is a metadata-only, PyTorch-style reshape view.

## Contract

- Input must be a valid contiguous tensor.
- Target rank must be `<= GD_MAX_DIMS`.
- Target dimensions must be positive, except one dimension may be `-1` and is inferred.
- Element count must be preserved.
- Zero-size dimensions are rejected because the current tensor runtime does not represent zero-element tensors.
- Output aliases input storage, keeps the same `view_offset`, uses contiguous strides for the target shape, and is marked `is_view=true`.
- Forward supports all runtime dtypes.
- Autograd is enabled for `f16` and `f32`, matching gradient accumulation support.

## Backward

`gd_reshape_backward(ctx, x, grad_out, grad_x)` returns a metadata-only view of `grad_out` with `x`'s shape. The autograd rule then accumulates that contribution into `x`.

## Metal/performance note

There is no Metal kernel or PSO for reshape. The optimized path avoids materialization entirely: forward and direct backward perform descriptor validation plus tensor-id assignment only. Autograd performance is dominated by the runtime's gradient-slot zeroing and accumulation kernels, not by reshape itself.

Run:

```sh
make op-perf OP=reshape
GD_RESHAPE_PERF_PROFILE=all make op-perf OP=reshape
```
