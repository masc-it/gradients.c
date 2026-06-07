# rms_norm

Root-mean-square normalization over the last dimension:

```c
gd_rms_norm(ctx, x, weight, eps, out)
gd_rms_norm_backward(ctx, x, weight, grad_out, eps, grad_x, grad_weight)
```

For each logical row and channel `c`:

```text
inv_rms[row] = 1 / sqrt(mean_c(x[row,c]^2) + eps)
out[row,c] = x[row,c] * weight[c] * inv_rms[row]
```

## Contract

- `x`: contiguous `f16` or `f32`, rank `>= 1`.
- `weight`: contiguous 1D tensor `[x.shape[-1]]`, same dtype as `x`.
- `eps > 0`.
- `out`: materialized contiguous tensor with `x` shape/dtype.
- Backward `grad_out` must match `x` shape/dtype and be contiguous.
- `grad_x` and `grad_weight` are optional; emitted gradients use the corresponding input dtype.

## Implementation notes

- Forward accumulates row sum-of-squares in FP32.
- Training forward saves compact `f32 [rows]` `inv_rms` stats when either input requires grad.
- Autograd backward consumes saved `inv_rms` to avoid recomputing RMS statistics.
- Direct public backward computes `inv_rms` only when needed for `grad_weight`; `grad_x`-only direct backward uses a fused recompute path.
- Weight gradient uses a coalesced two-stage reduction:
  1. row-block/channel-block partials in `f32 [row_blocks, last_dim]`,
  2. column-wise partial reduction to the requested output dtype.
- Weight-gradient row blocks are adaptive: 64 rows for small batches, 128 rows for large row counts to cut partial traffic and row-block launch work.

## Metal kernels

- `gd_rms_norm_forward_{f16,f32}_kernel`
- `gd_rms_norm_forward_stats_{f16,f32}_kernel`
- `gd_rms_norm_inv_{f16,f32}_kernel`
- `gd_rms_norm_backward_{f16,f32}_kernel`
- `gd_rms_norm_backward_stats_{f16,f32}_kernel`
- `gd_rms_norm_wgrad_stage_stats_{f16,f32}_kernel`
- `gd_rms_norm_wgrad_reduce_{f16,f32}_kernel`

Reduction threadgroup width is selected by last dimension; F16 math-sensitive paths accumulate in FP32.
