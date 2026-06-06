# concat

Materialized PyTorch-style tensor concatenation.

```c
gd_status gd_concat(gd_context *ctx,
                    const gd_tensor *const *inputs,
                    uint32_t n_inputs,
                    int32_t axis,
                    gd_tensor *out);

gd_status gd_concat_backward(gd_context *ctx,
                             const gd_tensor *grad_out,
                             const gd_tensor *const *inputs,
                             uint32_t n_inputs,
                             int32_t axis,
                             gd_tensor *grad_inputs);
```

Contract:

- Inputs must be contiguous, non-scalar tensors.
- All inputs must have matching dtype/rank and equal non-axis dimensions.
- Negative axes are accepted.
- Forward supports dtype element sizes 1/2/4 bytes (`u8`, `f16`/`bf16`, `f32`/`i32`).
- Autograd records only for floating tensors currently supported by gradient accumulation (`f16`, `f32`).
- Output and backward gradients are materialized contiguous tensors.

Performance probe:

```sh
GD_CONCAT_PERF_PROFILE=smoke make op-perf OP=concat
GD_CONCAT_PERF_PROFILE=all GD_CONCAT_PERF_WARMUP=10 GD_CONCAT_PERF_ITERS=100 make op-perf OP=concat
```
