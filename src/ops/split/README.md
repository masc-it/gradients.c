# split

Materialized PyTorch-style tensor split for contiguous tensors.

```c
gd_split(ctx, x, sizes, n_outputs, axis, outputs)
gd_split_backward(ctx, x, grad_outputs, sizes, n_outputs, axis, grad_x)
```

`sizes[i]` is the length of output `i` along `axis`. Negative axes are
accepted. All sizes must be positive and sum exactly to `x.shape[axis]`.

## Semantics

- Forward materializes `n_outputs` new contiguous scratch tensors.
- `outputs[i].shape == x.shape` except `outputs[i].shape[axis] = sizes[i]`.
- Backward materializes `grad_x` by placing each contiguous `grad_outputs[i]`
  slice back into the full input-gradient tensor.
- Input and gradients must be contiguous; outputs are contiguous.
- Forward supports 1-, 2-, and 4-byte runtime dtypes (`u8`, `f16`, `bf16`,
  `f32`, `i32`).
- Autograd supports `f16` and `f32`, matching gradient accumulation support.

This is intentionally materialized for the first transformer use case: split
packed QKV activations into contiguous Q/K/V tensors for downstream attention
kernels that currently require contiguous descriptors.

## Metal

The Metal capsule uses element-size-specialized copy kernels:

- `gd_split_from_full_{u8,u16,u32}_kernel`
- `gd_split_to_full_{u8,u16,u32}_kernel`

For aligned slices whose contiguous slice block is a multiple of 16 bytes, the
host dispatches vectorized kernels:

- `gd_split_from_full_vec16_kernel`
- `gd_split_to_full_vec16_kernel`

The vectorized path targets common QKV layouts such as
`[B,T,3,H,Dh] -> {[B,T,1,H,Dh] x 3}` and `[B,T,3*D] -> {[B,T,D] x 3}`.
