# permute

Materialized axis permutation for contiguous tensors.

```c
gd_permute(ctx, x, axes, n_axes, out)
gd_permute_backward(ctx, x, grad_out, axes, n_axes, grad_x)
```

`axes[d]` names the input axis copied into output axis `d`, matching
`torch.permute`. Negative axes are accepted and normalized. The axis list length
must equal `x->rank`, and every axis must appear exactly once.

## Semantics

- Forward materializes a new contiguous scratch tensor with
  `out.shape[d] = x.shape[axes[d]]`.
- Backward materializes `grad_x = grad_out.permute(inverse(axes))` with `x`'s
  shape.
- Inputs and gradients must be contiguous; outputs are contiguous.
- Forward supports 1-, 2-, and 4-byte runtime dtypes (`u8`, `f16`, `bf16`,
  `f32`, `i32`).
- Autograd supports `f16` and `f32`, matching gradient accumulation support.

## Metal

The Metal capsule uses element-size-specialized generic kernels plus optimized
shape classes:

- `gd_permute_{u8,u16,u32}_kernel` generic element copy.
- `gd_permute_block_{u8,u16,u32}_kernel` for contiguous suffix blocks.
- `gd_permute_suffix16_kernel` for 16-byte vectorized contiguous suffix copies.
- `gd_permute_transpose_{u8,u16,u32}_kernel` for rank-2 transpose and batched
  rank-3 `[B,M,N] -> [B,N,M]` using a padded threadgroup tile.
- `gd_permute_hwc_to_chw_{u8,u16,u32}_kernel` and
  `gd_permute_chw_to_hwc_{u8,u16,u32}_kernel` for image/channel layouts with
  small channel counts.

The host precomputes the largest identity contiguous suffix (`inner`) so common
transformer layouts such as `[B,T,H,Dh] -> [B,H,T,Dh]` copy the `Dh` suffix with
no per-element remap for that suffix, and dispatches specialized kernels for
common transpose/channel-layout patterns.
