# rope

Rotary positional embedding for contiguous tensors shaped `[.., heads, head_dim]`.

- `pos_ids`: contiguous I32 tensor with element count equal to the product of the leading dimensions before `heads`.
- `theta <= 0`: defaults to `10000`.
- `n_dims <= 0`: rotates the full `head_dim`; otherwise must be even and `<= head_dim`.
- `interleaved=false`: NeoX half-split pairs `(i, i + n_dims/2)`.
- `interleaved=true`: GPT-J even/odd pairs `(2i, 2i + 1)`.

Metal kernels are dtype-specialized (`f16`/`f32`) and launch one thread per rotary pair plus fused tail copy lanes.
