# swiglu

SwiGLU gated activation for F16 tensors.

- `gd_swiglu(ctx, x1, x2, out)` computes `x1 * (x2 * sigmoid(x2))`.
- `gd_swiglu_split(ctx, x12, out)` treats the last dimension as `[x1, x2]` and computes the fused split activation.
- Backward kernels compute both split gradients in one pass.

The GPT LM example uses `gd_swiglu_split_linear`, which composes this fused activation with the optimized linear/GEMM backend.
