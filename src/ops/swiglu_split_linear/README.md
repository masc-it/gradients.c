# swiglu_split_linear

Fused SwiGLU gated-MLP projection:

```c
y = linear(swiglu_split(x12), w, bias)
```

`x12` has shape `[..., 2H]`, `w` has shape `[H, N]`, and optional `bias` has shape `[N]`.
The op records one autograd node, saves the compact activation by default, and uses the optimized linear/GEMM backend for projection gradients.
