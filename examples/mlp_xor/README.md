# XOR MLP example

A tiny full-batch XOR training run that exercises the current v2 API:

- module tree + child `gd_linear_layer`s
- parameter collection and param groups
- F16 `linear -> relu -> linear`
- in-graph MSE loss with `sub -> mul -> reduce_mean`
- reverse-mode autograd from scalar loss
- `gd_backward_scaled` + AdamW AMP step
- FP32 AdamW master weights for F16 params

Run from this directory:

```sh
make run
```

or manually from the repo root:

```sh
make build
cc -Iinclude -std=c11 -O2 examples/mlp_xor/main.c build/libgradients.a \
  -framework Foundation -framework Metal -o examples/mlp_xor/mlp_xor
GRADIENTS_METALLIB=build/gradients.metallib examples/mlp_xor/mlp_xor
```

This is now a fully in-graph loss/backward path. Host reads are used only for
logging and final validation.
