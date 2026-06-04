# XOR MLP example

A tiny full-batch XOR training run that exercises the current v2 API:

- module tree + child `gd_linear_layer`s
- parameter collection and param groups
- F16 `linear -> relu -> linear`
- reverse-mode autograd
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

## Current v2 limitation shown here

The runtime does not yet have differentiable loss/reduction ops such as MSE,
mean/sum, sigmoid, or binary cross entropy. This example therefore reads the
batch predictions, computes `d MSE / d output` on the CPU, uploads that tensor,
and uses it as the backward seed. That is enough for a first end-to-end MLP run
and makes the missing op surface explicit.
