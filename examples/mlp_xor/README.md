# XOR MLP example

A tiny full-batch XOR training run that exercises the current v2 API:

- GDDS disk-backed dataset generation via `dataset.py`
- generic GDDS dataloader/collate path
- module tree + child `gd_linear_layer`s
- parameter collection and param groups
- F16 `linear -> relu -> linear`
- fused in-graph Huber loss with `gd_huber`
- reverse-mode autograd from scalar loss
- `gd_backward_scaled` + AdamW AMP step
- FP32 AdamW master weights for F16 params

Run from this directory:

```sh
make run
```

The Makefile first runs `dataset.py`, which writes `data/xor-00000.gdds` with
shared utilities from `tools/gdds_utils.py`.

or manually from this directory:

```sh
python3 dataset.py --out-dir data --split xor
make -C ../.. build
cc -I../../include -std=c11 -O2 main.c ../../build/libgradients.a \
  -pthread -framework Foundation -framework Metal -o mlp_xor
GRADIENTS_METALLIB=../../build/gradients.metallib ./mlp_xor
```

This is now a fully in-graph loss/backward path. Host reads are used only for
logging and final validation.
