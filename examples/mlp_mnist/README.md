# MNIST MLP example

A compact MNIST classifier that follows the same v2 patterns as
`examples/mlp_xor`:

- `dataset.py` downloads raw MNIST IDX gzip files and writes GDDS shards
- images are normalized to `[0, 1]`, flattened to `[784]`, and stored as F16
- labels are stored as scalar I32 class ids for `gd_cross_entropy`
- generic GDDS dataloader/collate path
- random no-replacement training sampler
- module tree + child `gd_linear_layer`s
- F16 `linear -> relu -> linear` logits
- fused in-graph cross-entropy loss
- reverse-mode autograd from scalar loss
- `gd_backward_scaled` + AdamW AMP step

Run from this directory:

```sh
make run
```

The Makefile first runs:

```sh
python3 dataset.py --out-dir data
```

which writes `data/train-*.gdds`, `data/test-*.gdds`, and `data/manifest.json`.
Raw MNIST gzip files are cached under `data/raw`.

Useful knobs:

```sh
GD_MNIST_TRAIN_STEPS=1500 GD_MNIST_REPORT_EVERY=100 make run
GD_MNIST_MIN_ACCURACY=0.80 make run
make smoke  # uses DATA_DIR=data_smoke and a small GDDS subset
```

To create a smaller local GDDS dataset manually:

```sh
python3 dataset.py --out-dir data --train-limit 4096 --test-limit 1000
```

`main.c` opens `data` by default; set `GD_MNIST_DATA_DIR=/path/to/gdds`
or run `make DATA_DIR=/path/to/gdds run` to use another directory.

Then rebuild/run:

```sh
make -C ../.. build
cc -I../../include -std=c11 -O2 main.c ../../build/libgradients.a \
  -pthread -framework Foundation -framework Metal -o mlp_mnist
GRADIENTS_METALLIB=../../build/gradients.metallib ./mlp_mnist
```
