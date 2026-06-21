# MNIST MLP example

A compact MNIST classifier that follows the same v2 patterns as
`examples/mlp_xor`:

- `dataset.py` downloads raw MNIST IDX gzip files and writes GDDS shards
- images are flattened to `[784]` and stored as raw U8 to reduce dataset IO
- `main.c` attaches a dataset transform that normalizes U8 images to F16 `[0, 1]`
- labels are stored as scalar I32 class ids for `gd_cross_entropy`
- self-describing GDDS dataloader/collate path
- random no-replacement training sampler
- module tree + child `gd_linear_layer`s
- F16 `linear -> relu -> dropout -> linear` logits
- fused in-graph cross-entropy loss
- reverse-mode autograd from scalar loss
- `gd_backward_scaled` + AdamW AMP step

Run from this directory:

```sh
make run
```

Runtime options live in a flat `key: value` YAML file (`config.yaml` by default); the executable only takes a config path:

```sh
./gd_main_mlp_mnist --config config.yaml
```

The Makefile first runs:

```sh
python3 dataset.py --out-dir data
```

which writes `data/train-*.gdds`, `data/test-*.gdds`, and `data/manifest.json`.
Subsequent `make run` calls reuse matching GDDS shards instead of rebuilding them.
Raw MNIST gzip files are cached under `data/raw`.

Useful knobs are edited in YAML instead of environment variables:

```yaml
train_epochs: 3
report_every: 100
dropout_p: 0.10
lr_max: 0.001
lr_min: 0.0001
lr_warmup_steps: 100
```

Use another config file with:

```sh
make run CONFIG=my_config.yaml
make smoke  # writes data_smoke/config.yaml and uses a small GDDS subset
```

To create a smaller local GDDS dataset manually:

```sh
python3 dataset.py --out-dir data --train-limit 4096 --test-limit 1000
```

Dataset prep writes shards with dynamic byte-based flushing. Use
`--max-shard-bytes` to cap shard file size.

`main.c` opens the `data_dir` named in the YAML config. If you run dataset
prep with another `DATA_DIR`, point the runtime config at the same directory.

Then rebuild/run:

```sh
make -C ../.. build
cc -I../../include -I../common -std=c11 -O2 \
  main.c ../common/gd_example_config.c ../../build/libgradients.a \
  -pthread -framework Foundation -framework Metal -o gd_main_mlp_mnist
GRADIENTS_METALLIB=../../build/gradients.metallib ./gd_main_mlp_mnist --config config.yaml
```
