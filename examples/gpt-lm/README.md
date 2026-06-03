# GPT-LM real training example

## Build

```bash
make gpt-lm
```

No args prints usage and exits 0, so `make examples` stays safe.

## First run: prepare Promessi shards + train

```bash
GD_BENCH_DTYPE=f16 \
GD_BENCH_AMP=1 \
GD_BENCH_ATTN_WINDOW=512 \
GD_GPT_MLP_POWLU=1 \
GD_BENCH_FUSED_LMCE=1 \
GD_METAL_MPS=1 \
GD_DEVICE=metal \
make gpt-lm GPT_LM_ARGS="--prepare --steps 1000 --batch-size 8"
```

Defaults:
- corpus: `$HOME/projects/dnn.c/docs/promessi_sposi.txt`
- data dir: `data/gpt-lm-promessi`
- context len: `512`
- model dims/knobs: same env/defaults as `tests/gpt_bench.c`
  (`d_model=256`, `layers=6`, `heads=4`, `head_dim=64`, `vocab=8000`)
- dropout: `GD_BENCH_DROPOUT=0.1` enables train-only inverted dropout on
  embedding/residual branches; eval graph disables it.
- logs: `loss` is mean train loss since previous log, `loss_last` is latest
  train batch, and eval prints eval-mode `train_loss`, `val_loss`, `gap`.
  Set `GD_LM_TRAIN_EVAL=0` or pass `--no-train-eval` to skip train eval.

## Reuse prepared shards

```bash
GD_BENCH_DTYPE=f16 GD_BENCH_AMP=1 GD_BENCH_ATTN_WINDOW=512 \
GD_GPT_MLP_POWLU=1 GD_BENCH_FUSED_LMCE=1 GD_METAL_MPS=1 GD_DEVICE=metal \
make gpt-lm GPT_LM_ARGS="--steps 1000 --batch-size 8"
```

Trainer auto-loads:
- `data/gpt-lm-promessi/train-00000.gdtok`
- `data/gpt-lm-promessi/val-00000.gdtok`
