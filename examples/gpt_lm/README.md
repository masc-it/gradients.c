# GPT language model example

Decoder-only LM example for the current Italian dictionary (`ita_dict`) GDDS dataset, with training, validation checkpointing, full-resume checkpoints, standalone checkpoint inference, interactive generation, and KV-cache decode.

The Makefile still supports the older Promessi Sposi dataset via `GPT_LM_DATASET=promessi`, but the current dataset path documented here is `ita_dict`.

## Model defaults

- vocab size: 2048
- context length: 512 tokens
- `d_model = 512`
- `n_layers = 3` by default, about 8.93M trainable parameters
  - set `layers: 2` in the YAML config for about 6.30M trainable parameters
- `n_heads = 8`
- `head_dim = 64`
- MLP hidden size: 1024 (`2 * d_model`)
- default architecture: dense GPT attention (`sdpa_varlen`) with a 256-token sliding window for training/prompt prefill
- optional `architecture: minimax_m3`: MiniMax M3-style attention; the example now auto-falls back to the fast dense 256-token window at the default 512-token context, and uses the Metal lightning indexer (`gd_minimax_m3_index_topk`) plus sparse attention (`gd_minimax_m3_sparse_attention`) only when the context is long enough for sparsity to win
- batched decode-time KV cache with `kv_cache_append_packed`, `kv_cache_append_positions`, and `sdpa_decode_positions`
- biased dense projections throughout the transformer block, plus a tied LM head that reuses `token_embedding` with a separate `lm_head_bias` for fused LM cross-entropy and generation logits
- RMSNorm, RoPE, SwiGLU gated MLP, dropout, AMP AdamW, gradient clipping, and optional logits softcap

## Dataset: `ita_dict`

Select the current dataset explicitly when invoking the example:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict data
```

`dataset_ita_dict.py` expects two directories of Italian dictionary JSON files:

- clean definitions: `GPT_LM_ITA_CLEAN_DIR`
- enriched definitions/examples: `GPT_LM_ITA_ENRICHED_DIR`

If the script defaults do not exist on your machine, pass them explicitly:

```sh
make -C examples/gpt_lm \
  GPT_LM_DATASET=ita_dict \
  GPT_LM_ITA_CLEAN_DIR=/path/to/definizioni-clean \
  GPT_LM_ITA_ENRICHED_DIR=/path/to/definizioni-clean-enriched \
  data
```

The builder merges clean and enriched entries, prefers enriched entries when present, splits by dictionary term before tokenizer training, writes `raw_ita_dict/{train,val}.{txt,jsonl}`, trains the BPE tokenizer on `train.jsonl` only, wraps each dictionary entry with `<|im_start|>` / `<|im_end|>`, then packs the token stream into fixed 512-token LM rows.

The GDDS dataloader exposes these fields:

```text
input_ids       i32 [B * 512]
positions       i32 [B * 512]          # reset at each dictionary-entry fragment
target_ids      i32 [B * 512]          # next token; -1 masks cross-entry targets
segment_lengths i32 [num_segments]     # per-row entry-fragment lengths
cu_seqlens      i32 [num_segments + 1] # generated from segment_lengths
```

The model consumes `input_ids`, `target_ids`, `positions`, and `cu_seqlens`.

`GPT_LM_DATASET=ita_dict_v2` now writes compact GDDS shards: on disk each sample stores `tokens` as `u16 [context_length + 1]` plus small `segment_lengths` metadata. The training dataloader expands those fields into the runtime tensors above and replaces cross-document targets with the tokenizer `<|pad|>` id, which the fused LM CE ignores.

Useful data-prep variables:

```text
GPT_LM_VOCAB_SIZE=2048
GPT_LM_CONTEXT_LENGTH=512
GPT_LM_VAL_FRACTION=0.05
GPT_LM_SPLIT_SEED=17
GPT_LM_MIN_FREQUENCY=2
GPT_LM_MAX_EXAMPLES_PER_DEFINITION=-1
GPT_LM_MAX_SHARD_BYTES=2147483648
DATA_DIR=data
```

## Train

Training runtime options live in a flat YAML file (`config.yaml` by default). Edit values such as `epochs`, `batch_size`, `layers`, and learning-rate settings there, then run:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run
```

Use another config with:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run CONFIG=my_config.yaml
```

Select the sparse MiniMax M3 attention architecture in YAML:

```yaml
architecture: minimax_m3
epochs: 2
batch_size: 64
```

Sparse attention knobs are `minimax_m3_topk_blocks`, `minimax_m3_init_blocks`, and `minimax_m3_local_blocks` (block size is fixed at 128). At the default 512-token context, `minimax_m3` uses the dense-window fallback because sparse M3 would attend most blocks and pay indexer overhead; the true sparse kernels are selected automatically for longer contexts. The implementation uses the regular GPT Q/K projections as lightning-index Q/K and supports the example head size (`head_dim=64`).

By default, training evaluates the `val` split at the end of each epoch, writes `checkpoints/gpt_lm_best.gdckpt` when validation improves, writes `checkpoints/gpt_lm_latest.gdckpt` for full resume, emits async JSONL metrics under `data/metrics/gpt_lm/`, and stops after 10 validation epochs without improvement. Optimizer/scaler/trainer state lives in sidecars next to each model checkpoint: `*.optim.gdckpt` and `*.train`.

If both `save_best: false` and `early_stopping_patience: 0` are set, validation is skipped. `save_latest: false` disables latest/full-resume checkpoint writes. Metrics can be disabled with `metrics_enabled: false`, or routed with `metrics_dir`, `metrics_project`, and `metrics_run_id`.

Run the live dashboard from the repository root with:

```sh
uv run tools/gd_dash/main.py --metrics-dir data/metrics
```

## Smoke tests

Tiny overfit smoke test:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict overfit-smoke
```

The smoke configs live under `examples/gpt_lm/configs/`; copy one and edit `architecture: minimax_m3` for a MiniMax M3 smoke test (dense fallback at the default 512-token context).

Generation-only smoke test with random/current in-memory weights:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict generate-smoke
```

Periodic batched generation during training:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict periodic-generate-smoke
```

The periodic path currently runs one batched generation over the five debug prefixes `a/e/i/o/u`.

## Checkpoint inference

After training:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict infer ARGS="--checkpoint checkpoints/gpt_lm_best.gdckpt --prompt '<|im_start|>Termine: casa Definizioni:' --max-new-tokens 64"
```

Interactive checkpoint generation:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict interactive ARGS="--checkpoint checkpoints/gpt_lm_best.gdckpt --temperature 0.8 --min-p 0.05 --repetition-penalty 1.1"
```

Interactive mode sends prompts verbatim (include `<|im_start|>` yourself if desired) and stops early when `<|im_end|>` is emitted. Multi-line pastes that arrive together are grouped into one prompt; for explicit paste mode, type `:paste`, paste the text, then finish with a line containing only `:end`.

## Resume training

Set these in a YAML config:

```yaml
resume_checkpoint_path: checkpoints/gpt_lm_latest.gdckpt
epochs: 500
batch_size: 64
```

then run:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run CONFIG=my_resume.yaml
```

Resume loads model weights plus optimizer/scaler/trainer sidecars. The current YAML LR schedule is used for the resumed run.

## Useful runtime config keys

The training binary accepts only `--config PATH`. The YAML file must contain all runtime keys; optional paths/prompts use an empty string when unset.

```yaml
data_dir: data
tokenizer_path: ""
generate_prompt: ""        # epochs: 0 requires this to be non-empty
checkpoint_path: checkpoints/gpt_lm_best.gdckpt
latest_checkpoint_path: checkpoints/gpt_lm_latest.gdckpt
load_checkpoint_path: ""    # model weights only
resume_checkpoint_path: ""  # model + optimizer/scaler/trainer sidecars
val_split: val
metrics_dir: ../../data/metrics
metrics_project: gpt_lm
metrics_run_id: ""
local_shard_cache_dir: ""

epochs: 2
batch_size: 64
dataloader_workers: 2
dataloader_prefetch_factor: 4 # workers * prefetch <= 63; one data slot is reserved
layers: 3
architecture: gpt             # gpt or minimax_m3
shuffle_scope: global          # global, shard, or none
minimax_m3_topk_blocks: 3
minimax_m3_init_blocks: 1
minimax_m3_local_blocks: 1
report_every: 10
eval_every_n_epochs: 1
early_stopping_patience: 10    # 0 disables
warmup_steps: -1               # -1 means total_steps / 10
latest_every_n_steps: 0
generate_every_n_steps: 0
save_best: true
save_latest: true
metrics_enabled: true
keep_shard_cache: false
overfit_num_samples: 0
seed: 0x6750746c6d5eed00

dropout_p: 0.05
lr_max: 0.0003
lr_min: 0.00005
weight_decay: 0.0001
grad_clip_norm: 1.0            # 0 disables

max_new_tokens: 64
temperature: 0.0               # 0 means greedy
min_p: 0.0                     # filter tokens below P * top probability; 0 disables
repetition_penalty: 1.0        # 1 disables
logits_softcap: 30.0           # final logits softcap; 0 disables
```

Generation uses the selected architecture for batched prompt prefill (`sdpa_varlen` for `gpt`, MiniMax M3 sparse attention for long enough `minimax_m3` contexts, otherwise the dense-window fallback), `kv_cache_append_packed` to materialize variable-length prompt K/V, then appends one token per batch row with `kv_cache_append_positions` and currently calls dense `sdpa_decode_positions` for each autoregressive step. Sampling is CPU-side, so very small generations are latency-bound by per-step logits readback.


---

## fineweb2 sketch

Copy `examples/gpt_lm/config.yaml` to a separate YAML file and edit the dataset-specific keys, for example:

```yaml
data_dir: "/Volumes/Seagate 2TB/datasets/gd_fineweb2/gdds"
tokenizer_path: "/Volumes/Seagate 2TB/datasets/gd_fineweb2/tokenizer/tokenizer-v8192.json"
epochs: 1
batch_size: 8
report_every: 4
generate_every_n_steps: 1000
early_stopping_patience: 0
weight_decay: 0.00001
lr_min: 0.00003
lr_max: 0.0003
eval_every_n_epochs: 1
logits_softcap: 0
architecture: gpt
latest_checkpoint_path: "/Volumes/Seagate 2TB/datasets/gd_fineweb2/checkpoints/gpt_lm_latest.gdckpt"
latest_every_n_steps: 1000
shuffle_scope: shard
```

Then build with the matching compile-time shape and run the config:

```sh
make -C examples/gpt_lm \
  GPT_LM_VOCAB_SIZE=8192 \
  GPT_LM_CONTEXT_LENGTH=2048 \
  all
GRADIENTS_METALLIB="$(pwd)/build-gpt_lm/gradients.metallib" \
  examples/gpt_lm/gd_main_gpt_lm --config /path/to/fineweb2.yaml
```