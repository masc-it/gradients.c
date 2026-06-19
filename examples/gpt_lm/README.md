# GPT language model example

Decoder-only LM example for the current Italian dictionary (`ita_dict`) GDDS dataset, with training, validation checkpointing, full-resume checkpoints, standalone checkpoint inference, interactive generation, and KV-cache decode.

The Makefile still supports the older Promessi Sposi dataset via `GPT_LM_DATASET=promessi`, but the current dataset path documented here is `ita_dict`.

## Model defaults

- vocab size: 2048
- context length: 512 tokens
- `d_model = 512`
- `n_layers = 3` by default, about 8.92M trainable parameters
  - use `--layers 2` for about 6.29M trainable parameters
- `n_heads = 8`
- `head_dim = 64`
- MLP hidden size: 1024 (`2 * d_model`)
- default architecture: dense GPT attention (`sdpa_varlen`) with a 256-token sliding window for training/prompt prefill
- optional `--architecture minimax_m3`: MiniMax M3-style attention; the example now auto-falls back to the fast dense 256-token window at the default 512-token context, and uses the Metal lightning indexer (`gd_minimax_m3_index_topk`) plus sparse attention (`gd_minimax_m3_sparse_attention`) only when the context is long enough for sparsity to win
- batched decode-time KV cache with `kv_cache_append_packed`, `kv_cache_append_positions`, and `sdpa_decode_positions`
- tied LM head: token embedding weight is reused by `gd_linear_transposed_weight`
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

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--epochs 2 --batch-size 64"
```

Select the sparse MiniMax M3 attention architecture with:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--architecture minimax_m3 --epochs 2 --batch-size 64"
```

Sparse attention knobs are `--minimax-topk-blocks`, `--minimax-init-blocks`, and `--minimax-local-blocks` (block size is fixed at 128). At the default 512-token context, `minimax_m3` uses the dense-window fallback because sparse M3 would attend most blocks and pay indexer overhead; the true sparse kernels are selected automatically for longer contexts. The implementation uses the regular GPT Q/K projections as lightning-index Q/K and supports the example head size (`head_dim=64`).

By default, training evaluates the `val` split at the end of each epoch, writes `checkpoints/gpt_lm_best.gdckpt` when validation improves, writes `checkpoints/gpt_lm_latest.gdckpt` for full resume, emits async JSONL metrics under `data/metrics/gpt_lm/`, and stops after 10 validation epochs without improvement. Optimizer/scaler/trainer state lives in sidecars next to each model checkpoint: `*.optim.gdckpt` and `*.train`.

If both `--no-save-best` and `--early-stopping-patience 0` are set, validation is skipped. `--no-save-latest` disables latest/full-resume checkpoint writes. Metrics can be disabled with `--no-metrics`, or routed with `--metrics-dir`, `--metrics-project`, and `--metrics-run-id`.

Run the live dashboard from the repository root with:

```sh
uv run tools/gd_dash/main.py --metrics-dir data/metrics
```

## Smoke tests

Tiny overfit smoke test:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--epochs 1 --batch-size 1 --layers 1 --overfit-num-samples 1 --report-every 1 --no-save-best --no-save-latest --early-stopping-patience 0"
```

MiniMax M3 architecture smoke test (dense fallback at the default 512-token context):

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--architecture minimax_m3 --epochs 1 --batch-size 1 --layers 1 --overfit-num-samples 1 --report-every 1 --no-save-best --no-save-latest --early-stopping-patience 0"
```

Generation-only smoke test with random/current in-memory weights:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--epochs 0 --generate '<|im_start|>Termine: casa Definizioni:' --max-new-tokens 32"
```

Periodic batched generation during training:

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--epochs 1 --batch-size 1 --layers 1 --overfit-num-samples 1 --report-every 1 --generate-every-n-steps 1 --max-new-tokens 4 --no-save-best --no-save-latest --early-stopping-patience 0"
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

Interactive mode prefixes each line you type with `<|im_start|>` and stops early when `<|im_end|>` is emitted.

## Resume training

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--resume-checkpoint checkpoints/gpt_lm_latest.gdckpt --epochs 500 --batch-size 64"
```

Resume loads model weights plus optimizer/scaler/trainer sidecars. The current CLI LR schedule is used for the resumed run.

## Useful runtime options

```text
--data-dir PATH
--tokenizer-path PATH
--epochs N                  # 0 allowed with --generate
--batch-size N
--layers N
--architecture NAME          # gpt or minimax_m3
--minimax-topk-blocks N
--minimax-init-blocks N
--minimax-local-blocks N
--dropout P
--overfit-num-samples N
--report-every N
--lr-max LR
--lr-min LR
--warmup-steps N            # default -1 means total_steps / 10
--weight-decay WD
--grad-clip-norm N          # 0 disables
--generate TEXT             # skips training unless --epochs is set
--generate-every-n-steps N
--checkpoint-path PATH        # best-val model checkpoint
--latest-checkpoint-path PATH # full-resume checkpoint saved every epoch
--load-checkpoint PATH        # model weights only
--resume-checkpoint PATH      # model + optimizer/scaler/trainer sidecars
--val-split NAME
--metrics-dir PATH          # default data/metrics
--metrics-project NAME      # default gpt_lm
--metrics-run-id ID         # default timestamp-pid
--no-metrics
--no-save-best
--no-save-latest
--early-stopping-patience N # 0 disables; default 10
--max-new-tokens N
--temperature T             # 0 means greedy
--min-p P                   # filter tokens below P * top probability; 0 disables
--repetition-penalty P      # 1 disables
--logits-softcap C          # final logits softcap; 0 disables
--seed N
```

Generation uses the selected architecture for batched prompt prefill (`sdpa_varlen` for `gpt`, MiniMax M3 sparse attention for long enough `minimax_m3` contexts, otherwise the dense-window fallback), `kv_cache_append_packed` to materialize variable-length prompt K/V, then appends one token per batch row with `kv_cache_append_positions` and currently calls dense `sdpa_decode_positions` for each autoregressive step. Sampling is CPU-side, so very small generations are latency-bound by per-step logits readback.
