# GPT language model example

Decoder LM example over the Promessi Sposi GDDS dataset, with training, validation checkpointing, standalone checkpoint inference, and KV-cache generation paths.

Model defaults:

- vocab size: 2048
- context length: 512 tokens
- `d_model = 256`
- `n_layers = 7` by default, about 7.87M trainable parameters
- `n_heads = 4`
- `head_dim = 64`
- local causal `sdpa_varlen` sliding window: 256 for training/prompt prefill
- batched decode-time KV cache with `kv_cache_append_packed`, `kv_cache_append_positions`, and `sdpa_decode_positions`
- tied LM head: token embedding weight is reused by `gd_linear_transposed_weight`
- RMSNorm, RoPE, PoWLU gated MLP, and dropout

Build and run training:

```sh
make -C examples/gpt_lm run ARGS="--epochs 2 --batch-size 32"
```

At the end of every epoch, training evaluates the `val` split and saves `checkpoints/gpt_lm_best.gdckpt` whenever validation loss improves.

Overfit smoke tests:

```sh
make -C examples/gpt_lm run ARGS="--epochs 2 --batch-size 1 --overfit-num-samples 1 --report-every 1 --no-save-best"
make -C examples/gpt_lm run ARGS="--epochs 1 --batch-size 32 --overfit-num-samples 32 --report-every 1"
```

Generation-only smoke test with random/current in-memory weights:

```sh
make -C examples/gpt_lm run ARGS="--epochs 0 --generate 'Don Abbondio' --max-new-tokens 64"
```

Checkpoint inference after training:

```sh
make -C examples/gpt_lm infer ARGS="--checkpoint checkpoints/gpt_lm_best.gdckpt --prompt 'Don Abbondio' --max-new-tokens 64"
```

Periodic batched generation during training:

```sh
make -C examples/gpt_lm run ARGS="--epochs 1 --batch-size 1 --overfit-num-samples 1 --generate-every-n-steps 1 --max-new-tokens 4"
```

The periodic path runs one batched generation over the five vowel prefixes `a/e/i/o/u`.

Useful runtime options:

```text
--epochs N              # 0 allowed with --generate
--batch-size N
--layers N
--dropout P
--overfit-num-samples N
--report-every N
--lr-max LR
--lr-min LR
--warmup-steps N        # default -1 means total_steps / 10
--generate TEXT
--generate-every-n-steps N
--checkpoint-path PATH
--load-checkpoint PATH
--val-split NAME
--no-save-best
--max-new-tokens N
--temperature T         # 0 means greedy
--tokenizer-path PATH
```

The dataset builder writes `train` and `val` GDDS splits. The GDDS dataloader provides packed fields directly:

```text
input_ids   i32 [B * 512]
target_ids  i32 [B * 512]
cu_seqlens  i32 [B + 1]
positions   i32 [B * 512]
```

Generation uses packed `sdpa_varlen` for batched prompt prefill, `kv_cache_append_packed` to materialize variable-length prompt K/V, then appends one token per batch row with `kv_cache_append_positions` and calls `sdpa_decode_positions` for each autoregressive step. Sampling is currently CPU-side, so very small generations are latency-bound by per-step logits readback.
