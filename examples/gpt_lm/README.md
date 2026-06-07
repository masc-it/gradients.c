# GPT language model example

Training-only decoder LM example over the Promessi Sposi GDDS dataset.

Model defaults:

- vocab size: 2048
- context length: 512 tokens
- `d_model = 256`
- `n_layers = 7` by default, about 7.87M trainable parameters
- `n_heads = 4`
- `head_dim = 64`
- local causal `sdpa_varlen` sliding window: 256
- tied LM head: token embedding weight is reused by `gd_linear_transposed_weight`
- RMSNorm, RoPE, PoWLU gated MLP, and dropout

Build and run:

```sh
make -C examples/gpt_lm run ARGS="--epochs 2 --batch-size 32"
```

Overfit smoke tests:

```sh
make -C examples/gpt_lm run ARGS="--epochs 2 --batch-size 1 --overfit-num-samples 1 --report-every 1"
make -C examples/gpt_lm run ARGS="--epochs 1 --batch-size 32 --overfit-num-samples 32 --report-every 1"
```

Useful runtime options:

```text
--epochs N
--batch-size N
--layers N
--dropout P
--overfit-num-samples N
--report-every N
--lr-max LR
--lr-min LR
--warmup-steps N    # default -1 means total_steps / 10
```

The GDDS dataloader provides packed fields directly:

```text
input_ids   i32 [B * 512]
target_ids  i32 [B * 512]
cu_seqlens  i32 [B + 1]
positions   i32 [B * 512]
```

No generation path is implemented yet.
