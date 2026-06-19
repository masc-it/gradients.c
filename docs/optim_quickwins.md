# GPT-LM `arch=gpt` optimization quick wins

This note records performance optimization candidates for `examples/gpt_lm` when running with
`--architecture gpt`.

The intent is deliberately conservative: proposals below are limited to places where the current
code path, tensor shapes, and wasted work are directly visible in the repository. This is **not** a
list of generic transformer folklore. Items that would change model semantics, training data
semantics, or quality-risk architecture choices are called out separately and not recommended as
quick wins.

## Scope and current assumptions

Target code path:

- Example: `examples/gpt_lm`
- Architecture: `--architecture gpt`
- Backend of interest: current Metal backend
- Default GPT shape constants from `examples/gpt_lm/gpt_lm_shared.h`:
  - `GPT_CONTEXT_LENGTH = 512`
  - `GPT_D_MODEL = 512`
  - `GPT_N_HEADS = 8`
  - `GPT_HEAD_DIM = 64`
  - `GPT_SDPA_WINDOW = 256`
  - `GPT_MLP_HIDDEN = 1024`
  - default batch size: `64`

Default training tokens per step:

```text
64 rows * 512 tokens = 32768 tokens/step
```

Current `arch=gpt` block path in `examples/gpt_lm/gpt_lm_shared.c`:

```text
embedding
optional embedding dropout
for each block:
  rms_norm
  qkv linear
  qkv_split_rope
  sdpa_varlen causal sliding-window attention
  attn projection
  dropout_add residual
  rms_norm
  up_gate linear
  swiglu_split_linear
  dropout_add residual
final rms_norm
lm_cross_entropy_softcapped_bias
```

## Priority summary

| Priority | Proposal | Applies to | Confidence | Risk | Expected type of win |
|---:|---|---|---:|---:|---|
| 1 | Precompute RoPE sin/cos table for GPT shapes | Train + inference | Very high | Low | Removes millions of per-step transcendental evaluations |
| 2 | Generation prefill: project only final prompt row(s) | Inference/generation | Very high | Very low | Avoids unused LM-head logits for all prompt tokens |
| 3 | Compact `sdpa_varlen` launch geometry for packed short segments | Train + inference prefill | High | Medium | Avoids structurally inactive attention threadgroups |

Recommended implementation order:

1. RoPE table.
2. Generation `prefill_last_logits` path.
3. SDPA varlen launch compaction after benchmarking the first two.

---

## Quick win 1: precompute RoPE sin/cos table

### Current behavior

The current fused QKV split + RoPE op computes RoPE frequencies and trigonometric functions inside
the Metal kernel.

Relevant file:

```text
src/ops/qkv_split_rope/metal_qkv_split_rope.metal
```

Current forward kernel does, per token and per RoPE pair:

```c
const float angle = gd_qkv_split_rope_pair_angle(args, pair, pos[token]);
const float c = cos(angle);
const float s = sin(angle) * args.sin_sign;
```

The helper computes:

```c
return float(position) * exp2(float(pair) * args.freq_scale);
```

The same pattern exists in the backward kernel.

The dispatch size is set in:

```text
src/ops/qkv_split_rope/metal_qkv_split_rope.m
```

as:

```c
thread_count = args.tokens * (args.head_dim / 2U);
```

For the default GPT training shape:

```text
tokens per step                  = 32768
head_dim / 2                     = 32
RoPE pairs per layer per forward = 32768 * 32 = 1,048,576
layers                           = 3
forward + backward               = 2 kernel passes
```

So the current implementation performs approximately:

```text
1,048,576 * 3 * 2 = 6,291,456
```

`exp2`/`cos`/`sin` evaluations per optimizer step for RoPE alone.

That count is derived directly from the kernel grid and GPT constants; it does not rely on external
assumptions.

### Proposal

Add a precomputed RoPE table for the GPT case:

```text
cos_table[GPT_CONTEXT_LENGTH][GPT_HEAD_DIM / 2]
sin_table[GPT_CONTEXT_LENGTH][GPT_HEAD_DIM / 2]
```

For current constants:

```text
512 positions * 32 pairs * 2 tables * sizeof(float) = 131072 bytes = 128 KiB
```

Then add a fast path in `qkv_split_rope` that uses table loads instead of computing:

```text
exp2 + cos + sin
```

at every token/pair.

The generic dynamic path should remain as fallback for non-GPT shapes or unusual RoPE configs.

### Suggested fast-path conditions

Use the table only when all are true:

```text
qkv dtype              == f16
head_dim               == 64
n_dims                 == 64
interleaved            == false
positions are in range 0 <= pos < 512
theta                  == 10000.0f
```

These match the current GPT path:

```c
const gd_rope_config rope_cfg = {
    .theta = 10000.0f,
    .n_dims = GPT_HEAD_DIM,
    .interleaved = false,
};
```

### Implementation shape

Possible implementation approaches:

1. Add a new `qkv_split_rope` backend path accepting optional sin/cos tensors.
2. Add a GPT-specialized op, e.g. `qkv_split_rope_table`, leaving the existing op untouched.
3. Store the table as persistent backend/state tensors attached to the model or context.

Most conservative approach:

- Keep existing `gd_qkv_split_rope` behavior unchanged.
- Add a separate table-backed op and call it from `examples/gpt_lm` when the GPT constants match.
- Retain existing op as fallback.

### Correctness test

For random `qkv` and random positions in `[0, 511]`:

1. Run existing `gd_qkv_split_rope`.
2. Run table-backed variant.
3. Compare `q`, `k`, `v` outputs.
4. Run backward comparison for `grad_qkv`.

Accept only small f16-level differences.

### Why this is high-confidence

- RoPE angle depends only on `(position, pair, theta, n_dims, interleaved)`.
- In GPT, those are fixed and bounded.
- The table is tiny.
- The current kernel visibly recomputes deterministic values millions of times per step.
- This preserves tensor shapes and model math.

### Expected impact

This removes expensive transcendental work from the RoPE split forward/backward kernels. Whole-step
speedup depends on the share of time spent outside GEMMs, but this is a clean ops-level win with low
risk.

---

## Quick win 2: generation prefill should project only final prompt row(s)

This is an inference/generation win. It does not affect training.

### Current behavior

Generation calls:

```text
gpt_lm_prefill_logits(...)
```

in:

```text
examples/gpt_lm/gpt_lm_shared.c
```

Inside `gpt_lm_prefill_logits`, after computing all prompt hidden states, the current code projects
**every prompt token** through the LM head:

```c
st = gd_linear_transposed_weight(ctx, &final_norm, &model->lm_head, &model->lm_head_bias, logits);
```

Then the generation code slices only the last token's logits per prompt:

```c
TRY(ctx, gd_tensor_slice(ctx, &logits, 0U, (int64_t)cu[b + 1] - 1, 1, &last_logits[b]));
```

So for a single prompt of length `P`, the code computes logits for:

```text
P rows
```

but uses logits for only:

```text
1 row
```

For batched prompts, it computes all prompt-token logits and uses only one row per prompt.

### Proposal

Add a generation prefill function that returns only final prompt logits:

```c
gpt_lm_prefill_last_logits(...)
```

Logical flow:

```text
embedding all prompt tokens
run all transformer blocks over all prompt tokens, filling KV cache
final rms_norm over all prompt hidden states
select final hidden row per prompt: cu[b + 1] - 1
LM-head projection only for those selected rows
```

For single interactive generation this can be very simple:

```text
slice final_norm last row
project [1, D] -> [1, vocab]
```

For batched generation, add a small gather/select-last-rows helper if no suitable op already exists.

### Why this is mathematically safe

The LM head projection is row-wise independent:

```text
logits[row] = hidden[row] @ lm_head.T + bias
```

Projecting only row `cu[b + 1] - 1` produces exactly the same logits as projecting all rows and then
slicing that row.

### Expected impact

For prompt length `P`, the final prefill LM-head projection work changes from:

```text
P * D_MODEL * VOCAB_SIZE
```

to:

```text
1 * D_MODEL * VOCAB_SIZE
```

for single-prompt generation.

For a 300-token prompt, that LM-head projection component is about 300x smaller.

Whole prefill still must run transformer blocks over all prompt tokens to populate KV cache, so this
does not make prompt prefill 300x faster overall. But the eliminated work is exact and currently
unused.

### Suggested implementation path

1. Split `gpt_lm_prefill_logits` into two variants:
   - existing all-token logits path, kept for tools/eval if needed
   - new last-token logits path for generation
2. For `n_prompts == 1`, use `gd_tensor_slice` on `final_norm` before LM head.
3. For `n_prompts > 1`, either:
   - loop over prompts and project slices, or
   - add a small gather-last-rows op for better batching.

Most conservative first patch:

- Optimize only `n_prompts == 1`, which is the interactive path.
- Keep batched vowels/generation behavior unchanged until a gather op is available.

### Correctness test

For a fixed prompt:

1. Run existing `gpt_lm_prefill_logits` and slice last logits.
2. Run new `gpt_lm_prefill_last_logits`.
3. Compare logits exactly or within f16/f32 readback tolerance.
4. Verify generated token IDs match at temperature `0`.

### Why this is high-confidence

- The current code visibly computes all prompt logits and immediately slices only final rows.
- The LM head is independent per row.
- The change is inference-only if wired only into generation.
- No model parameters, logits math, or KV cache semantics change.

---

## Quick win 3: compact `sdpa_varlen` launches for packed short segments

This is a real source of wasted attention launch work on the current packed dictionary dataset. It is
more involved than the first two items, so it should be implemented after measuring them.

### Current behavior

GPT attention calls:

```c
gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
```

with config:

```c
const gd_sdpa_varlen_config sdpa_cfg = {
    .scale = 0.0f,
    .causal = true,
    .sliding_window = GPT_SDPA_WINDOW,
    .prefix_len = 0,
    .max_seqlen = GPT_CONTEXT_LENGTH,
};
```

For GPT:

```text
sliding_window = 256
max_seqlen     = 512
head_dim       = 64
```

The Metal fast path in:

```text
src/ops/sdpa_varlen/metal_sdpa_varlen.m
```

computes:

```c
p.n_qb_max = (args->max_seqlen + GD_METAL_SDPA_CAUSAL_QROWS - 1U) /
             GD_METAL_SDPA_CAUSAL_QROWS;
```

`GD_METAL_SDPA_CAUSAL_QROWS` is `16`, so with `max_seqlen=512`:

```text
n_qb_max = ceil(512 / 16) = 32
```

It then launches approximately:

```text
segment_count * heads * n_qb_max
```

threadgroups.

That means every packed document segment receives 32 query-block groups, even when the segment has
far fewer than 512 tokens.

### Dataset evidence

I parsed the current `examples/gpt_lm/data/train-00000.gdds` `segment_lengths` field.

Observed training split stats:

```text
samples:               12637
segments:              34890
total tokens:           6470144
avg segment length:     185.4
median segment length:  135
p90 segment length:     462
p99 segment length:     512
max segment length:     512
```

Fractions:

```text
segment <= 16 tokens:   4.1%
segment <= 32 tokens:   8.0%
segment <= 64 tokens:   16.1%
segment <= 128 tokens:  47.9%
segment <= 192 tokens:  69.3%
segment <= 256 tokens:  77.5%
```

Query-block launch math using 16 rows per query block:

```text
active qblocks  = sum(ceil(segment_len / 16)) = 419674
launched blocks = segment_count * 32            = 1116480
active fraction = 37.6%
waste factor    = 2.66x
```

So about 62% of query-block groups are structurally inactive for this dataset under the current
rectangular launch geometry.

### Proposal

Replace the rectangular `(segment, qblock)` launch with a compact worklist of actual query blocks.

Conceptually:

```text
for each segment b:
  for qb in 0 .. ceil(segment_len[b] / 16)-1:
    append (b, qb)
```

Then launch:

```text
actual_qblocks * heads
```

instead of:

```text
segment_count * fixed_n_qb_max * heads
```

The kernel maps a compact work item back to `(segment, query_block)`.

### Scope

To keep this correct, compaction must cover:

- forward attention kernel
- forward stats kernel if used
- backward DQ path
- backward DKV path or equivalent per-key-block compaction

The forward-only version is easier, but training performance needs backward too.

### Why this is high-confidence

- Current fixed `n_qb_max=32` follows directly from code.
- Actual segment length distribution was measured from the current GDDS shard.
- Inactive query blocks return early in the kernel because `i >= T`.
- Launching only active blocks does not change attention math.

### Caveat

This is not as small a patch as the RoPE table or generation-prefill fix. It likely requires:

- auxiliary compact block index generation from `cu_seqlens`, or
- a GPT-specific SDPA op that understands packed lengths, or
- CPU-side precomputed block metadata in the dataloader/batch.

Because it touches SDPA forward/backward dispatch, it should be done after the two simpler wins and
benchmarked carefully.

---

## Non-recommended or lower-confidence ideas

The following are intentionally not recommended as quick wins under the requested standard.

### Do not start by replacing generic GEMM/matmul

The current Metal matmul path already delegates large dense GEMMs to MPS where appropriate.

A quick sanity run disabling MPS matmul thresholds made the one-batch GPT training run slower:

```text
baseline after warmup:        ~50k tok/s
MPS effectively disabled:     ~36-37k tok/s
```

So the current matmul path is already doing important work. Replacing it is not the lowest-risk quick
win.

### Do not change `GPT_SDPA_WINDOW` as a performance quick win

Reducing `GPT_SDPA_WINDOW=256` would reduce attention work, but it changes model context behavior and
training semantics. It is not a pure ops optimization.

### Do not collapse packed document boundaries

Treating each 512-token row as one uninterrupted sequence would simplify attention and reduce varlen
overhead, but it lets unrelated packed documents attend to each other. That changes the data/training
semantics and is not safe.

### Do not switch to MQA/GQA as a quick win

MQA/GQA would reduce K/V and attention costs, but changes the model architecture and likely quality.
It may be worth a separate experiment, but it is not a 99%-safe optimization.

### Do not treat `--dropout 0` or `--logits-softcap 0` as architecture/ops wins

Those can change training behavior. Quick smoke tests did not show a sufficiently clear speed win to
justify recommending them as performance patches.

### Do not focus on MiniMax for this task

This document is specifically for `--architecture gpt`. MiniMax-specific sparse attention is outside
scope.

---

## Suggested benchmark protocol

Use a stable, short overfit run to compare patches before/after. Example:

```sh
GRADIENTS_METALLIB=build-gpt_lm/gradients.metallib \
examples/gpt_lm/gd_main_gpt_lm \
  --data-dir examples/gpt_lm/data \
  --architecture gpt \
  --epochs 5 \
  --batch-size 64 \
  --overfit-num-samples 64 \
  --report-every 1 \
  --no-save-best \
  --no-save-latest \
  --early-stopping-patience 0 \
  --no-metrics \
  --generate-every-n-steps 0
```

Ignore the first step for warmup and compare later `tok/s`.

For generation prefill, use fixed prompt(s), fixed checkpoint, and `--temperature 0`; verify selected
next token is unchanged.

---

## Final recommendation

Implement in this order:

1. **RoPE sin/cos table fast path** for GPT shapes.
2. **Generation `prefill_last_logits` path**, at least for `n_prompts == 1`.
3. **Compact SDPA varlen launches** for packed short segments, once the simpler wins are measured.

The first two are the most certain quick wins: they remove work that is visibly redundant while
preserving the model computation.
