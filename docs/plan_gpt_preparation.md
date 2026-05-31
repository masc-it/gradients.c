# gradients.c — Text GPT Training Preparation

Status: draft v0.1

Goal: turn the existing text-only GPT core into a production-ready training and
inference stack. Current model math is mostly present; missing pieces are the
systems around it: tokenizer, dataset build, fast loader, mutable training
controls, checkpoint/resume, eval/sampling, and a mandatory KV-cache inference
path.

Target first real-training shape:

```text
context_len = 512
sequence format = <|im_start|>...<|im_end|>
packing = contiguous 512-token blocks, incomplete tail dropped
training = next-token LM, tokens[B,512] -> targets[B,512]
```

Speed is a core principle. Build everything so the hot training loop does not do
text parsing, tokenization, allocation, graph rebuilds, or synchronous CPU
readbacks.

References:
- [`docs/plan_gpt.md`](plan_gpt.md) for current GPT op/model state.
- [`docs/plan_prefix_sdpa.md`](plan_prefix_sdpa.md) for recent SDPA mask work.

---

## 1. Current state

Already available:

- [x] GPT builder: token embedding, RoPE, causal SDPA, GQA, RMSNorm, MLP, tied
  LM head.
- [x] Mean CE and fused tied-LM-head CE.
- [x] AdamW.
- [x] Static train graph reusable for fixed `B,T`.
- [x] CPU_REF and Metal parity tests.
- [x] Benchmark harness with dummy/random token data.

Not enough for real GPT training:

- [ ] tokenizer stack
- [ ] corpus preprocessing / packed token shards
- [ ] production data loader
- [ ] LR scheduler without graph rebuilds
- [ ] global grad clipping
- [ ] checkpoint/resume
- [ ] parameter groups
- [ ] eval/generation loop
- [ ] KV cache

---

## 2. Target training pipeline

```text
raw records
  -> validate/normalize records formatted <|im_start|>...<|im_end|>
  -> byte-level BPE tokenizer with special tokens and digit splitting
  -> packed token shards, context_len=512, tail dropped
  -> mmap double-buffer data loader
  -> precompiled train graph:
       GPT forward_loss -> backward -> grad clip -> AdamW(lr_tensor)
  -> periodic eval/checkpoint/sample
```

Train graph stays static. Mutable state enters through persistent tensors:

- batch token tensors for each loader buffer
- target tensors for each loader buffer
- position tensors for each loader buffer
- scalar LR tensor
- optional scalar grad norm output

---

## 3. Tokenizer stack

### 3.1 Requirements

Implement a full byte-level BPE tokenizer stack suitable for GPT training.
Tokenizer v1 must train on our corpus, not only load an existing vocab.

Must support:

- Train BPE vocab/merges from input corpus.
- UTF-8 text input, byte-fallback semantics (no unknown normal text).
- Special tokens reserved and matched as atomic tokens before normal tokenization:
  - `<|im_start|>`
  - `<|im_end|>`
  - `<|pad|>` reserved for future batched generation/eval
  - optional future: `<|bos|>`, `<|eos|>`
- Digits tokenized separately:
  - `"123" -> "1", "2", "3"`
  - prevent BPE merges across digit-digit boundaries during training and encode.
- Encode/decode round trip for non-special text.
- Stable vocab ids and merge ranks across save/load.
- Tokenizer metadata hash stored into dataset/checkpoints.

### 3.2 API sketch

```c
typedef struct gd_tokenizer gd_tokenizer;

typedef struct gd_tokenizer_config {
    int split_digits;       /* v1 true for GPT training */
    int allow_special;      /* exact-match special token parser */
} gd_tokenizer_config;

typedef struct gd_bpe_train_config {
    int vocab_size;
    int min_frequency;
    int split_digits;       /* v1 true */
    int n_special_tokens;
    const char **special_tokens;
    uint64_t seed;          /* deterministic tie breaks / corpus order */
} gd_bpe_train_config;

gd_status gd_bpe_tokenizer_train(const char **input_paths,
                                 int n_input_paths,
                                 const gd_bpe_train_config *cfg,
                                 gd_tokenizer **out);

gd_status gd_bpe_tokenizer_save(gd_tokenizer *tok, const char *tokenizer_path);

gd_status gd_bpe_tokenizer_load(gd_context *ctx,
                                const char *tokenizer_path,
                                const gd_tokenizer_config *cfg,
                                gd_tokenizer **out);
void gd_tokenizer_destroy(gd_tokenizer *tok);

gd_status gd_tokenizer_encode(gd_tokenizer *tok,
                              const char *text,
                              int32_t **ids_out,
                              int *n_ids_out);

gd_status gd_tokenizer_decode(gd_tokenizer *tok,
                              const int32_t *ids,
                              int n_ids,
                              char **text_out);

gd_status gd_tokenizer_id(gd_tokenizer *tok, const char *special, int32_t *id_out);
```

CLI tools:

```bash
gradients-train-bpe \
  --input ~/projects/dnn.c/docs/promessi_sposi.txt \
  --output tokenizer.json \
  --vocab-size 8192 \
  --min-frequency 2 \
  --split-digits \
  --special '<|pad|>' \
  --special '<|im_start|>' \
  --special '<|im_end|>'

gradients-tokenizer-info --tokenizer tokenizer.json
gradients-tokenize --tokenizer tokenizer.json --input train.txt --output train.tokens
```

### 3.3 Implementation notes

Reference implementation is cloned locally:

```text
data/tiktoken  # https://github.com/openai/tiktoken, commit 08a5f3b
```

Use it as design/reference material, especially:

- `data/tiktoken/src/lib.rs` for fast byte-pair encode/merge mechanics.
- `data/tiktoken/tiktoken/_educational.py` for readable BPE training flow.
- `data/tiktoken/tiktoken/core.py` for special-token policy and public API shape.
- `data/tiktoken/tests/` for behavior coverage ideas.

Implementation still belongs in gradients.c C code; do not shell out to Python or
make training depend on tiktoken at runtime.

Quality requirements:

- Tokenizer training is deterministic: same corpus/config => same vocab ids and
  merge ranks.
- Initial alphabet includes byte-level symbols plus reserved specials.
- Special-token trie for exact longest-match before BPE.
- Byte encoder/decoder table like GPT-2/tiktoken.
- BPE trainer uses pair counts + heap/hash invalidation; avoid O(vocab * corpus)
  rescans where possible.
- Encode path follows tiktoken-style rank-ordered byte-pair merging with compact
  arrays and hash lookups.
- BPE merge ranks in a fast hash table.
- Digit pre-tokenizer emits one token span per ASCII digit and disallows merges
  across adjacent digits during training and encode.
- No heap churn per token in hot encode loop; reusable scratch buffers.
- Include perf tests against a fixed corpus slice; regressions over baseline need
  explicit approval.

### 3.4 Tests

- BPE trainer is deterministic for fixed corpus/config.
- Save/load preserves vocab ids and merge ranks.
- `<|im_start|>` and `<|im_end|>` each become one id.
- `<|pad|>` is reserved and never produced from normal text.
- Special tokens embedded in text are preserved.
- `abc123def` encodes digits as three separate digit tokens.
- No learned merge crosses digit-digit boundaries.
- Decode round trip for UTF-8 strings.
- Invalid special-token policy errors clearly.

---

## 4. Dataset preprocessing

### 4.1 Input contract

Input records are already formatted:

```text
<|im_start|>...<|im_end|>
<|im_start|>...<|im_end|>
...
```

Preprocessor validates every record:

- starts with `<|im_start|>`
- ends with `<|im_end|>`
- no malformed special-token spans
- no empty records unless explicitly allowed

Bootstrap corpus for first end-to-end run:

```text
~/projects/dnn.c/docs/promessi_sposi.txt
```

This source is plain text (~1.3 MB), so the dataset builder should support a
bootstrap wrapping mode:

```text
paragraph_or_chunk -> <|im_start|>paragraph_or_chunk<|im_end|>
```

Use this only for smoke/overfit/profiling. Production datasets should provide
already-formatted records.

### 4.2 Train/val split

Dataset builder always writes separate train and validation shards.

Default split:

```text
train = 90%
val   = 10%
```

For tiny bootstrap corpora (including `promessi_sposi.txt`), split by record or
paragraph before packing so validation does not share overlapping 512-token
windows with training. Shuffle records deterministically with `--seed` before the
split unless `--no-shuffle-split` is passed.

CLI controls:

```text
--train-ratio 0.9
--val-ratio 0.1
--seed 1234
```

Output layout:

```text
data/promessi/train-00000.gdtok
data/promessi/train-00000.json
data/promessi/val-00000.gdtok
data/promessi/val-00000.json
```

Metadata must record split counts:

```json
{
  "split": "train|val",
  "train_ratio": 0.9,
  "val_ratio": 0.1,
  "split_seed": 1234,
  "n_records_total": 0,
  "n_records_split": 0
}
```

### 4.3 Packing contract

Tokenize records, concatenate ids into a stream, then pack fixed blocks.

```text
block_len = 512
sample i tokens  = stream[i*512 : i*512 + 512]
sample i targets = stream[i*512 + 1 : i*512 + 513]
```

Drop data that cannot form a full `(tokens, targets)` pair:

- incomplete tail shorter than 513 lookahead tokens is dropped
- final 512 block in a shard is dropped if it lacks the next-token target

This preserves model context length 512 while keeping labels exact.

### 4.4 Shard format

Use mmap-friendly binary shards.

```text
*.gdtok
header:
  magic/version
  dtype: uint16 if vocab <= 65535 else uint32
  vocab_size
  block_len = 512
  n_tokens
  n_samples
  tokenizer_hash
payload:
  packed token ids, contiguous
```

Also write JSON sidecar:

```json
{
  "format": "gdtok-v1",
  "block_len": 512,
  "dtype": "u16|u32",
  "vocab_size": 32000,
  "tokenizer_hash": "...",
  "source_files": [...],
  "n_records": 0,
  "n_tokens": 0,
  "n_samples": 0,
  "dropped_tail_tokens": 0
}
```

### 4.5 CLI

```bash
gradients-build-dataset \
  --tokenizer tokenizer.json \
  --input ~/projects/dnn.c/docs/promessi_sposi.txt \
  --output data/promessi \
  --block-len 512 \
  --train-ratio 0.9 \
  --val-ratio 0.1 \
  --seed 1234 \
  --wrap-plain-text \
  --split-digits \
  --special im_start='<|im_start|>' \
  --special im_end='<|im_end|>'
```

### 4.6 Tests

- train/val split is deterministic for fixed seed.
- train and val record ids are disjoint.
- packed samples produce correct shifted targets.
- tail dropping count is exact per split.
- special token ids appear at record boundaries.
- shard can be mmaped and read without copies.
- tokenizer hash mismatch is detected.

---

## 5. Data loader — double buffered from v1

### 5.1 Requirements

The training loop must never tokenize or parse text. Loader reads pre-tokenized
mmap shards and fills two alternating batch buffers.

Core behavior:

- random sample mode for training
- sequential sample mode for eval
- deterministic seed
- shard shuffle
- resume from loader state
- double buffering from first implementation
- optional CPU worker thread for prefetch

### 5.2 Buffer model

Use two full batch slots:

```c
typedef struct gd_batch_slot {
    gd_tensor *tokens;    /* int32 [B,512] */
    gd_tensor *targets;   /* int32 [B,512] */
    gd_tensor *positions; /* int32 [B,512], usually 0..511 repeated */
} gd_batch_slot;
```

Training alternates slots:

```text
step N:     graph(slot A) runs on device
parallel:   loader fills slot B host buffers
step N+1:   graph(slot B) runs
parallel:   loader fills slot A host buffers
```

Because compiled graphs bind tensor handles, either:

1. compile two train graphs, one per slot, sharing model/optimizer tensors; or
2. add graph input rebinding later.

V1 should compile two graphs. This keeps graph execution static and avoids
runtime rebinding complexity.

### 5.3 Device copy policy

- Fill inactive slot on CPU.
- Copy into inactive device tensors before the graph uses it.
- Avoid overwriting tensors referenced by an in-flight graph.
- Keep positions precomputed once per slot unless sequence length changes.

### 5.4 API sketch

```c
typedef struct gd_token_dataset gd_token_dataset;
typedef struct gd_dataloader gd_dataloader;

typedef struct gd_dataloader_config {
    int batch_size;
    int block_len;       /* 512 */
    uint64_t seed;
    int shuffle;
    int double_buffer;   /* must be true in v1 */
    gd_device device;
} gd_dataloader_config;

gd_status gd_token_dataset_open(const char **paths, int n_paths,
                                gd_token_dataset **out);
void gd_token_dataset_close(gd_token_dataset *ds);

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_token_dataset *ds,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out);
void gd_dataloader_destroy(gd_dataloader *dl);

gd_status gd_dataloader_prefetch(gd_dataloader *dl);
gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch_slot **slot_out);

gd_status gd_dataloader_state_save(gd_dataloader *dl, const char *path);
gd_status gd_dataloader_state_load(gd_dataloader *dl, const char *path);
```

### 5.5 Metrics

Loader reports:

- samples/sec prepared
- host fill ms
- host->device copy ms
- wait-for-batch ms
- dropped samples/tail count

Acceptance: training step should not block on loader after warmup for target
batch sizes on local SSD.

---

## 6. Mutable LR scheduler

Current AdamW stores LR in node attrs at graph construction. That blocks real
schedulers unless graph is rebuilt every step. Fix this before trainer v1.

### 6.1 Required change

Add LR as scalar tensor input to AdamW step.

```c
gd_status gd_optimizer_step_lr(gd_context *ctx,
                               gd_optimizer *optimizer,
                               gd_tensor *lr_scalar);
```

`_GD_OP_ADAMW_STEP` reads `lr` from tensor input instead of fixed attr when
provided. Existing `gd_optimizer_step()` remains for constant LR.

### 6.2 Scheduler policy

Implement host-side scheduler that writes LR scalar before each step:

```text
linear warmup for warmup_steps
cosine decay to min_lr over total_steps
```

Config:

```c
typedef struct gd_lr_scheduler_config {
    float max_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;
} gd_lr_scheduler_config;
```

Tests:

- exact LR values at step 0, warmup end, mid cosine, final.
- AdamW update changes when LR tensor changes without graph rebuild.

---

## 7. Global gradient clipping

Required for stable GPT training.

### 7.1 API

```c
gd_status gd_clip_grad_norm(gd_context *ctx,
                            gd_tensor **params,
                            int n_params,
                            float max_norm,
                            gd_tensor **norm_out);
```

Emits graph work before AdamW:

```text
global_norm = sqrt(sum_p sum(grad_p^2))
scale = min(1, max_norm / (global_norm + eps))
grad_p *= scale
```

### 7.2 Implementation requirements

- Device-side reduction; no per-step CPU readback required.
- CPU_REF oracle first.
- Metal fused/reduction path next.
- Handles missing grads as zeros.
- Writes optional `norm_out` for logging; trainer may read it periodically, not
  every step.

Tests:

- no-op when norm <= max_norm
- scales all grads by common factor when norm > max_norm
- AdamW after clipping matches reference
- CPU↔Metal parity

---

## 8. AdamW parameter groups

Current AdamW uses one config for all params. Add parameter groups before serious
runs.

Needed groups:

- decay: matrix weights (`wq`, `wk`, `wv`, `wo`, MLP matrices, LM head if untied)
- no_decay: RMSNorm weights, biases if added later
- configurable: embeddings decay on/off

API sketch:

```c
typedef struct gd_param_group {
    gd_tensor **params;
    int n_params;
    float weight_decay;
    float lr_scale;
} gd_param_group;

gd_status gd_adamw_create_groups(gd_context *ctx,
                                 const gd_param_group *groups,
                                 int n_groups,
                                 const gd_adamw_config *base,
                                 gd_optimizer **out);
```

Need model parameter tagging from `gd_gpt_create()` or helper:

```c
gd_status gd_gpt_parameter_groups(gd_gpt *gpt,
                                  gd_param_group **groups_out,
                                  int *n_groups_out);
```

---

## 9. Checkpoint / resume

Required for production training.

Checkpoint contents:

- model config
- tokenizer metadata/hash and special ids
- model tensors
- optimizer state tensors (`m`, `v`, step)
- scheduler state
- data loader state (shard order, cursor, RNG)
- global step
- training args

V1 format can be simple:

```text
checkpoint_dir/
  manifest.json
  model.bin
  optimizer.bin
  loader.json
  tokenizer.json or tokenizer_hash.txt
```

Hard requirements:

- atomic write via temp dir + rename
- checksum per tensor file
- exact resume test: train N steps uninterrupted vs M + resume + N-M matches
  within deterministic tolerance
- versioned manifest

Add optimizer state export/import APIs; optimizer internals are currently opaque.

---

## 10. Trainer executable

Add a real trainer, separate from demo/bench.

```bash
gradients-gpt-train --config config.json
```

Config sections:

```json
{
  "model": { "vocab_size": 32000, "d_model": 512, "n_layers": 8, "n_heads": 8, "context_len": 512 },
  "data": { "train": ["data/train-*.gdtok"], "val": ["data/val-*.gdtok"], "batch_size": 16 },
  "optim": { "max_lr": 0.0003, "min_lr": 0.00003, "warmup_steps": 1000, "weight_decay": 0.1, "grad_clip": 1.0 },
  "train": { "steps": 100000, "eval_interval": 1000, "ckpt_interval": 1000, "seed": 1234, "device": "metal" }
}
```

Loop:

```text
init model/optimizer/scheduler/dataloader
compile two train graphs for two batch slots
prefetch both slots
for step in resume_step..total_steps:
    lr = scheduler(step)
    copy lr to lr tensor
    run graph for current slot
    async prefetch next inactive slot
    periodic: read loss/lr/grad_norm, eval, sample, checkpoint
```

Logging:

- step
- train loss
- val loss / perplexity
- LR
- grad norm
- tokens/sec
- loader wait ms
- device/profile summary optional

---

## 11. Eval and generation

### 11.1 Eval

- Sequential validation loader.
- No optimizer/backward.
- Mean loss and perplexity.
- Fixed number of validation tokens for comparable runs.

### 11.2 Generation

Needed for sanity checks and mandatory KV-cache path.

Sampling controls:

- temperature
- top-k
- top-p later
- max_new_tokens
- stop on `<|im_end|>`

CPU sampling from last-token logits is acceptable v1; logits row is small.

---

## 12. KV cache — mandatory

KV cache is required even for text-only GPT release. Training uses full prefill,
but production inference and sample generation must use cached decode.

### 12.1 Requirements

- Prefill prompt of length `Tprefill`.
- Decode one token at a time with `Tq=1`.
- Cache K/V per layer in fixed max context storage.
- No recomputing old K/V during decode.
- Native Metal path; CPU_REF oracle.
- Correct causal semantics with live cache length.

### 12.2 Data model

Per layer:

```text
k_cache[L][B,max_seq_len,Hkv,Dh]
v_cache[L][B,max_seq_len,Hkv,Dh]
cache_len[B] or scalar for uniform batch decode
```

V1 can require uniform `cache_len` across batch. Per-row lengths can come later
for batched variable-length generation.

### 12.3 Ops / API sketch

```c
typedef struct gd_kv_cache gd_kv_cache;

gd_status gd_kv_cache_create(gd_context *ctx,
                             const gd_gpt_config *cfg,
                             int batch_size,
                             int max_seq_len,
                             gd_device device,
                             gd_kv_cache **out);
void gd_kv_cache_destroy(gd_kv_cache *cache);
gd_status gd_kv_cache_reset(gd_kv_cache *cache);

gd_status gd_kv_cache_append(gd_context *ctx,
                             gd_kv_cache *cache,
                             int layer,
                             gd_tensor *k_new,  /* [B,Tnew,Hkv,Dh] */
                             gd_tensor *v_new,  /* [B,Tnew,Hkv,Dh] */
                             gd_tensor *positions);
```

Model APIs:

```c
gd_status gd_gpt_prefill(gd_context *ctx,
                         gd_gpt *gpt,
                         gd_kv_cache *cache,
                         gd_tensor *tokens,
                         gd_tensor *positions,
                         gd_tensor **logits_out);

gd_status gd_gpt_decode_step(gd_context *ctx,
                             gd_gpt *gpt,
                             gd_kv_cache *cache,
                             gd_tensor *token,       /* [B,1] */
                             gd_tensor *position,    /* [B,1] */
                             gd_tensor *cache_len,   /* scalar or [B] */
                             gd_tensor **logits_out);
```

### 12.4 SDPA cache support

Existing SDPA static shapes are not enough for efficient decode because `Tk_live`
changes every step. Add cached decode SDPA path:

```c
gd_status gd_sdpa_decode(gd_context *ctx,
                         gd_tensor *q,       /* [B,1,Hq,Dh] */
                         gd_tensor *k_cache, /* [B,max_seq,Hkv,Dh] */
                         gd_tensor *v_cache, /* [B,max_seq,Hkv,Dh] */
                         gd_tensor *cache_len,
                         gd_tensor **out);
```

Kernel uses `cache_len` as live key bound and ignores unwritten cache slots.
Avoid dense additive masks over `max_seq_len` in production path.

### 12.5 Cache phases

- [ ] K0 CPU_REF cache append + decode SDPA correctness.
- [ ] K1 Metal cache append kernel.
- [ ] K2 Metal decode SDPA kernel with live length bound.
- [ ] K3 GPT prefill/decode builder APIs.
- [ ] K4 generation CLI uses KV cache by default.

Tests:

- cached decode logits match full forward logits token-by-token.
- cache reset clears length/contents semantics.
- append at positions writes exact rows.
- decode stops at `<|im_end|>`.
- CPU↔Metal parity.

---

## 13. Open holes / decisions

These are not optional polish; decide or implement before calling the stack
production-ready.

- [x] Tokenizer source decision: v1 trains a new byte-level BPE from our corpus.
  Add trainer, vocab-size config, min-frequency, and reproducible merge ordering.
- [x] Boundary-loss policy: train across `<|im_end|><|im_start|>` by default.
  Markers make boundaries explicit; no per-token loss mask in v1.
- [x] Padding policy: no padding in training. Fixed 512 and tail dropped.
  Reserve `<|pad|>` for future batched generation/eval.
- [x] Positional policy beyond 512: hard cap v1. Require
  `prompt_len + new_tokens <= 512`; no RoPE extrapolation.
- [x] Config validation: fail fast on tokenizer vocab/model vocab, special ids,
  dataset block_len=512, tokenizer hash, checkpoint hash.
- [x] Initialization policy: GPT-2 style normal std=0.02; scale residual
  projections (`wo`, `w_down`) by `1/sqrt(2*n_layers)`; RMSNorm weights=1; no
  biases.
- [x] Reproducibility: one master seed split into model/split/loader/sample
  seeds. Save RNG + loader state. Exact CPU resume; Metal resume within tolerance.
- [x] Memory/OOM planning: add preflight estimator before compile for params,
  grads, Adam m/v, activations, scratch, double buffers, and KV cache.
- [x] In-flight safety: slot state machine `FILLING -> READY -> IN_USE -> FREE`;
  never overwrite `IN_USE`. V1 graph run may be blocking but states stay.
- [x] Metrics smoothing: JSONL + human logs, raw + EMA for loss, val loss, ppl,
  LR, grad norm, tok/s, loader wait, copy ms.
- [x] NaN/Inf policy: fail fast. Periodic assert-finite loss/grad norm; on
  failure write emergency checkpoint + metadata, then stop.
- [x] Sampling logits policy: generation only computes/reads last-token logits
  via KV decode `[B,1,V]`; avoid full `[B,T,V]` for sampling.
- [x] Threading portability: use pthread worker + condvars v1 behind tiny
  portability wrapper; fallback sync loader if threads unavailable.

---

## 14. Nice-to-have after v1

Not required for first real training, but important later:

- [ ] gradient accumulation
- [ ] mixed precision / bf16
- [ ] activation checkpointing
- [ ] multi-worker data loading
- [ ] distributed training
- [ ] sharded checkpoints
- [ ] tokenizer retraining/tuning experiments beyond v1 production tokenizer
- [ ] graph input rebinding to avoid compiling two train graphs
- [ ] per-sample packed loss masks / ignore index, if boundary-loss policy needs it

---

## 15. Delivery phases

### P0 — Specs and config

- [ ] Lock tokenizer/dataset/trainer config schema.
- [ ] Lock special token ids.
- [ ] Lock `context_len=512` as default.

### P1 — BPE tokenizer

- [x] Clone tiktoken reference under `data/tiktoken`.
- [ ] Study/port tiktoken-style byte-pair encode mechanics into C.
- [ ] Byte-level BPE trainer on our corpus.
- [ ] Reproducible merge ordering, vocab-size config, min-frequency.
- [ ] Byte-level BPE load/save/encode/decode.
- [ ] Special token support with `<|pad|>`, `<|im_start|>`, `<|im_end|>`.
- [ ] Digit splitting during train and encode.
- [ ] Unit + perf tests.

### P2 — Dataset builder

- [ ] Validate `<|im_start|>...<|im_end|>` records.
- [ ] Bootstrap-wrap plain-text corpus (`~/projects/dnn.c/docs/promessi_sposi.txt`).
- [ ] Deterministic train/val split before packing.
- [ ] Tokenize and pack 512-token blocks per split.
- [ ] Drop incomplete/final no-lookahead tail per split.
- [ ] Write mmap shards + metadata for train and val.

### P3 — Double-buffer dataloader

- [ ] mmap shard reader.
- [ ] two batch slots.
- [ ] train/eval sampling modes.
- [ ] loader metrics.

### P4 — Mutable LR + scheduler

- [ ] LR tensor input for AdamW.
- [ ] warmup + cosine scheduler.
- [ ] tests without graph rebuild.

### P5 — Gradient clipping

- [ ] global norm clip op/API.
- [ ] CPU_REF + Metal.
- [ ] trainer integration.

### P6 — Parameter groups + checkpoint/resume

- [ ] no_decay group support.
- [ ] checkpoint manifest + tensors.
- [ ] exact resume test.

### P7 — KV cache + generation

- [ ] cache tensors + append.
- [ ] cached decode SDPA.
- [ ] GPT prefill/decode APIs.
- [ ] generation CLI with KV cache.

### P8 — Trainer v1 hardening

- [ ] full config-driven trainer.
- [ ] eval/perplexity.
- [ ] periodic samples.
- [ ] profile/logging.
- [ ] failure handling.

---

## 16. Acceptance criteria

A first production-ready text GPT training run is accepted when:

- [ ] tokenizer trains on real corpus and encodes it with special tokens and separate digits.
- [ ] dataset builder creates separate train/val 512-token packed shards and reports dropped tail.
- [ ] loader double buffers and reports near-zero wait after warmup.
- [ ] train graph uses mutable LR scheduler and global grad clipping.
- [ ] checkpoint/resume reproduces deterministic training continuation.
- [ ] validation perplexity and sample generation run periodically.
- [ ] KV-cache generation matches full-forward logits on a short prompt.
- [ ] CPU_REF tests pass; Metal parity tests pass.
- [ ] `make check` green.
