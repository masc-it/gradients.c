# gradients.c — GPU-first tensor/autograd runtime in C

> Active development. Not released as a public/stable API yet.

`gradients.c` is a compact C training stack: eager tensor
ops, reverse-mode autograd, explicit arena memory, and first-class accelerated
backends. The current supported backends are **CUDA** and **Metal**.

The goal is not a general graph compiler. Model code is plain C, ops execute
eagerly inside explicit train/eval/infer scopes, and optimization happens by
adding explicit fused ops and backend kernels under `src/ops/<op>/`.

## Examples

### XOR MLP

[`examples/mlp_xor`](examples/mlp_xor/README.md) — full-batch XOR training run
that exercises the v2 API end to end: GDDS dataset generation, random sampler,
module tree, F16 `linear -> relu -> linear`, in-graph Huber loss, autograd,
AdamW, AMP loss scaling, and FP32 master weights for F16 params.

```sh
make -C examples/mlp_xor run
```

### MNIST MLP

[`examples/mlp_mnist`](examples/mlp_mnist/README.md) — compact MNIST classifier.
Dataset prep downloads IDX gzip files, stores images as U8 GDDS shards, then a
sample transform normalizes them to F16 for `linear -> relu -> dropout -> linear`
with in-graph cross entropy.

```sh
make -C examples/mlp_mnist run
make -C examples/mlp_mnist smoke
```

### GPT language model

[`examples/gpt_lm`](examples/gpt_lm/README.md) — decoder-only LM with training,
validation, checkpoints, resume, checkpoint inference, interactive generation,
and KV-cache decode. Defaults are vocab 2048, context 512, `d_model=512`,
3 layers, 8 heads, DH=64, about 9.98M trainable parameters. The model uses
RMSNorm, RoPE, packed variable-length causal attention, SwiGLU, biased dense
projections, an untied biased LM head, fused LM cross entropy, dropout, AdamW
AMP, gradient clipping, and optional logits softcap.

```sh
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict data
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict run ARGS="--epochs 2 --batch-size 64"
make -C examples/gpt_lm GPT_LM_DATASET=ita_dict infer ARGS="--checkpoint checkpoints/gpt_lm_best.gdckpt --prompt '<|im_start|>Termine: casa Definizioni:'"
```

Metrics are written as JSONL and can be watched with:

```sh
uv run tools/gd_dash/main.py --metrics-dir data/metrics
```

## Architecture

### Execution model

Execution is eager and scoped:

```c
gd_begin_step(ctx, GD_SCOPE_TRAIN, batch);
model_forward(ctx, model, inputs, &loss);
gd_backward_amp(ctx, &loss, NULL, scaler);
gd_optimizer_step_amp_clip_lr(ctx, optimizer, scaler, lr, 1.0f);
gd_end_step(ctx);
```

`gd_begin_step()` selects ring-buffer slots, opens the backend command scope,
and enables tape recording for training. `gd_end_step()` submits/records backend
work and attaches fences to scratch/data/state lifetimes.

There is no public graph API, no IR compiler, and no implicit CPU fallback.
If the selected backend does not implement an op, the call returns a `gd_status`
error.

### Memory model

Four arena classes are owned by `gd_context` (`include/gradients/memory.h`,
`src/core/memory.c`):

| Arena | Lifetime | Contents |
|---|---|---|
| `params` | model / optimizer lifetime | parameters, persistent buffers, leaf grads |
| `state` | explicit runtime state | AdamW slots, AMP state, KV caches, long-lived buffers |
| `scratch` | one train/eval/infer step | activations, temporary outputs, tape entries, temporary grads |
| `data` | one loaded batch/input step | GDDS batch tensors and input buffers |

`params` can be sealed after initialization with `gd_context_seal_params()`.
`scratch` and `data` are ring-buffered and guarded by backend fences and slot
generations. Tensor bytes live in backend buffers; tensor descriptors store an
arena span, byte offset, shape, strides, dtype, and view metadata.

### Tensor

`gd_tensor` is a concrete tensor descriptor, not a symbolic graph node.
It supports rank up to 8 (`GD_MAX_DIMS`), row-strided layouts, views, slices,
explicit contiguity, and dtypes `F16`, `BF16`, `F32`, `I32`, and `U8`. The hot
Metal training path is primarily F16 storage with FP32 accumulation/state where
needed.

### Autograd

Autograd is reverse-mode tape (`src/autograd/`). Differentiable ops record op
kind, input/output tensor snapshots, scalar attrs, saved tensors, and a backward
rule. `gd_backward*()` walks the tape in reverse and accumulates gradients into
leaf grad buffers. Tape and temporary gradients live in `scratch`; parameter
grads are persistent and are zeroed/updated by the optimizer.

### Backend

For Metal (`src/backends/metal/`), on macOS the build
compiles Objective-C dispatch code plus all `.metal` files under `src/backends/metal/`,
`src/ops/`, and `src/optim/` into `build/gradients.metallib`. Runtime loads the
metallib from `GRADIENTS_METALLIB`, the app bundle, or `build/gradients.metallib`.

<CUDA WIP>

## Ops

Ops live as capsules under `src/ops/<op>/` with core validation/allocation,
autograd rules, Metal dispatch/kernels, PyTorch oracle scripts, tests, and
optional perf probes.

- **Arithmetic / broadcasting** — `add`, `sub`, `mul`
- **Matrix / projection** — batched `matmul`, `linear`, `linear_transposed_weight`
- **Activations** — `relu`, `sigmoid`, `tanh`, `dropout`, fused `dropout_add`
- **Gated MLPs** — `swiglu`, `swiglu_split_linear`, `powlu`, `powlu_split_linear`
- **Losses** — `mse`, `huber`, `cross_entropy`, fused tied `lm_cross_entropy`, optional LM logits softcap
- **Reductions** — `reduce_sum`, `reduce_mean`, axis reductions and broadcast-backward helpers
- **Tensor layout** — `reshape` views, `slice`, `contiguous`, `concat`, `split`, `permute`
- **Embeddings / norms** — `embedding`, `rms_norm`
- **Transformer hot paths** — `rope`, fused `qkv_split_rope`, packed `sdpa_varlen`, decode-time `sdpa_decode`, and K/V cache append variants

Generated registries wire op kinds, public prototypes, autograd rules, and
backend stubs together. Do not hand-edit generated files such as
`src/ops/op_kind.h`, `src/ops/op_registry.c`, `include/gradients/ops_generated.h`,
or generated backend glue.

## Modules, checkpoints, data

- **Modules** (`src/nn/`) — plain C structs embed `gd_module` for named params,
  buffers, child modules, `ModuleList`/`ModuleDict`, train/eval mode,
  freeze/unfreeze, recursive parameter collection, and param groups.
- **Checkpoints** — `gd_module_save_state()` / `gd_module_load_state()` write a
  backend-independent `.gdckpt` state dictionary with optional metadata.
  AdamW optimizer state has its own save/load path.
- **Datasets** (`src/dataset/`) — GDDS is a self-describing mmap-backed shard
  format with stack, pad-longest, packed-sequence, and generated-field collation.
  Dataloaders prefetch into the `data` arena and expose batch fields as tensors.
- **Tokenizer** (`src/tokenizer/`) — native byte-level BPE tokenizer with train,
  save/load, encode/decode, special tokens, and a `gradients-tokenize` tool.
- **Training helpers** (`src/train/`) — `gd_train_batch()` and
  `gd_eval_mean_loss()` wrap the fragile dataloader/scope/backward/optimizer
  ordering while leaving model forward code in user-owned C.

## Optimizer and AMP

`gradients.c` currently ships AdamW plus a linear-warmup/cosine LR helper:

- AdamW defaults: `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, weight decay `0.01`, bias correction on
- FP32 `m`/`v` moments in `state`
- FP32 master weights for F16 trainable params
- gradient clipping by global norm
- device-side dynamic loss scaling via `gd_amp_scaler`
- AMP optimizer step can unscale grads, check finite values, skip/update params,
  update scaler state, and advance optimizer state without a CPU loss-scale control dependency

See [`docs/guides/adamw_amp.md`](docs/guides/adamw_amp.md).

## Metal optimizations

The Metal backend is intentionally explicit rather than compiler-driven:

- offline metallib and pre-created PSOs for op-local kernels
- scoped command-buffer batching; blocking reads/writes synchronize explicitly
- custom F16 GEMM kernels for normal/transposed/batched training paths
- fused ops for common transformer traffic: `dropout_add`, `qkv_split_rope`,
  `swiglu_split_linear`, `powlu_split_linear`, and `lm_cross_entropy`
- saved forward stats for expensive backward paths: cross entropy row stats,
  RMSNorm inverse RMS, and SDPA stats
- packed variable-length attention with DH=64 prefix/window fast paths and
  decode-time Tq=1 kernels
- vectorized embedding/split/permute paths for common contiguous shapes
- device-side AdamW/AMP and gradient norm/clip kernels
- arena/ring reuse guarded by fences instead of per-tensor allocations

Performance probes live either under `probes/` or as op-local
`src/ops/<op>/perf_test.c` files.

## Adding an op

Use the scaffold tool when possible:

```sh
make tools
build/tools/gradients-new-op my_op
build/tools/gradients-new-op --binary --f16-only my_op
```

Then implement the capsule under `src/ops/my_op/` and validate it with C tests,
PyTorch oracle scripts, and an op-local perf probe:

```sh
uv run src/ops/my_op/fwd.py
uv run src/ops/my_op/bwd.py
make test
make op-perf OP=my_op
```

Full guide: [`docs/guides/register_op.md`](docs/guides/register_op.md). Metal
ownership/performance notes: [`docs/guides/metal_capsules.md`](docs/guides/metal_capsules.md)
and [`docs/guides/metal_tips.md`](docs/guides/metal_tips.md).

## Project structure

```text
include/gradients/       Public C API headers
src/core/                context, arenas, tensor runtime, backend bridge
src/autograd/            eager reverse-mode tape
src/backends/metal/      Metal backend, command scopes, shared buffers, PSOs
src/backends/null/       unsupported compile-only backend
src/ops/<op>/            op capsules: core, autograd, Metal, oracles, perf probes
src/optim/               AdamW, AMP scaler, LR schedule, optimizer checkpoints
src/nn/                  modules, parameter traversal, checkpoint IO
src/dataset/             GDDS datasets, samples, dataloader, batching
src/tokenizer/           native byte-level BPE tokenizer
src/train/               train/eval transaction helpers
examples/                XOR, MNIST, GPT LM examples
tests/                   C unit/correctness tests
tools/                   op generator, GDDS/tokenizer utilities, metrics dashboard
probes/                  standalone design/performance probes
bench/metal/             ad-hoc Metal GEMM precision/throughput probes
docs/guides/             op registration, Metal, AdamW/AMP notes
```

## Build

Requires macOS with Metal for real execution. The core is C11 plus Objective-C
for Metal dispatch.

```sh
make build                    # build libgradients.a + gradients.metallib
make test                     # build and run C tests
make tools                    # build gen_ops, gradients-new-op, gradients-tokenize
make metal-probe              # standalone Metal arena probe
make op-perf OP=relu          # optimized op-local performance probe
make SAN=1 test               # sanitizer test build
```

Example builds use their own `build-<example>` core directories through
`examples/common.mk`.

## Design principles

- GPU-first: no CPU execution backend and no hidden fallback.
- Eager scoped execution, not graph capture exposed as a public API.
- Explicit arenas and lifetimes; no per-tensor heap allocation in the hot path.
- Add fusion by adding explicit ops/kernels, not by relying on a compiler pass.
- Model code stays plain C with typed forward functions.
- Public APIs return `gd_status`; assertions are for internal invariants.
- Tests compare behavior against PyTorch oracle scripts and C tests.
- Generated registries are generated; op capsules are the source of truth.
