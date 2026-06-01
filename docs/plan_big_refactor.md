# gradients.c — Big Refactor Plan: Operator Capsules + C Codegen

Status: proposal
Goal: keep `gradients.c` maintainable as ops and backends grow (CPU_REF, Metal, CUDA, Vulkan) while keeping source files small and agent-friendly.

---

## 1. Problem

Current design centralizes too much op-specific logic:

- op enum and names in graph internals
- public wrappers in one op schema file
- shape/meta logic in one shape file
- autograd derivative rules in one switch
- CPU execution in one large switch
- Metal host encode/compile/fusion logic in one very large file
- Metal shaders in one very large `.metal` file

This does not scale to CUDA/Vulkan. Each new op or backend adds more scattered touchpoints and larger files.

Refactor target:

- one operator lives in one directory
- backend implementation files use backend prefix
- codegen discovers files by naming convention
- no Python; C/preprocessor only
- no source implementation file over 1k LOC
- backend runtime files contain scheduling/storage only, not op logic

---

## 2. Design principles

1. **Operator capsule.** Each op owns its core schema, meta, grad rule, and backend impl files under `src/ops/<op>/`.
2. **Backend prefix.** Backend-specific files start with backend name: `cpu_`, `metal_`, `cuda_`, `vulkan_`.
3. **Role suffix.** Role appears at end: `_fwd`, `_bwd`, `_wbwd`, etc.
4. **Filename is contract.** C generator derives op kind, registry entries, and expected symbols from filenames. It does not parse C.
5. **Core semantics backend-neutral.** Core op and grad rules emit IR only. They do not know Metal/CUDA/Vulkan details.
6. **Backend runtime generic.** Backend runtime owns storage, compile loop, scheduling, synchronization. Per-op support/plan/encode/launch lives in op capsule.
7. **Small files.** Fail CI if implementation file exceeds 1k LOC. Split complex ops inside their op dir.
8. **Generated boilerplate only.** Generated registries live under `build/generated/`; source of truth remains human-readable op capsule files.

---

## 3. Target layout

### 3.1 Operator capsules

```text
src/ops/
  add/
    core_add_fwd.c
    grad_add.c
    cpu_add_fwd.c
    metal_add_fwd.m
    metal_add_fwd.metal
    cuda_add_fwd.cu          # future
    vulkan_add_fwd.c         # future

  relu/
    core_relu_fwd.c
    grad_relu.c
    core_relu_bwd.c          # internal RELU_BWD op schema/meta
    cpu_relu_fwd.c
    cpu_relu_bwd.c
    metal_relu_fwd.m
    metal_relu_fwd.metal
    metal_relu_bwd.m
    metal_relu_bwd.metal

  sdpa/
    core_sdpa_fwd.c
    grad_sdpa.c
    core_sdpa_bwd.c
    cpu_sdpa_fwd.c
    cpu_sdpa_bwd.c
    metal_sdpa_fwd.m
    metal_sdpa_fwd.metal
    metal_sdpa_bwd.m
    metal_sdpa_bwd.metal
```

Notes:

- Not every op needs every file.
- `grad_<op>.c` is backend-neutral backward rule for public forward op.
- `core_<op>_bwd.c` exists only when backward is represented as an internal IR op, e.g. `RELU_BWD`, `SDPA_BWD`.
- Do **not** create `cpu_add_bwd.c` unless there is an actual `ADD_BWD` IR op. Add backward can compose existing ops.

### 3.2 Backend runtimes

```text
src/backends/cpu_ref/
  backend.c             # vtable glue
  runtime.c             # executable/value storage glue
  kernels_common.c      # common scalar helpers only
  cpu_op.h              # CPU op entry type

src/backends/metal/
  backend.m             # vtable glue
  runtime.m             # device/queue/library/pipeline cache
  storage.m             # alloc/free/upload/download/host ptr
  compile.m             # generic graph planning loop
  execute.m             # generic command buffer loop
  fusion.m              # backend-local fusion planner
  metal_op.h            # Metal op entry + plan/encode ctx
  metal_common.h
  metal_common.metal
```

Per-op Metal code moves out of `src/backends/metal/metal_backend.m` and into `src/ops/<op>/metal_<op>_<role>.m`.

Per-op shader code moves out of `src/backends/metal/kernels.metal` and into `src/ops/<op>/metal_<op>_<role>.metal`.

---

## 4. Filename grammar

Generator discovers `src/ops/*/*` and applies this grammar:

```text
core_<op>_fwd.c          -> core op definition for _GD_OP_<OP>
core_<op>_bwd.c          -> core op definition for _GD_OP_<OP>_BWD
core_<op>_<role>.c       -> core op definition for _GD_OP_<OP>_<ROLE>

grad_<op>.c              -> autograd rule for _GD_OP_<OP>

cpu_<op>_<role>.c        -> CPU_REF backend entry for _GD_OP_<OP>_<ROLE>
metal_<op>_<role>.m      -> Metal host entry for _GD_OP_<OP>_<ROLE>
metal_<op>_<role>.metal  -> Metal shader source for _GD_OP_<OP>_<ROLE>
cuda_<op>_<role>.cu      -> CUDA backend entry for _GD_OP_<OP>_<ROLE>
vulkan_<op>_<role>.c     -> Vulkan backend entry for _GD_OP_<OP>_<ROLE>
```

Examples:

```text
core_rms_norm_fwd.c       -> _GD_OP_RMS_NORM
core_rms_norm_bwd.c       -> _GD_OP_RMS_NORM_BWD
core_rms_norm_wbwd.c      -> _GD_OP_RMS_NORM_WBWD
metal_sdpa_bwd.m          -> _GD_OP_SDPA_BWD
cpu_lm_cross_entropy_bwd.c -> _GD_OP_LM_CROSS_ENTROPY_BWD
```

Canonical naming:

- `<op>` is lowercase snake_case.
- `<role>` is lowercase snake_case.
- `_fwd` maps to base op, not `_FWD` suffix.
- Non-`fwd` roles append uppercase role to op kind.

---

## 5. Symbol contract

Generator emits `extern` declarations based on filenames. Each source file must define expected symbol.

### 5.1 Core op file

File:

```text
src/ops/relu/core_relu_bwd.c
```

Must define:

```c
const _gd_op_def _gd_opdef_relu_bwd = { ... };
```

File:

```text
src/ops/add/core_add_fwd.c
```

Must define:

```c
const _gd_op_def _gd_opdef_add = { ... };
```

### 5.2 Grad rule file

File:

```text
src/ops/relu/grad_relu.c
```

Must define:

```c
const _gd_bwd_rule _gd_bwd_rule_relu = { ... };
```

### 5.3 CPU file

File:

```text
src/ops/relu/cpu_relu_bwd.c
```

Must define:

```c
const _gd_cpu_op _gd_cpu_op_relu_bwd = { ... };
```

### 5.4 Metal host file

File:

```text
src/ops/relu/metal_relu_bwd.m
```

Must define:

```c
const _gd_metal_op _gd_metal_op_relu_bwd = { ... };
```

### 5.5 Link failure is validation

If file exists but expected symbol is missing, generated registry references it and link fails. This catches drift early.

---

## 6. Core op API

### 6.1 Op kind generation

`tools/gen_ops.c` emits:

```text
build/generated/op_kind.h
```

Example output:

```c
typedef enum _gd_op_kind {
    _GD_OP_INVALID = 0,
    _GD_OP_ADD,
    _GD_OP_MUL,
    _GD_OP_RELU,
    _GD_OP_RELU_BWD,
    _GD_OP_SDPA,
    _GD_OP_SDPA_BWD,
    _GD_OP_COUNT
} _gd_op_kind;
```

`src/graph/graph_internal.h` includes generated op kind header.

### 6.2 Op definition

```c
typedef enum _gd_op_flags {
    GD_OPF_PUBLIC      = 1u << 0,
    GD_OPF_INTERNAL    = 1u << 1,
    GD_OPF_DIFF        = 1u << 2,
    GD_OPF_MUTATES     = 1u << 3,
    GD_OPF_SIDE_EFFECT = 1u << 4,
    GD_OPF_DEBUG       = 1u << 5,
    GD_OPF_BROADCAST   = 1u << 6,
    GD_OPF_AUX_OUTS    = 1u << 7
} _gd_op_flags;

typedef gd_status (*_gd_meta_fn)(const gd_tensor_desc *const *inputs,
                                 int n_inputs,
                                 _gd_op_attrs *attrs,
                                 gd_tensor_desc *outputs,
                                 int *n_outputs);

typedef struct _gd_op_def {
    _gd_op_kind kind;
    const char *name;
    int min_inputs;
    int max_inputs;
    int n_outputs;
    unsigned flags;
    _gd_meta_fn meta;
} _gd_op_def;
```

### 6.3 Op registry generation

`tools/gen_ops.c` emits:

```text
build/generated/op_registry.inc
```

Example:

```c
extern const _gd_op_def _gd_opdef_add;
extern const _gd_op_def _gd_opdef_relu;
extern const _gd_op_def _gd_opdef_relu_bwd;

static const _gd_op_def *const g_op_defs[_GD_OP_COUNT] = {
    [_GD_OP_ADD] = &_gd_opdef_add,
    [_GD_OP_RELU] = &_gd_opdef_relu,
    [_GD_OP_RELU_BWD] = &_gd_opdef_relu_bwd,
};
```

Core lookup:

```c
const _gd_op_def *_gd_op_def_for(_gd_op_kind kind);
const char *_gd_op_kind_name(_gd_op_kind kind);
gd_status _gd_op_validate_arity(_gd_op_kind kind, int n_inputs, int n_outputs);
bool _gd_op_is_differentiable(_gd_op_kind kind);
```

---

## 7. Core op file example

`src/ops/add/core_add_fwd.c`

```c
#include "../op_impl.h"
#include "gradients/ops/add.h"

static gd_status add_meta(const gd_tensor_desc *const *in,
                          int n_inputs,
                          _gd_op_attrs *attrs,
                          gd_tensor_desc *out,
                          int *n_outputs)
{
    (void)n_inputs;
    (void)attrs;
    *n_outputs = 1;
    return _gd_meta_elementwise(in[0], in[1], &out[0]);
}

const _gd_op_def _gd_opdef_add = {
    .kind = _GD_OP_ADD,
    .name = "add",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF | GD_OPF_BROADCAST,
    .meta = add_meta,
};

gd_status gd_add(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out)
{
    gd_tensor *inputs[2] = {a, b};

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_add argument is NULL");
    }
    return _gd_emit_checked(ctx, _GD_OP_ADD, inputs, 2, NULL, out, 1);
}
```

---

## 8. Meta layer

Meta functions validate and infer:

- dtype
- device
- rank
- shape
- layout constraints
- output descriptors
- normalized attrs

Meta functions do not:

- allocate storage
- emit graph nodes
- call backend APIs
- inspect tensor data

Shared helpers live in:

```text
src/ops/meta_common.c
src/ops/meta_common.h
```

Examples:

```c
gd_status _gd_meta_elementwise(const gd_tensor_desc *a,
                               const gd_tensor_desc *b,
                               gd_tensor_desc *out);

gd_status _gd_meta_unary_float(const gd_tensor_desc *x,
                               gd_tensor_desc *out);

gd_status _gd_meta_reduce_to(const gd_tensor_desc *x,
                             const gd_tensor_desc *target,
                             gd_tensor_desc *out);
```

Per-op special meta stays inside op capsule when it is not generic.

---

## 9. Checked emit path

Add helper:

```c
gd_status _gd_emit_checked(gd_context *ctx,
                           _gd_op_kind op,
                           gd_tensor **inputs,
                           int n_inputs,
                           _gd_op_attrs *attrs,
                           gd_tensor **outputs,
                           int n_outputs);
```

Flow:

```text
public wrapper / autograd rule
  -> _gd_emit_checked
     -> require active graph
     -> lookup op def
     -> validate arity
     -> collect input descs
     -> run meta
     -> graph emit single/multi/inplace
```

This removes shape logic from public wrappers and keeps op-specific shape logic next to op schema.

---

## 10. Autograd design

### 10.1 Engine vs rules

```text
src/autograd/autograd.c          # engine only
src/autograd/autograd_internal.h # bwd ctx + helpers for rules
```

Engine owns:

- seed `dL/dloss = 1`
- reverse traversal
- forward value handle cache
- gradient accumulation
- leaf grad writes
- cleanup

Rules live in op capsules:

```text
src/ops/<op>/grad_<op>.c
```

### 10.2 Rule type

```c
typedef struct _gd_bwd_ctx _gd_bwd_ctx;

typedef gd_status (*_gd_bwd_rule_fn)(_gd_bwd_ctx *b,
                                     const _gd_node *node);

typedef struct _gd_bwd_rule {
    _gd_op_kind op;
    _gd_bwd_rule_fn fn;
    const char *unsupported_reason;
} _gd_bwd_rule;
```

Generated registry:

```text
build/generated/bwd_registry.inc
```

Example:

```c
extern const _gd_bwd_rule _gd_bwd_rule_add;
extern const _gd_bwd_rule _gd_bwd_rule_relu;

static const _gd_bwd_rule *const g_bwd_rules[_GD_OP_COUNT] = {
    [_GD_OP_ADD] = &_gd_bwd_rule_add,
    [_GD_OP_RELU] = &_gd_bwd_rule_relu,
};
```

### 10.3 Helpers exposed to grad files

```c
gd_status _gd_bwd_fwd(_gd_bwd_ctx *b, int value_id, gd_tensor **out);
gd_tensor *_gd_bwd_grad(_gd_bwd_ctx *b, int value_id);

gd_status _gd_bwd_accumulate(_gd_bwd_ctx *b, int value_id, gd_tensor *contrib);
gd_status _gd_bwd_accumulate_broadcast(_gd_bwd_ctx *b, int value_id, gd_tensor *contrib);

gd_status _gd_bwd_emit(_gd_bwd_ctx *b,
                       _gd_op_kind op,
                       gd_tensor **inputs,
                       int n_inputs,
                       const _gd_op_attrs *attrs,
                       const gd_tensor_desc *out_desc,
                       gd_tensor **out);

gd_status _gd_bwd_emit_multi(_gd_bwd_ctx *b,
                             _gd_op_kind op,
                             gd_tensor **inputs,
                             int n_inputs,
                             const _gd_op_attrs *attrs,
                             const gd_tensor_desc *out_descs,
                             int n_outputs,
                             gd_tensor **outs);
```

### 10.4 Grad rule example

`src/ops/add/grad_add.c`

```c
#include "../grad_impl.h"

static gd_status add_backward(_gd_bwd_ctx *b, const _gd_node *n)
{
    gd_tensor *go = _gd_bwd_grad(b, n->outputs[0]);

    if (go == NULL) {
        return GD_OK;
    }
    GD_TRY(_gd_bwd_accumulate_broadcast(b, n->inputs[0], go));
    return _gd_bwd_accumulate_broadcast(b, n->inputs[1], go);
}

const _gd_bwd_rule _gd_bwd_rule_add = {
    .op = _GD_OP_ADD,
    .fn = add_backward,
    .unsupported_reason = NULL,
};
```

### 10.5 Backend-neutral rule

Autograd rules emit IR. They never call CPU/Metal/CUDA/Vulkan directly.

Example:

```text
RELU backward rule emits RELU_BWD internal IR op.
Backend compiles RELU_BWD like any other op.
```

---

## 11. CPU backend op API

```c
typedef struct _gd_cpu_exec _gd_cpu_exec;

typedef gd_status (*_gd_cpu_support_fn)(_gd_backend *backend,
                                        const gd_graph *graph,
                                        const _gd_node *node);

typedef gd_status (*_gd_cpu_run_fn)(_gd_cpu_exec *exe,
                                    const _gd_node *node);

typedef struct _gd_cpu_op {
    _gd_op_kind op;
    _gd_cpu_support_fn supports;
    _gd_cpu_run_fn run;
} _gd_cpu_op;
```

Generated registry:

```text
build/generated/cpu_registry.inc
```

Example:

```c
extern const _gd_cpu_op _gd_cpu_op_add;
extern const _gd_cpu_op _gd_cpu_op_relu;
extern const _gd_cpu_op _gd_cpu_op_relu_bwd;

static const _gd_cpu_op *const g_cpu_ops[] = {
    &_gd_cpu_op_add,
    &_gd_cpu_op_relu,
    &_gd_cpu_op_relu_bwd,
};
```

CPU execute loop:

```c
const _gd_cpu_op *op = _gd_cpu_op_for(node->op);
if (op == NULL) {
    return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF missing op implementation");
}
return op->run(exe, node);
```

---

## 12. CPU file example

`src/ops/add/cpu_add_fwd.c`

```c
#include "../cpu_op_impl.h"

static gd_status add_cpu_support(_gd_backend *backend,
                                 const gd_graph *graph,
                                 const _gd_node *node)
{
    (void)backend;
    return _gd_cpu_support_f32_contiguous(graph, node);
}

static gd_status add_cpu_run(_gd_cpu_exec *exe, const _gd_node *node)
{
    return _gd_cpu_run_elementwise(exe, node, _GD_CPU_EW_ADD);
}

const _gd_cpu_op _gd_cpu_op_add = {
    .op = _GD_OP_ADD,
    .supports = add_cpu_support,
    .run = add_cpu_run,
};
```

---

## 13. Metal backend op API

```c
typedef struct _gd_metal_plan_ctx _gd_metal_plan_ctx;
typedef struct _gd_metal_encode_ctx _gd_metal_encode_ctx;

typedef gd_status (*_gd_metal_support_fn)(_gd_backend *backend,
                                          const gd_graph *graph,
                                          const _gd_node *node);

typedef gd_status (*_gd_metal_plan_fn)(_gd_metal_plan_ctx *ctx,
                                       const _gd_node *node);

typedef gd_status (*_gd_metal_encode_fn)(_gd_metal_encode_ctx *ctx,
                                         const _gd_node *node);

typedef struct _gd_metal_op {
    _gd_op_kind op;
    const char *kernels[4];
    _gd_metal_support_fn supports;
    _gd_metal_plan_fn plan;
    _gd_metal_encode_fn encode;
} _gd_metal_op;
```

Generated registry:

```text
build/generated/metal_registry.inc
```

Example:

```c
extern const _gd_metal_op _gd_metal_op_add;
extern const _gd_metal_op _gd_metal_op_relu_bwd;

static const _gd_metal_op *const g_metal_ops[] = {
    &_gd_metal_op_add,
    &_gd_metal_op_relu_bwd,
};
```

Metal runtime uses registry to:

- load all kernels named in `.kernels`
- resolve op support
- run per-node plan
- encode per-node command

---

## 14. Metal file examples

### 14.1 Host entry

`src/ops/add/metal_add_fwd.m`

```objc
#include "../metal_op_impl.h"

static gd_status add_metal_support(_gd_backend *backend,
                                   const gd_graph *graph,
                                   const _gd_node *node)
{
    (void)backend;
    return _gd_metal_support_f32_contiguous(graph, node);
}

static gd_status add_metal_encode(_gd_metal_encode_ctx *ctx,
                                  const _gd_node *node)
{
    return _gd_metal_encode_binary(ctx, node);
}

const _gd_metal_op _gd_metal_op_add = {
    .op = _GD_OP_ADD,
    .kernels = {"gd_add", NULL, NULL, NULL},
    .supports = add_metal_support,
    .plan = NULL,
    .encode = add_metal_encode,
};
```

### 14.2 Shader

`src/ops/add/metal_add_fwd.metal`

```metal
#include "metal_common.metal"

kernel void gd_add(device const float *a [[buffer(0)]],
                   device const float *b [[buffer(1)]],
                   device float *out [[buffer(2)]],
                   constant gd_metal_ew_params &p [[buffer(3)]],
                   uint tid [[thread_position_in_grid]])
{
    if (tid >= (uint)p.numel) {
        return;
    }
    out[tid] = a[tid] + b[tid];
}
```

---

## 15. Backend support errors

Change backend support API from bool to `gd_status` so backend can explain why op is unsupported.

Current concept:

```c
bool (*supports_node)(_gd_backend *self, const _gd_node *node);
```

Target:

```c
gd_status (*supports_node)(_gd_backend *self,
                           const gd_graph *graph,
                           const _gd_node *node);
```

Examples:

```text
backend 'metal' does not support op 'sdpa' node 12: head_dim=192 exceeds max tiled head_dim=128
backend 'cuda' does not support op 'matmul' node 7: BF16 requires compute capability >= 80
backend 'vulkan' does not support op 'softmax' node 5: rank 5 exceeds shader rank limit 4
```

---

## 16. Compile flow

```text
gd_graph_compile(graph, target)
  -> validate graph state
  -> validate graph structure
  -> validate every node against core op registry
  -> lookup target backend
  -> call backend supports_node for every node
  -> if unsupported and fallback policy is CPU_REF: choose CPU_REF for whole graph
  -> else fail with backend reason
  -> backend compile creates executable
```

Backend compile does not contain op-specific switch. It uses generated backend op registry.

---

## 17. Fusions

Fusions remain backend-local.

Examples:

- Metal may fuse `silu + mul` into `gd_silu_mul`.
- CUDA may fuse different patterns.
- Vulkan may skip fusion initially.

Fusion code lives in backend runtime:

```text
src/backends/metal/fusion.m
src/backends/cuda/fusion.cu
```

Fused kernels may live in op dir if tied to one op, or backend runtime if truly cross-op:

```text
src/backends/metal/fused/metal_silu_mul.metal
```

Core op registry should not contain backend-only fused implementation details unless fusion becomes public semantic IR.

---

## 18. C-only codegen

### 18.1 No Python

Generator is a C program:

```text
tools/gen_ops.c
```

It scans file paths passed by Makefile. It does not parse C contents.

### 18.2 Inputs

Makefile discovers:

```make
OP_FILES := $(sort $(wildcard src/ops/*/core_*.c))
GRAD_FILES := $(sort $(wildcard src/ops/*/grad_*.c))
CPU_OP_FILES := $(sort $(wildcard src/ops/*/cpu_*.c))
METAL_OP_FILES := $(sort $(wildcard src/ops/*/metal_*.m))
METAL_SHADER_FILES := $(sort $(wildcard src/ops/*/metal_*.metal))
```

Run:

```make
$(BUILD_DIR)/tools/gen_ops: tools/gen_ops.c
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: generated
generated: $(BUILD_DIR)/tools/gen_ops
	@mkdir -p $(BUILD_DIR)/generated
	@$< \
	  --out $(BUILD_DIR)/generated \
	  --core $(OP_FILES) \
	  --grad $(GRAD_FILES) \
	  --cpu $(CPU_OP_FILES) \
	  --metal $(METAL_OP_FILES) \
	  --metal-shaders $(METAL_SHADER_FILES)
```

### 18.3 Outputs

```text
build/generated/op_kind.h
build/generated/op_registry.inc
build/generated/bwd_registry.inc
build/generated/cpu_registry.inc
build/generated/metal_registry.inc
build/generated/metal_shaders.mk
build/generated/op_matrix.md
```

### 18.4 Generated public umbrella

Optional later:

```text
build/generated/include/gradients/ops.h
```

For first implementation pass, keep current `include/gradients/ops.h` manually to reduce risk.

---

## 19. Build integration

### 19.1 Source discovery

Current source discovery can stay broad:

```make
SRC := $(shell find $(SRC_DIR) -type f -name '*.c' 2>/dev/null | sort)
MSRC := $(shell find $(SRC_DIR) -type f -name '*.m' 2>/dev/null | sort)
```

Add dependency so generated headers exist before compile.

### 19.2 Include paths

Add:

```make
CPPFLAGS += -I$(BUILD_DIR)/generated
```

### 19.3 Metal shaders

Use generated shader list or wildcard:

```make
METAL_SHADERS := $(shell find $(SRC_DIR) -type f -name '*.metal' 2>/dev/null | sort)
```

All `.metal` files can compile to `.air` and link into one `gradients.metallib`.

---

## 20. File size policy

Add `make size-check`.

Policy:

- fail if `src/**/*.{c,h,m,metal,cu}` > 1000 LOC
- warn if > 800 LOC
- generated files under `build/generated/` exempt
- tests/docs can have separate soft limits

Example shell implementation:

```bash
find src include -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' -o -name '*.cu' \) \
  | while read -r f; do
      n=$(wc -l < "$f")
      if [ "$n" -gt 1000 ]; then
        echo "size-check: FAIL $f has $n lines"
        fail=1
      elif [ "$n" -gt 800 ]; then
        echo "size-check: WARN $f has $n lines"
      fi
    done
exit ${fail:-0}
```

`make check` should run `size-check`.

---

## 21. Add-op workflow after refactor

Add `layer_norm`:

```text
src/ops/layer_norm/core_layer_norm_fwd.c
src/ops/layer_norm/grad_layer_norm.c
src/ops/layer_norm/core_layer_norm_bwd.c       # if using internal bwd op
src/ops/layer_norm/cpu_layer_norm_fwd.c
src/ops/layer_norm/cpu_layer_norm_bwd.c
src/ops/layer_norm/metal_layer_norm_fwd.m      # optional
src/ops/layer_norm/metal_layer_norm_fwd.metal  # optional
include/gradients/ops/layer_norm.h             # optional later split
```

No central edits for:

- op enum
- op name switch
- autograd switch
- CPU run switch
- Metal kernel list
- Metal encode switch

Generator picks files up. Missing expected symbols fail at link.

---

## 22. Backend bring-up workflow

Add CUDA backend gradually:

```text
src/backends/cuda/backend.cu
src/backends/cuda/runtime.cu
src/backends/cuda/storage.cu
src/backends/cuda/cuda_op.h
src/ops/add/cuda_add_fwd.cu
src/ops/scale/cuda_scale_fwd.cu
```

CUDA starts with a tiny support table generated from files present. Unsupported ops fail with clear reason. No core changes required.

Same for Vulkan.

---

## 23. Tests

### 23.1 Registry tests

```text
tests/test_op_registry.c
```

Checks:

- every op kind has op def
- every op name non-empty
- arity sane
- output count sane
- meta fn non-null
- generated `_GD_OP_COUNT` matches discovered core files

### 23.2 Backend coverage tests

```text
tests/test_backend_registry.c
```

Checks:

- CPU_REF has implementations for all ops required by CPU tests
- Metal op table has kernel names resolvable when metallib loaded
- unsupported reason is non-empty

### 23.3 Autograd rule tests

```text
tests/test_autograd_registry.c
```

Checks:

- every `GD_OPF_DIFF` public op has grad rule or explicit unsupported reason
- internal backward ops do not accidentally require grad rule

### 23.4 Public symbol link test

Header compile test is not enough. Add test that references every public symbol from installed public headers so missing implementation fails at link.

This catches phantom API declarations.

---

## 24. Step-by-step implementation plan

Status key:

- [ ] not started
- [~] in progress
- [x] done

Current codebase inventory used to ground this plan:

- [x] CPU-only baseline passes: `make test GD_ENABLE_METAL=0`.
- [x] Current op logic touchpoints identified: `src/graph/graph_internal.h`, `src/graph/graph.c`, `src/ops/op_schema.c`, `src/ops/shape.c`, `src/autograd/autograd.c`, `src/backends/cpu_ref/cpu_ref.c`, `src/backends/cpu_ref/cpu_kernels.c`, `src/backends/metal/metal_backend.m`, `src/backends/metal/kernels.metal`, `include/gradients/ops.h`.
- [x] Current oversized source files identified: `src/backends/metal/metal_backend.m` (~3331 LOC), `src/backends/metal/kernels.metal` (~2459), `src/tokenizer/tokenizer.c` (~2032), `src/backends/cpu_ref/cpu_kernels.c` (~1368), `src/graph/graph.c` (~1268), `src/dataset/dataset.c` (~1125), `src/core/tensor.c` (~1123).

### 24.1 Add guardrails before moving code

- [x] Add `make size-check-report` that prints all files over 800/1000 LOC without failing.
- [x] Add `docs/size_allowlist.txt` containing current oversized files only, with owner/reason/removal target for each.
- [x] Add `make size-check` that fails for any new non-allowlisted file over 1000 LOC and warns over 800 LOC.
- [x] Add `size-check` to `make check` after allowlist exists.
- [x] Add `build/generated/` include path support: `CPPFLAGS += -I$(BUILD_DIR)/generated`.
- [x] Add placeholder generated dir creation before compile so clean builds do not race generated includes.
- [x] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.2 Add C-only generator skeleton

- [x] Create `tools/gen_ops.c`.
- [x] Implement filename scanning from argv only; do not parse C source.
- [x] Implement snake_case to UPPER_CASE conversion.
- [x] Implement role mapping: `_fwd` => base op kind, other roles append suffix.
- [x] Emit empty-safe files under `build/generated/`: `op_kind.h`, `op_registry.inc`, `bwd_registry.inc`, `cpu_registry.inc`, `metal_registry.inc`, `op_matrix.md`.
- [x] Wire Makefile discovery for `src/ops/*/core_*.c`, `grad_*.c`, `cpu_*.c`, `metal_*.m`, `metal_*.metal`.
- [x] Add generator target as dependency of library objects.
- [x] Validate empty/no-capsule generated files do not affect current build.
- [x] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.3 Introduce core op registry without changing execution

- [x] Add `src/ops/op_impl.h` with `_gd_op_def`, `_gd_meta_fn`, op flags, and `_gd_emit_checked` declaration.
- [x] Create core capsule files for every current `_GD_OP_*` in `src/graph/graph_internal.h`.
- [x] Include current public fwd ops: `add`, `mul`, `scale`, `matmul`, `linear`, `relu`, `silu`, `powlu`, `sum`, `mean`, `rms_norm`, `softmax`, `cross_entropy`, `lm_cross_entropy`, `cast`, `gelu`, `transpose`, `embedding`, `rope`, `sdpa`, `clip_grad_norm`.
- [x] Include current internal/support ops: `copy`, `relu_bwd`, `silu_bwd`, `powlu_bwd`, `softmax_bwd`, `sum_bwd`, `mean_bwd`, `cross_entropy_bwd`, `lm_cross_entropy_bwd`, `gelu_bwd`, `embedding_bwd`, `rope_bwd`, `sdpa_bwd`, `rms_norm_bwd`, `rms_norm_wbwd`, `step_inc`, `adamw_step`, `reduce_to`, `assert_finite`, `assert_close`.
- [x] Audit current reserved/legacy enum values: `_GD_OP_BACKWARD`, `_GD_OP_ZERO_GRAD`, `_GD_OP_OPTIMIZER_STEP`; either create explicit reserved core defs or remove them if no code emits them.
- [x] Generate `_GD_OP_*` enum from core files into `build/generated/op_kind.h`.
- [x] Replace manual enum in `src/graph/graph_internal.h` with generated enum include.
- [x] Generate op registry extern table into `op_registry.inc`.
- [x] Replace `_gd_op_kind_name()` switch in `src/graph/graph.c` with registry lookup.
- [x] Add `tests/test_op_registry.c` for op def presence, names, arity, output count, and enum count.
- [x] Validate generated names match current graph dump/test expectations.
- [x] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.4 Add checked emit and move meta/public wrappers into capsules

- [x] Implement `_gd_emit_checked()` using active graph lookup, op registry arity checks, input desc collection, meta call, and graph emit.
- [x] Move generic shape helpers from `src/ops/shape.c` into `src/ops/meta_common.c` / `src/ops/meta_common.h`.
- [x] Migrate simple public wrappers from `src/ops/op_schema.c` into capsules: `add`, `mul`, `scale`, `relu`, `silu`, `gelu`.
- [x] Migrate linalg wrappers/meta: `matmul`, `linear`.
- [x] Migrate reductions/meta: `sum`, `mean`, `softmax`, `reduce_to`.
- [x] Migrate norm/loss/embedding/rope/attention wrappers/meta: `rms_norm`, `cross_entropy`, `lm_cross_entropy`, `embedding`, `rope`, `sdpa`.
- [x] Migrate misc wrappers/meta: `powlu`, `cast`, `transpose`, `assert_finite`, `assert_close`.
- [x] Keep `include/gradients/ops.h` manual for first pass; only move implementations.
- [x] Delete or shrink `src/ops/op_schema.c` after all public wrappers move.
- [x] Delete or shrink `src/ops/shape.c` after all meta code moves.
- [x] Validate: `tests/test_ops`, `tests/test_graph`, `tests/test_debug` pass under CPU.
- [x] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.5 Move autograd rules into op capsules

- [x] Add `src/autograd/autograd_internal.h` exposing `_gd_bwd_ctx` helpers for rule files.
- [x] Add `_gd_bwd_rule` type and generated `bwd_registry.inc` lookup.
- [x] Move elementwise rules: `add`, `mul`, `scale`.
- [x] Move activation rules: `relu`, `silu`, `powlu`, `gelu`.
- [x] Move linalg rules: `matmul`, `linear`.
- [x] Move reduction rules: `sum`, `mean`, `softmax`.
- [x] Move loss rules: `cross_entropy`, `lm_cross_entropy`.
- [x] Move layout/index/attention/norm rules: `copy`, `transpose`, `embedding`, `rope`, `rms_norm`, `sdpa`.
- [x] Add explicit unsupported rule entry for `cast` with current message `cast backward is not supported in v1`.
- [x] Replace `backward_node()` switch in `src/autograd/autograd.c` with registry dispatch.
- [x] Keep engine responsibilities in `autograd.c`: traversal, seed grad, accumulation, leaf grad writes, cleanup.
- [x] Add `tests/test_autograd_registry.c` to require grad rules for every `GD_OPF_DIFF` public op or explicit unsupported reason.
- [x] Validate: `tests/test_autograd` and `tests/test_metal_gpt_train` pass under CPU.
- [x] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.6 Move CPU_REF op implementations into capsules

- [ ] Add `src/backends/cpu_ref/cpu_op.h` with `_gd_cpu_op`, support fn, run fn, and `_gd_cpu_exec` access helpers.
- [ ] Generate `cpu_registry.inc` from `cpu_<op>_<role>.c` files.
- [ ] Replace `cpu_run_node()` switch in `src/backends/cpu_ref/cpu_ref.c` with CPU op lookup.
- [ ] Move simple CPU ops into capsules: `add`, `mul`, `scale`, `relu`, `silu`, `gelu`, `copy`, `cast`.
- [ ] Move linalg/reduction/norm/loss/index/attention CPU ops into capsules.
- [ ] Move internal backward CPU ops into capsules.
- [ ] Split `src/backends/cpu_ref/cpu_kernels.c` into shared helpers plus per-op `cpu_*` files until file is under 1000 LOC or removed.
- [ ] Keep CPU_REF support complete for all graph ops used by existing tests.
- [ ] Add `tests/test_backend_registry.c` coverage for CPU_REF missing-op errors.
- [ ] Validate: `make test GD_ENABLE_METAL=0` still passes.

### 24.7 Split Metal runtime before moving Metal ops

- [ ] Split `src/backends/metal/metal_backend.m` into runtime files: `backend.m`, `runtime.m`, `storage.m`, `compile.m`, `execute.m`, `fusion.m`, plus shared headers.
- [ ] Keep behavior identical: same vtable, same env vars (`GRADIENTS_METALLIB`, `GD_METAL_MPS`), same fallback behavior.
- [ ] Move pipeline cache/loading helpers into Metal runtime, still using old central kernel table temporarily.
- [ ] Move staging/writeback/synchronize logic into runtime files.
- [ ] Move MPS GEMM planning helpers into either Metal runtime or linalg op capsule, whichever keeps files smaller.
- [ ] Ensure each new Metal runtime source file is under 1000 LOC.
- [ ] Validate: `make test` on macOS with Metal enabled passes or skips exactly as before.

### 24.8 Move Metal op host entries into capsules

- [ ] Add `src/backends/metal/metal_op.h` with `_gd_metal_op`, support fn, plan fn, encode fn, plan ctx, encode ctx.
- [ ] Generate `metal_registry.inc` from `metal_<op>_<role>.m` files.
- [ ] Replace Metal `supports_node` bool-style path with `gd_status` support checks that include concrete unsupported reason.
- [ ] Move simple host encoders into capsules: elementwise/unary/copy/cast.
- [ ] Move linalg host encoders: `matmul`, `linear`, MPS optional paths.
- [ ] Move reduction/norm/loss/embedding/rope host encoders.
- [ ] Move `sdpa` and `sdpa_bwd` host planning/encoding into their own op capsule files; split further inside `src/ops/sdpa/` if near 800 LOC.
- [ ] Move optimizer/debug host encoders: `step_inc`, `adamw_step`, `assert_finite`, `assert_close`.
- [ ] Delete old central `encode_node()` switch after all host entries move.
- [ ] Validate Metal fallback and unsupported messages with `tests/test_fallback` and Metal tests.

### 24.9 Split Metal shaders into op capsules

- [ ] Add `src/backends/metal/metal_common.metal` for shared shader helpers/types includes.
- [ ] Move simple kernels from `src/backends/metal/kernels.metal` into `src/ops/<op>/metal_<op>_<role>.metal`.
- [ ] Move linalg/reduction/norm/loss/embedding/rope shaders into op capsule shader files.
- [ ] Move SDPA fwd/bwd shaders into separate files under `src/ops/sdpa/`; split large fwd/bwd pieces if any file approaches 800 LOC.
- [ ] Update Makefile to compile all `src/**/*.metal` files into one `gradients.metallib`.
- [ ] Ensure `metal_kernel_types.h` remains shared ABI header and stays under size limit.
- [ ] Delete or shrink old `src/backends/metal/kernels.metal`.
- [ ] Validate: `make test` on macOS with Metal enabled passes.

### 24.10 Public headers and link coverage

- [ ] Add public symbol link test that references every function declared in `include/gradients/*.h`.
- [ ] Confirm current public headers with implementations: `ops.h`, `tokenizer.h`, `dataset.h`, `dataloader.h`, `optim.h`, `nn.h`, etc.
- [ ] Decide whether to split `include/gradients/ops.h` into `include/gradients/ops/<op>.h` plus generated umbrella.
- [ ] If splitting public op headers, generate umbrella under `build/generated/include/gradients/ops.h` or keep stable manual umbrella until install story exists.
- [ ] Validate header compile and public symbol link test in `make test`.

### 24.11 Split remaining non-op oversized files

- [ ] Split `src/tokenizer/tokenizer.c` into training, model IO, encode, decode, and common helpers under `src/tokenizer/`.
- [ ] Split `src/dataset/dataset.c` into dataset core, text/token storage, batching/index helpers, and IO helpers.
- [ ] Split `src/core/tensor.c` into descriptor/view logic, materialization/copy, grad metadata, and lifecycle/refcount glue.
- [ ] Split `src/graph/graph.c` into graph lifecycle/capture, emit/import, compile/run, debug/materialize helpers.
- [ ] Remove each file from `docs/size_allowlist.txt` once under 1000 LOC.
- [ ] Validate: `make test GD_ENABLE_METAL=0` still passes after each split.

### 24.12 Remove compatibility shims and enforce final policy

- [ ] Delete empty legacy files: old `op_schema.c`, old `shape.c`, old CPU/Metal central switches, old kernel monoliths.
- [ ] Remove all oversized source files from `docs/size_allowlist.txt`.
- [ ] Make `size-check` hard-fail all source files over 1000 LOC with no allowlist exceptions.
- [ ] Regenerate `build/generated/op_matrix.md` and verify it lists core/grad/CPU/Metal coverage by op.
- [ ] Run full CPU validation: `make clean && make test GD_ENABLE_METAL=0`.
- [ ] Run full macOS/Metal validation: `make clean && make test`.
- [ ] Run examples/bench smoke checks: `make mlp`, `make gpt`, and relevant GPT bench target.

### 24.13 Add CUDA/Vulkan skeletons after registry is stable

- [ ] Add CUDA backend runtime skeleton with empty generated CUDA op registry.
- [ ] Add `src/ops/add/cuda_add_fwd.cu` or `src/ops/scale/cuda_scale_fwd.cu` as first CUDA op.
- [ ] Verify CUDA unsupported-op path includes op name, dtype/layout/shape reason.
- [ ] Add Vulkan backend runtime skeleton with empty generated Vulkan op registry.
- [ ] Add `src/ops/add/vulkan_add_fwd.c` or `src/ops/scale/vulkan_scale_fwd.c` as first Vulkan op.
- [ ] Verify Vulkan unsupported-op path includes op name, dtype/layout/shape reason.

---

## 25. Risks and mitigations

### Risk: too many tiny files

Mitigation:

- predictable names
- generated `op_matrix.md` index
- docs explain lookup flow
- agent-friendly: one op dir contains relevant logic

### Risk: generator bugs

Mitigation:

- generator only parses filenames
- generated files checked by tests
- missing symbols fail at link

### Risk: staged refactor churn

Mitigation:

- move one op at a time
- keep old path until replacement path is complete
- preserve behavior; no perf work during structural moves

### Risk: shared helpers become huge

Mitigation:

- shared helpers must stay generic
- if helper becomes op-specific, move into op dir
- size-check catches growth

---

## 26. Final architecture summary

```text
src/ops/<op>/core_*      defines what op means and its meta inference
src/ops/<op>/grad_*      defines backend-neutral derivative rule
src/ops/<op>/cpu_*       defines CPU_REF implementation
src/ops/<op>/metal_*     defines Metal host/shader implementation
src/backends/<backend>/  defines runtime/storage/scheduling only
tools/gen_ops.c          wires files together from naming convention
```

Core rule:

```text
Operator owns semantics and implementations.
Backend runtime owns execution machinery.
Codegen owns boilerplate.
```

This keeps source small, touchpoints local, and future CUDA/Vulkan work tractable for both humans and coding agents.
