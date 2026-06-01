# gradients.c Design Spec

Status: draft v0.1

Goal: high-performance neural-net training library in C with multiple backends, multiple dtypes, quantized formats, autograd, and graph-level fusion.

Non-goal: make CPU fastest first. CPU is correctness/reference backend. GPU backends own peak performance.

---

## 1. Design pillars

1. **C public ABI, private internals.** Public API is plain C with opaque handles where possible. Private API may change freely. Anything under `src/` or prefixed `_gd_` is internal.
2. **Tensor/Storage separation.** Tensor describes logical data/view metadata. Storage owns bytes on a device. Execution does not live in tensor.
3. **Dtype + device + layout + quant are first-class dispatch keys.** Backend selection depends on scalar type, physical placement, memory layout, and quant format.
4. **Quant formats are static storage contracts.** Low-bit formats are opaque packed storage with static traits. Optimized kernels specialize by quant format and layout.
5. **Typed graph IR is core execution model.** All ops lower through IR. Immediate debugging is one-shot graph build/compile/run, not a separate execution engine.
6. **Backend compiler owns fusion, layout selection, and memory planning.** Metal/CUDA/Vulkan lower graph segments to fused kernels, choose physical layouts, plan workspaces, and cache/autotune kernels.
7. **CPU reference is oracle, not hidden production fallback.** Every op has simple CPU reference coverage for correctness, tests, debugging, and backend parity. Production fails loudly when required optimized kernels are missing unless fallback policy explicitly allows CPU reference.
8. **Autograd is IR-aware and backend-independent.** Grad metadata is optional. Backward graph can be optimized, fused, recomputed, or lowered by backend.
9. **No hot-loop virtual decode.** Quant callbacks only for packing/unpacking reference paths; optimized kernels know format at compile/JIT time.
10. **Graph debugging is first-class.** IR must be inspectable, partially executable, comparable across backends, and compilable with passes disabled. Gradient bugs must be localizable to the first bad node/op.
11. **Async/multi-device-ready design.** API must not block future streams/events, overlapping transfer/compute, collectives, tensor parallelism, or sharded optimizers.
12. **Error returns over asserts.** Public API returns `gd_status`. Assertions only for internal impossible states.

---

## 2. Project layout

```text
include/gradients/        Public headers
  gradients.h             Umbrella include
  status.h                Error/status codes
  dtype.h                 Dtypes and dtype helpers
  device.h                Device/backend descriptors
  tensor.h                Tensor/storage public API
  quant.h                 Quant format descriptors
  graph.h                 Graph/autograd public API
  ops.h                   Public tensor ops
  module.h                Module/parameter API
  optim.h                 Optimizers

src/                      Private implementation
  core/                   Tensor/storage/allocator/device internals
  graph/                  IR, autograd engine, graph passes
  ops/                    Op schemas and CPU reference kernels
  backends/               Backend implementations
    cpu_ref/
    metal/
    cuda/
    vulkan/
  nn/                     Optional higher-level layers/models

tests/                    Unit, gradcheck, backend parity, fuzz
bench/                    Backend/operator/model benchmarks
docs/                    Design docs
```

---

## 3. Naming and visibility

### Public symbols

Public functions/types use `gd_` prefix.

```c
gd_status gd_tensor_create(...);
gd_status gd_add(...);
typedef struct gd_tensor gd_tensor;
```

Public headers live under `include/gradients/`.

### Private symbols

Private functions/types use `_gd_` prefix or internal-only headers.

```c
_gd_tensor_impl;
_gd_dispatch_lookup(...);
_gd_graph_lower(...);
```

Private headers live under `src/**`. They are not installed.

### Handle ownership convention

Unless an API explicitly says otherwise:

- `*_create`, `*_empty`, `*_from_storage`, and op output parameters return owned handles with refcount 1.
- Query/accessor functions return borrowed handles/pointers that are valid only while the owner remains alive.
- Borrowed handles must be retained by the caller if stored beyond the owner's lifetime.

### Opaque public structs

Prefer opaque handles for ABI stability:

```c
typedef struct gd_context gd_context;
typedef struct gd_tensor gd_tensor;
typedef struct gd_graph gd_graph;
typedef struct gd_storage gd_storage;
typedef struct gd_module gd_module;
```

Small value structs may be public:

```c
typedef struct gd_shape_view { const int64_t *dims; int ndim; } gd_shape_view;
```

---

## 4. Status and error model

Public functions return `gd_status` unless returning a simple query value or acting as an unconditional destructor/release function.

```c
typedef enum gd_status {
    GD_OK = 0,
    GD_ERR_INVALID_ARGUMENT,
    GD_ERR_OUT_OF_MEMORY,
    GD_ERR_UNSUPPORTED,
    GD_ERR_BACKEND,
    GD_ERR_DTYPE,
    GD_ERR_SHAPE,
    GD_ERR_DEVICE,
    GD_ERR_INVALID_STATE,
    GD_ERR_IO,
    GD_ERR_INTERNAL
} gd_status;
```

Thread-local last error stores message:

```c
const char *gd_last_error(void);
```

Rules:

- Public API validates inputs and returns status.
- Private API may assert invariants.
- Backend kernels propagate backend-specific failures through `GD_ERR_BACKEND`.

---

## 5. Context

`gd_context` owns global library state: registered backends, allocators, default device, default compute policy, active graph, fallback policy, RNG, logging, kernel cache.

```c
typedef struct gd_context gd_context;

gd_status gd_context_create(gd_context **out);
void      gd_context_destroy(gd_context *ctx);

gd_status gd_context_set_default_device(gd_context *ctx, gd_device device);
gd_device gd_context_default_device(const gd_context *ctx);

gd_status gd_context_set_compute_policy(gd_context *ctx,
                                        gd_compute_policy policy);
gd_compute_policy gd_context_compute_policy(const gd_context *ctx);
```

Fallback policy controls whether unsupported optimized kernels may use CPU reference execution:

```c
typedef enum gd_fallback_policy {
    GD_FALLBACK_NONE,        // fail if no exact/supported kernel exists
    GD_FALLBACK_CPU_REF      // allow CPU reference execution (debug/tests)
} gd_fallback_policy;

gd_status gd_context_set_fallback_policy(gd_context *ctx,
                                         gd_fallback_policy policy);
gd_fallback_policy gd_context_fallback_policy(const gd_context *ctx);
```

Default:

- `gd_context_create` starts with `GD_FALLBACK_NONE`
- tests/debug tools explicitly set `GD_FALLBACK_CPU_REF`

Design choices:

- No hidden global context for training. Explicit context makes multi-device and testing easier.
- `gd_context` is the root object and must outlive all tensors, storages, graphs, modules, optimizers, and quant descriptors created from it in v1.
- Fallbacks exist for correctness and development, not as invisible production runtime behavior.

---

## 6. Dtypes

Dtype means scalar interpretation for ordinary tensors. Quantized packed formats use `GD_DTYPE_QUANTIZED` plus `gd_quant_desc`.

```c
typedef enum gd_dtype {
    GD_DTYPE_F32,
    GD_DTYPE_F16,
    GD_DTYPE_BF16,
    GD_DTYPE_FP8_E4M3,
    GD_DTYPE_FP8_E5M2,
    GD_DTYPE_I8,
    GD_DTYPE_U8,
    GD_DTYPE_I32,
    GD_DTYPE_I64,
    GD_DTYPE_BOOL,
    GD_DTYPE_QUANTIZED
} gd_dtype;

size_t      gd_dtype_sizeof(gd_dtype dt);      // invalid for QUANTIZED
const char *gd_dtype_name(gd_dtype dt);
```

Execution can use separate precision policy:

```c
typedef struct gd_compute_policy {
    gd_dtype compute_dtype;   // e.g. F16/BF16/F32
    gd_dtype accum_dtype;     // e.g. F32 for reductions/matmul
} gd_compute_policy;

gd_compute_policy gd_compute_policy_default(void);
```

Design choice: storage dtype, compute dtype, and accumulation dtype are separate. This is mandatory for FP8/int8/int4 training.

---

## 7. Device and backend

```c
typedef enum gd_device_type {
    GD_DEVICE_CPU,
    GD_DEVICE_CUDA,
    GD_DEVICE_METAL,
    GD_DEVICE_VULKAN
} gd_device_type;

typedef struct gd_device {
    gd_device_type type;
    int index;
} gd_device;

gd_status gd_synchronize(gd_context *ctx, gd_device device);
```

Backend roles:

- `cpu_ref`: simple, complete, correctness backend.
- `cuda`: fused/JIT/specialized GPU backend.
- `metal`: fused/JIT/specialized Apple GPU backend.
- `vulkan`: fused/SPIR-V portable GPU backend.

Design choices:

- Backend is not just a kernel table. GPU backends receive graph segments and can fuse, plan memory, autotune, and cache kernels.
- Kernel implementation and compiler planning are separate: kernels are fast machines; IR/compiler decides which machines run, how tensors are laid out, what gets fused, and what never materializes.

---

## 8. Storage

Storage owns bytes on a device.

```c
typedef struct gd_storage gd_storage;

typedef enum gd_memory_kind {
    GD_MEM_HOST,
    GD_MEM_DEVICE,
    GD_MEM_PINNED_HOST,
    GD_MEM_UNIFIED
} gd_memory_kind;

typedef struct gd_storage_desc {
    gd_device device;
    gd_memory_kind memory;
    size_t nbytes;
    size_t alignment;       // 0 = backend default
} gd_storage_desc;

gd_status gd_storage_create(gd_context *ctx,
                            const gd_storage_desc *desc,
                            gd_storage **out);

gd_status gd_storage_retain(gd_storage *s);
void      gd_storage_release(gd_storage *s);

gd_status gd_storage_data_cpu(gd_storage *s, void **out);  // CPU-accessible storage only
gd_status gd_storage_copy_from_cpu(gd_context *ctx, gd_storage *dst,
                                   size_t dst_offset, const void *src,
                                   size_t nbytes);
gd_status gd_storage_copy_to_cpu(gd_context *ctx, gd_storage *src,
                                 size_t src_offset, void *dst,
                                 size_t nbytes);
size_t    gd_storage_nbytes(const gd_storage *s);
gd_device gd_storage_device(const gd_storage *s);
```

Design choices:

- Tensors can share storage via views. Storage refcount is independent of tensor lifetime.
- Host copy helpers are blocking in v1: when they return `GD_OK`, bytes are visible at the destination. Async copies can be added later via streams/events without changing storage ownership rules.

---

## 9. Memory management

Memory is managed in three layers.

### 9.1 Public handle lifetime

Public objects use explicit retain/release or destroy APIs:

```text
gd_tensor
gd_storage
gd_graph
gd_module
gd_optimizer
gd_quant_desc
```

Rules:

- Returned handles have refcount 1 unless documented otherwise.
- Tensors retain their storage if materialized.
- Virtual graph tensors retain graph/value metadata and are valid until their graph is reset/destroyed.
- A graph cannot be reset or destroyed while live virtual tensors from that graph exist; release or materialize those tensors first.
- Views retain the same underlying storage as their base tensor when materialized.
- `gd_quant_desc` is refcounted and immutable; it retains scale/zero/codebook tensors.
- Graph/compiler internals use private arenas, not public lifetime rules.

### 9.2 Storage/device allocations

`gd_storage` owns actual bytes on a device. Backend allocators provide the physical memory:

```text
CPU:    aligned host allocation
Metal:  MTLBuffer or backend-managed heap allocation
CUDA:   cudaMalloc/cudaMemPool allocation
Vulkan: VkBuffer/VMA allocation
```

`gd_memory_kind` describes placement. Legal device/kind combinations are backend-defined and validated by `gd_storage_create`. Storage free must be async-safe. If released while backend work may still reference it, backend defers physical free until the relevant command buffer/event/fence completes.

### 9.3 Graph runtime memory planner

Compiled graphs own a memory plan for intermediates and workspaces.

Compiler responsibilities:

1. infer sizes/alignments for intermediate values
2. compute tensor lifetimes
3. reuse buffers whose lifetimes do not overlap
4. assign backend-preferred layouts
5. allocate persistent workspace for graph execution
6. avoid materializing virtual tensors when possible

Graph run should be allocation-free after compile/warmup. Backend autotune may allocate during compile or first warmup run, but not every training step.

Graph memory classes:

```text
inputs/params: user-owned storages
outputs: returned tensors or user-provided out tensors
intermediates: graph-planned workspace slots
temps: backend scratch/workspace
virtual values: IR values with no physical storage until materialized
```

Quantized memory model:

```text
quant tensor storage: packed bytes only
quant desc: format + group metadata + retained scales/zeros/codebook tensors
```

Design choices:

- Public API lifetime is safe and explicit via refcounts/destructors.
- Private graph/compiler temporaries use arenas for speed.
- Device allocations are routed through backend allocators and may use pools.
- Compiled graph execution should not call malloc/cudaMalloc/newBuffer per op.
- Memory planner, not individual ops, owns intermediate allocation and reuse.

---

## 10. Tensor public API

Tensor is metadata over storage: logical shape, strides, dtype, layout, optional quant desc, optional autograd metadata.

```c
typedef struct gd_tensor gd_tensor;

#define GD_MAX_DIMS 8

typedef enum gd_layout {
    GD_LAYOUT_STRIDED,
    GD_LAYOUT_CONTIGUOUS,
    GD_LAYOUT_CHANNELS_LAST,
    GD_LAYOUT_PACKED_QUANT,
    GD_LAYOUT_BLOCKED,
    GD_LAYOUT_BACKEND_OPAQUE
} gd_layout;

typedef struct gd_tensor_desc {
    gd_dtype dtype;
    gd_device device;
    gd_layout layout;
    int ndim;
    int64_t sizes[GD_MAX_DIMS];
    int64_t strides[GD_MAX_DIMS];       // ignored/format-specific for packed layouts
    int64_t storage_offset_bytes;
    const struct gd_quant_desc *quant;  // nullable
} gd_tensor_desc;

gd_status gd_tensor_desc_contiguous(gd_dtype dtype,
                                    gd_device device,
                                    int ndim,
                                    const int64_t *sizes,
                                    gd_tensor_desc *out);

gd_status gd_tensor_desc_nbytes(const gd_tensor_desc *desc,
                                size_t *nbytes_out,
                                size_t *alignment_out);
```

Creation:

```c
gd_status gd_tensor_empty(gd_context *ctx,
                          const gd_tensor_desc *desc,
                          gd_tensor **out);

gd_status gd_tensor_from_storage(gd_context *ctx,
                                 gd_storage *storage,
                                 const gd_tensor_desc *desc,
                                 gd_tensor **out);

gd_status gd_tensor_retain(gd_tensor *t);
void      gd_tensor_release(gd_tensor *t);
```

Host transfer helpers:

```c
gd_status gd_tensor_copy_from_cpu(gd_context *ctx, gd_tensor *dst,
                                  const void *src, size_t nbytes);
gd_status gd_tensor_copy_to_cpu(gd_context *ctx, gd_tensor *src,
                                void *dst, size_t nbytes);
```

These are explicit transfers/materializations; they are not compute fallbacks. `gd_tensor_copy_from_cpu` requires `dst` to be materialized. `gd_tensor_copy_to_cpu` materializes virtual `src` if needed. Both helpers copy raw tensor storage bytes only: caller buffers must already be encoded in the tensor dtype/layout (for example binary16 bytes for `GD_DTYPE_F16`). Use `gd_cast` for numeric dtype conversion.

Queries:

```c
int              gd_tensor_ndim(const gd_tensor *t);
int64_t          gd_tensor_size(const gd_tensor *t, int dim);
int64_t          gd_tensor_stride(const gd_tensor *t, int dim);
gd_dtype         gd_tensor_dtype(const gd_tensor *t);
gd_device        gd_tensor_device(const gd_tensor *t);
gd_layout        gd_tensor_layout(const gd_tensor *t);
gd_storage      *gd_tensor_storage(const gd_tensor *t);  // borrowed; NULL for virtual tensors
const gd_quant_desc *gd_tensor_quant(const gd_tensor *t); // borrowed
```

Views:

```c
gd_status gd_tensor_view(gd_tensor *base,
                         const gd_tensor_desc *view_desc,
                         gd_tensor **out);

gd_status gd_tensor_reshape(gd_tensor *t, int ndim, const int64_t *sizes, gd_tensor **out);
gd_status gd_tensor_transpose(gd_tensor *t, int d0, int d1, gd_tensor **out);
gd_status gd_tensor_slice(gd_tensor *t, int dim, int64_t start, int64_t len, gd_tensor **out);
gd_status gd_tensor_contiguous(gd_context *ctx, gd_tensor *t, gd_tensor **out);
```

Design choices:

- `storage_offset_bytes` is bytes, not elements. Packed 4-bit/1.5-bit formats cannot use element offsets safely.
- Ordinary strided tensors use `sizes + strides` in elements.
- `gd_tensor_desc_contiguous` fills standard row-major strides and layout.
- `gd_tensor_desc_nbytes` is the single public helper for sizing storage/tensor allocations, including packed quant formats.
- Packed quant tensors use logical `sizes`; physical indexing belongs to quant format/layout.
- `reshape`, `transpose`, and `slice` are metadata/view operations when representable without copy.
- `gd_tensor_contiguous` is a copy-producing operation when input is not already contiguous; it records an IR node and therefore requires an active graph or one-shot helper.
- Tensor does not inline `grad`, `grad_fn`, or allocator fields.

---

## 11. Quantization public API

Quantization is a storage contract plus static traits.

```c
typedef struct gd_quant_format gd_quant_format;
typedef struct gd_quant_desc gd_quant_desc;

typedef gd_status (*gd_quant_pack_ref_fn)(const gd_tensor *src,
                                           const gd_quant_desc *qdesc,
                                           gd_tensor *dst_packed);
typedef gd_status (*gd_quant_unpack_ref_fn)(const gd_tensor *src_packed,
                                             gd_dtype out_dtype,
                                             gd_tensor *dst);

struct gd_quant_format {
    const char *name;          // "nf4", "i4", "fp8_e4m3", "ternary15"
    int bits_num;              // 3 for 1.5 bits
    int bits_den;              // 2 for 1.5 bits
    int values_per_block;      // logical values decoded per packed block
    int bytes_per_block;       // physical packed bytes per block
    gd_layout preferred_layout;

    /* Reference path only. Optimized kernels do not call these in hot loops. */
    gd_quant_pack_ref_fn   pack_ref;
    gd_quant_unpack_ref_fn unpack_ref;
};

struct gd_quant_desc {
    const gd_quant_format *format;
    int group_size;
    int axis;
    gd_dtype scale_dtype;
    gd_tensor *scales;         // nullable for codebook-only formats
    gd_tensor *zeros;          // nullable
    const void *extra;         // optional small immutable metadata copied into desc
    size_t extra_size;
};
```

Registration:

```c
gd_status gd_quant_register_format(gd_context *ctx,
                                   const gd_quant_format *fmt);

const gd_quant_format *gd_quant_find_format(gd_context *ctx,
                                            const char *name);
```

Descriptor lifecycle:

```c
gd_status gd_quant_desc_create(gd_context *ctx,
                               const gd_quant_format *fmt,
                               int group_size,
                               int axis,
                               gd_dtype scale_dtype,
                               gd_tensor *scales,
                               gd_tensor *zeros,
                               const void *extra,
                               size_t extra_size,
                               gd_quant_desc **out);

gd_status gd_quant_desc_retain(gd_quant_desc *qd);
void      gd_quant_desc_release(gd_quant_desc *qd);
```

Packing/unpacking reference API:

```c
gd_status gd_quantize(gd_context *ctx,
                      gd_tensor *src,
                      const gd_quant_desc *qdesc,
                      gd_tensor **packed_out);

gd_status gd_dequantize(gd_context *ctx,
                        gd_tensor *packed,
                        gd_dtype out_dtype,
                        gd_tensor **out);
```

`gd_quantize` and `gd_dequantize` are compute ops: with an active graph they record IR nodes; outside a graph they must be called via one-shot immediate helpers.

Design choices:

- `gd_quant_register_format` copies the format descriptor into the context registry; caller-owned `fmt` may be temporary.
- `GD_DTYPE_QUANTIZED` says tensor is packed. `gd_quant_format` says how.
- A weird type like 1.5-bit is represented naturally: `bits_num=3`, `bits_den=2`, block packed.
- Generic `gd_dequantize` exists for correctness, debugging, checkpoint inspection, and explicit user calls.
- Optimized kernels dispatch on quant format + layout. They do not call decode callbacks in hot loops.
- If fallback policy is `GD_FALLBACK_CPU_REF`, CPU reference kernels may internally dequantize as needed. If policy is `GD_FALLBACK_NONE`, missing quant kernels return `GD_ERR_UNSUPPORTED`.

---

## 12. Ops public API

Public ops are functional by default: input tensors immutable, output tensor handle returned by the op. Outputs may be virtual until graph compile/run/materialization. All public ops record typed IR nodes. In immediate/debug helpers, the library builds and runs a one-shot graph internally; there is no separate immediate execution engine.

Ops are specified by schemas. Each schema defines input arity, attributes, shape inference, dtype rules, device/layout constraints, differentiability, and CPU reference behavior.

```c
gd_status gd_add(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_mul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_scale(gd_context *ctx, gd_tensor *x, float scale, gd_tensor **out);
typedef struct gd_matmul_desc {
    bool trans_a;
    bool trans_b;
    gd_compute_policy compute;
} gd_matmul_desc;

typedef struct gd_linear_desc {
    bool trans_w;              // enables tied LM head: hidden @ embed.weight^T
    gd_compute_policy compute;
} gd_linear_desc;

gd_status gd_matmul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_matmul_ex(gd_context *ctx, const gd_matmul_desc *desc,
                       gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_linear(gd_context *ctx,
                    gd_tensor *x,
                    gd_tensor *w,
                    gd_tensor *bias,
                    gd_tensor **out);
gd_status gd_linear_ex(gd_context *ctx,
                       const gd_linear_desc *desc,
                       gd_tensor *x,
                       gd_tensor *w,
                       gd_tensor *bias,
                       gd_tensor **out);

gd_status gd_relu(gd_context *ctx, gd_tensor *x, gd_tensor **out);
gd_status gd_silu(gd_context *ctx, gd_tensor *x, gd_tensor **out);
gd_status gd_sum(gd_context *ctx, gd_tensor *x, int dim, bool keepdim,
                 gd_tensor **out);
gd_status gd_mean(gd_context *ctx, gd_tensor *x, int dim, bool keepdim,
                  gd_tensor **out);
gd_status gd_rms_norm(gd_context *ctx, gd_tensor *x, gd_tensor *weight,
                      float eps, gd_tensor **out);
gd_status gd_softmax(gd_context *ctx, gd_tensor *x, int dim, gd_tensor **out);
gd_status gd_cross_entropy(gd_context *ctx, gd_tensor *logits, gd_tensor *targets,
                           int class_dim, gd_tensor **loss);
gd_status gd_lm_cross_entropy(gd_context *ctx, gd_tensor *hidden,
                              gd_tensor *weight, gd_tensor *targets,
                              gd_tensor **loss);

gd_status gd_cast(gd_context *ctx, gd_tensor *x, gd_dtype dtype,
                  gd_tensor **out);
```

Optional out variants bind graph outputs to user-provided storage:

```c
gd_status gd_add_out(gd_context *ctx, gd_tensor *out, gd_tensor *a, gd_tensor *b);
gd_status gd_matmul_out(gd_context *ctx, gd_tensor *out, gd_tensor *a, gd_tensor *b);
```

These still record IR nodes. `out` tensors must be materialized tensors with compatible dtype/device/shape. They constrain output allocation/aliasing instead of executing immediately.

`*_ex` descriptors may be NULL, meaning default flags and context default compute policy.

### 12.1 Op semantics v1

Shape rules:

- `gd_add` / `gd_mul` use NumPy-style broadcasting.
- `gd_scale` multiplies by a scalar attribute; scalar is not a tensor input.
- `gd_matmul` supports 2D and batched ND matmul with broadcast batch dims.
- `gd_linear(x,w,b)` is `matmul(x, w) + b`; `bias` may be NULL and otherwise broadcasts over leading dims.
- `gd_sum` / `gd_mean` reduce one dimension; full-tensor reduction can be represented by repeated reductions in v1.
- Negative `dim` values are accepted for reduction/softmax/cross-entropy and normalized against rank.
- `keepdim=true` preserves rank with reduced dim size 1.
- `gd_cross_entropy` returns scalar mean loss over all non-class positions.
- `gd_lm_cross_entropy(hidden, weight, targets)` computes scalar mean CE for `hidden[...,D] @ weight[V,D]^T` without requiring a materialized logits tensor; `targets` has shape `hidden.shape[:-1]`.

Dtype rules:

- v1 does not do implicit numeric promotion except where an op explicitly defines compute/accum dtype.
- Elementwise inputs must have the same dtype, except scalar attributes such as `gd_scale`.
- `gd_matmul` / `gd_linear` input and weight storage dtypes must either match or be a supported quantized-weight combination.
- `gd_compute_policy` controls math/accumulation dtype for matmul/linear/reductions/norms where supported.
- `gd_softmax` output dtype matches input dtype; compute may use policy/internal higher precision.
- `gd_cross_entropy` logits must be floating/quant-dequant-supported, targets must be `I32` or `I64`, loss dtype follows accum dtype.
- `gd_lm_cross_entropy` hidden/weight must be floating with matching dtype, targets must be `I32` or `I64`, and loss dtype follows hidden dtype.
- `gd_cast` is explicit; implicit casts are not inserted by v1 graph builder.

Device/layout rules:

- v1 compiled graphs are single-device. Materialized compute inputs in a graph must all have the same device.
- Virtual output device is inferred from input devices; source ops with no tensor inputs use the context default device.
- `gd_graph_compile(g, target)` validates that target matches the graph device inferred during construction.
- Cross-device copies are explicit host/device transfer operations, not implicit op behavior.
- Non-contiguous strided inputs are legal only if backend supports them for that op; otherwise compiler inserts a contiguous copy node when legal or returns `GD_ERR_UNSUPPORTED`.
- Quantized tensors are accepted only by quant-aware ops (`linear`, `matmul`, attention projections). Unsupported ops return `GD_ERR_UNSUPPORTED` unless fallback policy explicitly allows CPU reference execution.

Autograd rules:

- Floating tensors may require gradients.
- Integer, bool, and packed quantized tensors cannot require gradients in v1.
- Gradients accumulate into leaf grad slots using accum dtype policy.
- Op backward definitions are part of op schema; non-differentiable inputs are marked explicitly (e.g. CE targets).

Design choices:

- Public op API does not expose backend kernels directly.
- All backends consume the same op schemas through IR.
- Immediate debugging is syntactic sugar for one-shot graph build/compile/run, not a second execution path.
- Quantized tensors are accepted only by quant-aware ops (`linear`, `matmul`, attention projections). Unsupported ops return `GD_ERR_UNSUPPORTED` unless fallback policy explicitly allows CPU reference execution.
- Ops that need constants prefer scalar attributes for v1; scalar tensor literals can be added later if needed.

---

## 13. Graph execution model

gradients.c has one core execution model: typed graph IR.

- Public ops append IR nodes to the active graph.
- If no graph is active, public ops return `GD_ERR_INVALID_STATE` unless called through a documented one-shot immediate/debug helper.
- `gd_graph_begin` fails with `GD_ERR_INVALID_STATE` if another graph is already active in the context; nested graph capture is out of scope for v1.
- Parameter/data tensor creation and host transfers are allowed outside a graph; compute ops are not.
- Compiled graphs reference input/parameter tensor handles. Reusing a graph with new batch data means copying new bytes into the same input tensors, then calling `gd_graph_run` again.
- Materialization happens at graph compile/run or when tensor data is explicitly requested.
- CPU reference, Metal, CUDA, and Vulkan all consume the same graph/op schema.
- Immediate helpers are implemented as one-shot graph build/compile/run for debugging and tests.
- There is no separate immediate backend path.

Graph API:

```c
typedef struct gd_graph gd_graph;

gd_status gd_graph_create(gd_context *ctx, gd_graph **out);
/* destroy fails with GD_ERR_INVALID_STATE if live virtual tensors still reference g */
gd_status gd_graph_destroy(gd_graph *g);

gd_status gd_graph_begin(gd_context *ctx, gd_graph *g);
gd_status gd_graph_end(gd_context *ctx);

gd_status gd_graph_compile(gd_graph *g, gd_device target);
gd_status gd_graph_run(gd_graph *g);
/* reset fails with GD_ERR_INVALID_STATE if live virtual tensors still reference g */
gd_status gd_graph_reset(gd_graph *g);
```

One-shot immediate helper for tests/debug:

```c
typedef gd_status (*gd_immediate_build_fn)(gd_context *ctx, void *user);

gd_status gd_graph_run_immediate(gd_context *ctx,
                                 gd_device target,
                                 gd_immediate_build_fn build,
                                 void *user);
```

`gd_graph_run_immediate` creates a temporary graph, runs `build` with that graph active, compiles/runs it, materializes any still-live virtual tensors produced by that graph, then destroys the temporary graph.

Design choices:

- Graph lifecycle states are explicit: `empty/building/finalized/compiled`.
- `gd_graph_begin` requires an empty or reset graph and transitions to `building`.
- `gd_graph_end` finalizes the graph; no more nodes can be appended until reset.
- `gd_graph_compile` requires a finalized graph and transitions to `compiled`.
- `gd_graph_run` requires a compiled graph; it does not implicitly compile.
- `gd_graph_reset` returns a compiled/finalized graph to empty state after live virtual tensor checks.
- Public tensor API stays same for explicit graph use and immediate/debug helpers. Backends may return virtual tensors with no physical storage until materialized.
- IR is the central contract between frontend ops and backend kernels. It exposes whole patterns such as `rms_norm → qkv_linear → rope → attention`, allowing backends to lower to FlashAttention-style kernels instead of launching one kernel per tiny op.
- One-shot immediate helpers use the same graph/op schema to avoid two semantic implementations.

---

## 14. Autograd public API

Autograd attaches optional metadata to tensors.

```c
gd_status gd_tensor_set_requires_grad(gd_tensor *t, bool requires_grad);
bool      gd_tensor_requires_grad(const gd_tensor *t);

gd_status gd_backward(gd_context *ctx, gd_tensor *loss);
gd_status gd_tensor_grad(gd_tensor *t, gd_tensor **grad_out);
gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params);
```

No-grad guard:

```c
typedef struct gd_no_grad_guard gd_no_grad_guard;

gd_status gd_no_grad_enter(gd_context *ctx, gd_no_grad_guard **guard);
void      gd_no_grad_exit(gd_no_grad_guard *guard);
```

Design choices:

- Autograd records and transforms IR nodes independent of backend execution.
- `gd_tensor_set_requires_grad` is valid for leaf/materialized tensors before they are used in a graph. Changing `requires_grad` for tensors already captured by a graph returns `GD_ERR_INVALID_STATE`.
- `gd_backward` is called while a graph is active; it appends backward IR nodes from `loss` to leaf grad slots.
- `gd_zero_grad` records zero-fill IR nodes when a graph is active; debug helpers may run it through a one-shot graph.
- Backward graph nodes can be interpreted by CPU reference or lowered by a target backend.
- Grad tensor dtype follows policy: usually accum dtype F32/BF16, not necessarily param storage dtype.
- Quantized trainable weights usually have master weights; packed weights are derived/cache tensors.

---

## 15. Module and parameters

Module API first-class for training ergonomics.

```c
typedef struct gd_module gd_module;

gd_status gd_module_create(gd_context *ctx, const char *type_name, gd_module **out);
void      gd_module_destroy(gd_module *m);

gd_status gd_module_param(gd_module *m, const char *name, gd_tensor *param);
gd_status gd_module_child(gd_module *m, const char *name, gd_module *child);

gd_status gd_module_parameters(gd_module *m, gd_tensor ***params_out, int *n_out);
gd_status gd_module_zero_grad(gd_context *ctx, gd_module *m);
gd_status gd_module_save(gd_module *m, const char *path);
gd_status gd_module_load(gd_module *m, const char *path, bool strict);
```

Design choices:

- Explicit registration, C-friendly.
- Weight tying is supported by registering the same tensor/storage range in multiple module paths, e.g. GPT token embedding and LM head.
- Deduplicate parameters by parameter identity `(storage, byte offset, extent)`, not only tensor pointer. Exact tied weights get one optimizer state and one update.
- Gradients for tied parameters accumulate into the same grad slot.
- Transposed tying should use op attributes or views, not copied parameters, e.g. `hidden @ embed.weight^T`.
- `gd_module_parameters` returns a module-owned flat array valid until module destruction; tensors in the array are retained by the module, not by the caller.
- `gd_module_zero_grad` records zero-grad IR nodes when a graph is active; debug helpers may run it through a one-shot graph.
- State dict names are stable path strings: `blocks.0.attn.q.weight`.
- State dict loading preserves construction-time ties. Future state dict format should record alias metadata so saves do not need duplicate tied tensor data.

---

## 16. Optimizers

Optimizers operate on tensors, but may own internal state tensors on same or chosen device.

```c
typedef struct gd_optimizer gd_optimizer;

typedef struct gd_adamw_config {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    gd_dtype state_dtype;      // 0/default = F32 in v1
    int use_state_device;      // 0 = parameter device in v1
    gd_device state_device;
} gd_adamw_config;

gd_status gd_adamw_create(gd_context *ctx,
                          gd_tensor **params,
                          int n_params,
                          const gd_adamw_config *cfg,
                          gd_optimizer **out);

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *opt);
gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *opt);
void      gd_optimizer_destroy(gd_optimizer *opt);
```

Future: 8-bit optimizer states, paged optimizer states, sharded states.

Design choices:

- Optimizer retains parameter handles and owns optimizer-state tensors until destroyed.
- Optimizer APIs record update/zero-grad IR nodes when a graph is active.
- Immediate/debug optimizer helpers use one-shot graphs, not a separate optimizer execution path.
- Optimizer state dtype/device comes from optimizer config/policy, not hardcoded.

---

## 17. Private backend API

Backend implementations register capabilities and lowering hooks.

Private sketch:

```c
typedef struct _gd_backend_vtable {
    gd_device_type type;
    const char *name;

    gd_status (*alloc)(_gd_backend *, const gd_storage_desc *desc, gd_storage **out);
    gd_status (*free)(_gd_backend *, gd_storage *s);

    gd_status (*run_op)(_gd_backend *, const _gd_op_node *node);
    gd_status (*lower_graph)(_gd_backend *, _gd_graph_segment *seg, _gd_executable **out);
    gd_status (*run_executable)(_gd_backend *, _gd_executable *exe);
} _gd_backend_vtable;
```

Private dispatch key:

```text
op + backend + input dtypes/layouts + output dtype/layout + quant formats + compute policy
```

Examples:

```text
linear / CUDA / x=bf16 rowmajor / w=nf4_g64 packed / out=bf16
attention / Metal / qkv=fp16 / causal / head_dim=64
rmsnorm_swiglu / Vulkan / bf16
```

Design choices:

- CPU ref implements broad reference coverage for tests/debug and explicit fallback policy.
- GPU backends register specialized kernels and graph rewrite passes.
- Kernel cache keyed by static shapes, layouts, dtypes, quant format, and tuning params.

---

## 18. Fusion and IR private design

Internal IR nodes represent ops and metadata:

```text
node: op kind, inputs, attrs, output value(s), dtype/layout/device constraints
value: logical tensor produced by node; may be virtual until materialized
```

Fusion examples:

```text
rms_norm + linear(qkv) + rope + attention + linear(out)
linear + bias + silu
matmul(dequant(Wq), X) + bias
softmax + cross_entropy + backward
```

Passes:

1. shape inference
2. dtype/compute policy inference
3. layout selection
4. quant format legality checks
5. fusion/pattern matching
6. memory planning
7. backend lowering
8. autotune/cache

Design choices:

- Public API remains op-based; private IR gives GPU backends whole-pattern visibility.
- Compiler may avoid materializing virtual tensors such as dequantized weights, transposes, softmax probabilities, and attention score matrices.
- Compiler owns activation lifetime and workspace reuse. Future passes should support recompute/activation-checkpoint policies for large-model training.

---

## 19. Low-bit training model

Preferred model:

```text
master weights: F32/BF16
packed forward weights: int8/int4/nf4/other quant format cache
activations: BF16/F16/FP8 depending backend
accumulation: F32 or backend-optimized mixed precision
grads: BF16/F32
optimizer state: F32/BF16 initially, 8-bit later
```

Design choices:

- In v1, packed quantized tensors are non-differentiable storage/cache tensors and cannot be leaf trainable params.
- This does not block future quantized training: trainable low-bit weights are represented by floating master params plus explicit quantization IR ops.
- Future QAT/low-bit training can add ops such as `fake_quant`, `pack_quant`, or `quant_linear_with_master` with STE/LSQ-style backward policies.
- Gradients for quantized training accumulate into master floating params and/or scale/codebook tensors, not into packed bit storage.
- Optimizers update master params; packed forward weights are refreshed/repacked as derived graph/cache values.

---

## 20. Debugging and gradient correctness

Graph-only execution must still be easy to debug. Debugging APIs are part of the core design, not optional tooling.

Graph inspection:

```c
gd_status gd_scope_push(gd_context *ctx, const char *name);
gd_status gd_scope_pop(gd_context *ctx);
gd_status gd_tensor_set_name(gd_tensor *t, const char *name);

typedef enum gd_dump_format {
    GD_DUMP_TEXT,
    GD_DUMP_DOT,
    GD_DUMP_JSON
} gd_dump_format;

gd_status gd_graph_dump(gd_graph *g, gd_dump_format fmt, const char *path);
gd_status gd_graph_validate(gd_graph *g);
```

Tensor inspection:

```c
gd_status gd_tensor_materialize(gd_context *ctx, gd_tensor *t);
gd_status gd_tensor_to_cpu(gd_context *ctx, gd_tensor *t,
                           void *dst, size_t nbytes);
gd_status gd_debug_print_tensor(gd_context *ctx, gd_tensor *t,
                                int max_elems);
```

Partial execution and backend comparison:

```c
gd_status gd_graph_run_until(gd_graph *g, int node_id);
gd_status gd_graph_compare(gd_graph *g,
                           gd_device reference,
                           gd_device target,
                           const gd_compare_options *opts);
```

Compiler debug controls:

```c
typedef struct gd_graph_compile_options {
    bool enable_fusion;
    bool enable_layout_planning;
    bool enable_quant_specialization;
    bool deterministic;
} gd_graph_compile_options;

gd_status gd_graph_compile_ex(gd_graph *g,
                              gd_device target,
                              const gd_graph_compile_options *opts);
```

Gradient debugging:

```c
gd_status gd_gradcheck(gd_context *ctx,
                       gd_graph *g,
                       gd_tensor *loss,
                       gd_tensor **inputs,
                       int n_inputs,
                       const gd_gradcheck_options *opts);

gd_status gd_graph_compare_grads(gd_graph *g,
                                 gd_device reference,
                                 gd_device target,
                                 const gd_compare_options *opts);
```

Debug ops can be inserted into graphs:

```c
gd_status gd_assert_finite(gd_context *ctx, gd_tensor *t);
gd_status gd_assert_close(gd_context *ctx, gd_tensor *a, gd_tensor *b,
                          float atol, float rtol);
```

Required workflow for gradient bugs:

1. Run graph on `CPU_REF`.
2. Run target backend with fusion/layout planning disabled.
3. Run target backend with one pass enabled at a time.
4. Compare forward tensors node-by-node.
5. Compare backward gradients node-by-node.
6. Use `gd_gradcheck` on smallest failing subgraph.
7. Dump graph/fusion group and inspect named tensors around first mismatch.

Design choices:

- Every graph node has stable id, op name, optional user name, shape, dtype, layout, device, and source scope.
- Compiler can emit both pre-pass and post-pass IR dumps.
- Backend comparison reports first mismatch with node id, tensor index, max abs/rel error, and coordinates.
- Gradient check uses CPU reference and finite differences; optimized backends prove parity against CPU reference gradients.

---

## 21. Testing strategy

Required test classes:

1. Tensor metadata/view tests.
2. Dtype conversion tests.
3. Quant pack/dequant roundtrip tests.
4. CPU ref op correctness tests.
5. Backend parity tests vs CPU ref.
6. Finite-difference gradcheck for differentiable ops.
7. Broadcast/shape fuzz tests.
8. Graph immediate-helper parity tests (one-shot graph run vs explicit graph run).
9. Fusion parity tests across debug ladder: `CPU_REF → GPU_SAFE → GPU_LAYOUT → GPU_FUSED → GPU_FULL`.
10. Low-bit tolerance tests per format.

Design choice: every optimized backend kernel must prove parity against CPU ref within dtype-specific tolerance.

---

## 22. Initial MVP

Phase 1: core correctness

- context/status
- dtype/device
- storage/tensor/view
- CPU ref allocator/backend
- basic ops: add, mul, matmul, sum, relu, silu, softmax, cross entropy
- autograd for MVP ops
- finite-difference gradcheck
- SGD/AdamW
- Linear + MLP train example

Phase 2: quant foundation

- quant format registry
- int8, int4, nf4 formats
- pack/dequant CPU ref
- quantized linear CPU reference path: dequant + matmul when `GD_FALLBACK_CPU_REF` is enabled
- tests for odd formats, including 1.5-bit toy format

Phase 3: GPU performance path

- Metal graph lowering
- GPU-safe execution path with simple kernels
- memory planner integration on Metal
- first fused backend kernel: quantized linear + bias + activation
- Metal first
- GPU debug ladder: `GPU_SAFE`, `GPU_LAYOUT`, `GPU_FUSED`, `GPU_FULL`

Phase 4: transformer path

- RMSNorm
- RoPE
- attention
- fused QKV/attention/out projection
- transformer block training parity

---

## 23. Closed v1 decisions

1. **First GPU backend: Metal.** Optimize for fastest local iteration. Keep backend API CUDA/Vulkan-ready.
2. **ABI stability: source-stable only until API matures.** Use opaque handles now; binary stability later.
3. **Ownership: public retain/release, private arenas.** Public handles use refcounts/destructors. Graph/compiler temporaries use internal arenas.
4. **Execution model: graph-only core.** All ops lower through typed IR. Immediate debugging is implemented as one-shot graph build/compile/run, not a separate execution engine.
5. **Shape policy: static compiled shapes in v1.** Use separate compiled graphs for distinct shapes, e.g. GPT prefill `[B,T]` and decode `[B,1]`. Symbolic/dynamic shapes later.
6. **Quant descriptor ownership: refcounted immutable object.** Tensor retains `gd_quant_desc`; desc can retain scale/zero/codebook tensors.
7. **Async API: minimal public sync in v1, async-ready internals.** Expose `gd_synchronize(...)` initially. Streams/events/collectives later without blocking design.
8. **Graph control flow: straight-line graphs in v1.** Static build-time control flow is handled in C while constructing graph. Runtime loops live in host C by repeatedly running compiled graphs. Runtime conditionals use masks or host sync. Graph-level `if/while` later.
