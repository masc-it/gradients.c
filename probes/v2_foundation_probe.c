/*
 * v2 foundation probe.
 *
 * Standalone design probe for gradients.c v2 foundations. No v1 headers, no real
 * backend, no kernels. Exercises invariants from docs/design_spec.md:
 * arenas, aligned spans, reset generations, sealed params, scratch/data rings,
 * scoped lifecycle, no hot-path heap allocation, concrete tensors/views,
 * Module/ModuleList/ModuleDict traversal, tied-weight dedup, param groups, and
 * checkpoint manifest shape.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROBE_MAX_NDIM 4
#define PROBE_MAX_PATH 256
#define PROBE_MAX_PARAMS 256
#define PROBE_MAX_ALIASES 8
#define PROBE_UNUSED(x) ((void)(x))

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "probe failed: %s (%s:%d)\n", (msg), __FILE__,   \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_EQ_U64(a, b, msg) CHECK((uint64_t)(a) == (uint64_t)(b), (msg))
#define CHECK_EQ_I64(a, b, msg) CHECK((int64_t)(a) == (int64_t)(b), (msg))
#define CHECK_OK(rc, msg) CHECK((rc) == 0, (msg))

typedef enum probe_dtype {
    PROBE_F16,
    PROBE_F32,
    PROBE_I32,
} probe_dtype;

typedef enum probe_scope_mode {
    PROBE_SCOPE_NONE,
    PROBE_SCOPE_TRAIN,
    PROBE_SCOPE_INFER,
} probe_scope_mode;

static size_t dtype_size(probe_dtype dtype)
{
    switch (dtype) {
    case PROBE_F16: return 2U;
    case PROBE_F32: return 4U;
    case PROBE_I32: return 4U;
    default: return 0U;
    }
}

static const char *dtype_name(probe_dtype dtype)
{
    switch (dtype) {
    case PROBE_F16: return "f16";
    case PROBE_F32: return "f32";
    case PROBE_I32: return "i32";
    default: return "?";
    }
}

static size_t align_up(size_t v, size_t align)
{
    CHECK(align != 0U && (align & (align - 1U)) == 0U, "alignment must be power of two");
    return (v + align - 1U) & ~(align - 1U);
}

typedef enum probe_arena_kind {
    PROBE_ARENA_PARAMS,
    PROBE_ARENA_STATE,
    PROBE_ARENA_SCRATCH,
    PROBE_ARENA_DATA,
} probe_arena_kind;

typedef enum probe_alloc_status {
    PROBE_ALLOC_OK,
    PROBE_ALLOC_OOM,
    PROBE_ALLOC_FROZEN,
} probe_alloc_status;

typedef struct probe_span {
    void *ptr;
    size_t offset;
    size_t nbytes;
    uint64_t generation;
} probe_span;

typedef struct probe_arena {
    char name[32];
    probe_arena_kind kind;
    unsigned char *base;
    size_t capacity;
    size_t offset;
    size_t watermark;
    uint64_t generation;
    bool frozen;
} probe_arena;

static uint64_t g_probe_heap_allocs;
static bool g_probe_heap_forbidden;

static const char *arena_kind_name(probe_arena_kind kind)
{
    switch (kind) {
    case PROBE_ARENA_PARAMS: return "params";
    case PROBE_ARENA_STATE: return "state";
    case PROBE_ARENA_SCRATCH: return "scratch";
    case PROBE_ARENA_DATA: return "data";
    default: return "?";
    }
}

static void *probe_calloc(size_t count, size_t size)
{
    void *ptr;
    CHECK(!g_probe_heap_forbidden, "heap allocation in hot path");
    ptr = calloc(count, size);
    CHECK(ptr != NULL, "heap allocation failed");
    g_probe_heap_allocs += 1U;
    return ptr;
}

static void *probe_aligned_zero_alloc(size_t align, size_t size)
{
    void *ptr = NULL;
    int rc;
    CHECK(!g_probe_heap_forbidden, "heap allocation in hot path");
    CHECK((align & (align - 1U)) == 0U, "aligned allocation power-of-two align");
    rc = posix_memalign(&ptr, align, align_up(size, align));
    CHECK(rc == 0 && ptr != NULL, "aligned heap allocation failed");
    memset(ptr, 0, size);
    g_probe_heap_allocs += 1U;
    return ptr;
}

static void arena_init(probe_arena *arena,
                       probe_arena_kind kind,
                       const char *name,
                       size_t capacity)
{
    memset(arena, 0, sizeof(*arena));
    snprintf(arena->name, sizeof(arena->name), "%s", name);
    arena->kind = kind;
    arena->base = (unsigned char *)probe_aligned_zero_alloc(256U, capacity);
    arena->capacity = capacity;
    arena->generation = 1U;
}

static void arena_destroy(probe_arena *arena)
{
    if (arena == NULL) {
        return;
    }
    free(arena->base);
    memset(arena, 0, sizeof(*arena));
}

static void arena_reset(probe_arena *arena)
{
    CHECK(arena != NULL, "arena_reset arena");
    CHECK(!arena->frozen, "arena_reset frozen arena");
    arena->offset = 0U;
    arena->generation += 1U;
    CHECK(arena->generation != 0U, "arena generation overflow");
}

static void arena_freeze(probe_arena *arena)
{
    CHECK(arena != NULL, "arena_freeze arena");
    arena->frozen = true;
}

static probe_alloc_status arena_try_alloc_span(probe_arena *arena,
                                               size_t nbytes,
                                               size_t align,
                                               probe_span *out)
{
    size_t off;
    void *ptr;
    CHECK(arena != NULL && out != NULL, "arena_try_alloc args");
    CHECK(nbytes > 0U, "arena_try_alloc nonzero");
    if (align == 0U) {
        align = 8U;
    }
    CHECK((align & (align - 1U)) == 0U, "arena_try_alloc power-of-two align");
    memset(out, 0, sizeof(*out));
    if (arena->frozen) {
        return PROBE_ALLOC_FROZEN;
    }
    {
        size_t base_mod = (size_t)((uintptr_t)arena->base & (uintptr_t)(align - 1U));
        off = align_up(arena->offset + base_mod, align) - base_mod;
    }
    if (off > arena->capacity || nbytes > arena->capacity - off) {
        return PROBE_ALLOC_OOM;
    }
    ptr = arena->base + off;
    memset(ptr, 0, nbytes);
    arena->offset = off + nbytes;
    if (arena->offset > arena->watermark) {
        arena->watermark = arena->offset;
    }
    out->ptr = ptr;
    out->offset = off;
    out->nbytes = nbytes;
    out->generation = arena->generation;
    return PROBE_ALLOC_OK;
}

static probe_span arena_alloc_span(probe_arena *arena, size_t nbytes, size_t align)
{
    probe_span span;
    probe_alloc_status status = arena_try_alloc_span(arena, nbytes, align, &span);
    CHECK(status == PROBE_ALLOC_OK, "arena alloc failed");
    return span;
}

static void *arena_alloc(probe_arena *arena, size_t nbytes, size_t align)
{
    return arena_alloc_span(arena, nbytes, align).ptr;
}

static char *arena_strdup(probe_arena *arena, const char *s)
{
    size_t n = strlen(s) + 1U;
    char *out = (char *)arena_alloc(arena, n, 1U);
    memcpy(out, s, n);
    return out;
}

static char *arena_strdup_i(probe_arena *arena, int value)
{
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%d", value);
    return arena_strdup(arena, tmp);
}

typedef struct fake_backend {
    uint64_t next_fence;
    uint64_t completed_fence;
    uint64_t waits;
} fake_backend;

static uint64_t fake_backend_record(fake_backend *backend)
{
    backend->next_fence += 1U;
    return backend->next_fence;
}

static void fake_backend_wait_until(fake_backend *backend, uint64_t fence)
{
    if (fence > backend->completed_fence) {
        backend->completed_fence = fence;
        backend->waits += 1U;
    }
}

static void fake_backend_complete_all(fake_backend *backend)
{
    backend->completed_fence = backend->next_fence;
}

typedef struct probe_ring_arena {
    char name[32];
    probe_arena *slots;
    uint64_t *slot_fence;
    int n_slots;
    int current;
    uint64_t waits;
} probe_ring_arena;

static void ring_init(probe_ring_arena *ring,
                      probe_arena_kind kind,
                      const char *name,
                      int n_slots,
                      size_t slot_capacity)
{
    int i;
    memset(ring, 0, sizeof(*ring));
    snprintf(ring->name, sizeof(ring->name), "%s", name);
    CHECK(kind == PROBE_ARENA_SCRATCH || kind == PROBE_ARENA_DATA, "ring arena kind");
    CHECK(n_slots > 0, "ring slots > 0");
    ring->slots = (probe_arena *)probe_calloc((size_t)n_slots, sizeof(*ring->slots));
    ring->slot_fence = (uint64_t *)probe_calloc((size_t)n_slots, sizeof(*ring->slot_fence));
    ring->n_slots = n_slots;
    ring->current = -1;
    for (i = 0; i < n_slots; ++i) {
        char slot_name[64];
        snprintf(slot_name, sizeof(slot_name), "%s[%d]", name, i);
        arena_init(&ring->slots[i], kind, slot_name, slot_capacity);
    }
}

static void ring_destroy(probe_ring_arena *ring)
{
    int i;
    if (ring == NULL) {
        return;
    }
    for (i = 0; i < ring->n_slots; ++i) {
        arena_destroy(&ring->slots[i]);
    }
    free(ring->slots);
    free(ring->slot_fence);
    memset(ring, 0, sizeof(*ring));
}

static int ring_oldest_busy_slot(const probe_ring_arena *ring)
{
    int i;
    int best = -1;
    uint64_t best_fence = UINT64_MAX;
    for (i = 0; i < ring->n_slots; ++i) {
        uint64_t f = ring->slot_fence[i];
        if (f != 0U && f < best_fence) {
            best_fence = f;
            best = i;
        }
    }
    return best;
}

static int ring_select_slot(probe_ring_arena *ring, fake_backend *backend)
{
    int attempt;
    int start;
    CHECK(ring != NULL && backend != NULL, "ring_select args");
    start = ring->current < 0 ? 0 : (ring->current + 1) % ring->n_slots;
    for (attempt = 0; attempt < ring->n_slots; ++attempt) {
        int idx = (start + attempt) % ring->n_slots;
        if (ring->slot_fence[idx] <= backend->completed_fence) {
            ring->current = idx;
            ring->slot_fence[idx] = 0U;
            arena_reset(&ring->slots[idx]);
            return idx;
        }
    }

    /* Ring exhausted: wait only for oldest in-flight slot. */
    {
        int oldest = ring_oldest_busy_slot(ring);
        CHECK(oldest >= 0, "ring exhausted but no busy slot");
        fake_backend_wait_until(backend, ring->slot_fence[oldest]);
        ring->waits += 1U;
        ring->current = oldest;
        ring->slot_fence[oldest] = 0U;
        arena_reset(&ring->slots[oldest]);
        return oldest;
    }
}

static void ring_record_fence(probe_ring_arena *ring, uint64_t fence)
{
    CHECK(ring != NULL && ring->current >= 0, "ring_record current slot");
    ring->slot_fence[ring->current] = fence;
}

typedef struct probe_context {
    probe_arena params;
    probe_arena state;
    probe_ring_arena scratch;
    probe_ring_arena data;
    fake_backend backend;
    probe_scope_mode mode;
    bool in_scope;
    int scratch_slot;
    int data_slot;
    uint64_t last_scope_fence;
} probe_context;

static void ctx_init(probe_context *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    arena_init(&ctx->params, PROBE_ARENA_PARAMS, "params", 1024U * 1024U);
    arena_init(&ctx->state, PROBE_ARENA_STATE, "state", 512U * 1024U);
    ring_init(&ctx->scratch, PROBE_ARENA_SCRATCH, "scratch", 2, 128U * 1024U);
    ring_init(&ctx->data, PROBE_ARENA_DATA, "data", 2, 64U * 1024U);
}

static void ctx_destroy(probe_context *ctx)
{
    arena_destroy(&ctx->params);
    arena_destroy(&ctx->state);
    ring_destroy(&ctx->scratch);
    ring_destroy(&ctx->data);
    memset(ctx, 0, sizeof(*ctx));
}

static probe_arena *ctx_scratch(probe_context *ctx)
{
    CHECK(ctx != NULL && ctx->in_scope, "scratch requires active scope");
    CHECK(ctx->scratch.current >= 0, "scratch current slot");
    return &ctx->scratch.slots[ctx->scratch.current];
}

static probe_arena *ctx_data(probe_context *ctx)
{
    CHECK(ctx != NULL && ctx->in_scope, "data requires active scope");
    CHECK(ctx->data.current >= 0, "data current slot");
    return &ctx->data.slots[ctx->data.current];
}

static void gd_begin(probe_context *ctx, probe_scope_mode mode)
{
    CHECK(ctx != NULL && !ctx->in_scope, "gd_begin state");
    CHECK(mode == PROBE_SCOPE_TRAIN || mode == PROBE_SCOPE_INFER, "gd_begin mode");
    ctx->scratch_slot = ring_select_slot(&ctx->scratch, &ctx->backend);
    ctx->data_slot = ring_select_slot(&ctx->data, &ctx->backend);
    ctx->mode = mode;
    ctx->in_scope = true;
}

static void gd_end(probe_context *ctx)
{
    uint64_t fence;
    CHECK(ctx != NULL && ctx->in_scope, "gd_end state");
    fence = fake_backend_record(&ctx->backend);
    ring_record_fence(&ctx->scratch, fence);
    ring_record_fence(&ctx->data, fence);
    ctx->last_scope_fence = fence;
    ctx->mode = PROBE_SCOPE_NONE;
    ctx->in_scope = false;
}

typedef struct probe_tensor {
    const char *name;
    probe_dtype dtype;
    int ndim;
    int64_t sizes[PROBE_MAX_NDIM];
    int64_t strides[PROBE_MAX_NDIM];
    probe_arena *header_arena;
    uint64_t header_generation;
    probe_arena *storage_arena;
    size_t allocation_offset;
    size_t allocation_nbytes;
    size_t view_offset;
    uint64_t storage_generation;
    bool is_view;
    bool requires_grad;
    bool trainable;
    struct probe_tensor *base;
} probe_tensor;

static int64_t tensor_numel(int ndim, const int64_t *sizes)
{
    int i;
    int64_t n = 1;
    for (i = 0; i < ndim; ++i) {
        CHECK(sizes[i] > 0, "tensor size positive");
        CHECK(n <= INT64_MAX / sizes[i], "tensor numel overflow");
        n *= sizes[i];
    }
    return n;
}

static void tensor_make_contiguous_strides(probe_tensor *t)
{
    int i;
    int64_t stride = 1;
    for (i = t->ndim - 1; i >= 0; --i) {
        t->strides[i] = stride;
        stride *= t->sizes[i];
    }
}

static bool tensor_is_contiguous(const probe_tensor *t)
{
    int i;
    int64_t stride = 1;
    if (t == NULL) {
        return false;
    }
    for (i = t->ndim - 1; i >= 0; --i) {
        if (t->strides[i] != stride) {
            return false;
        }
        stride *= t->sizes[i];
    }
    return true;
}

static size_t tensor_first_byte_offset(const probe_tensor *t)
{
    CHECK(t != NULL, "tensor_first_byte_offset tensor");
    return t->allocation_offset + t->view_offset;
}

static size_t tensor_logical_byte_span_from_allocation(const probe_tensor *t)
{
    int i;
    int64_t max_elem_offset = 0;
    CHECK(t != NULL, "tensor byte span tensor");
    for (i = 0; i < t->ndim; ++i) {
        CHECK(t->strides[i] >= 0, "negative strides not in probe");
        CHECK(max_elem_offset <= INT64_MAX - (t->sizes[i] - 1) * t->strides[i],
              "tensor view span overflow");
        max_elem_offset += (t->sizes[i] - 1) * t->strides[i];
    }
    return t->view_offset + (size_t)max_elem_offset * dtype_size(t->dtype) + dtype_size(t->dtype);
}

static void tensor_check_live(const probe_tensor *t)
{
    CHECK(t != NULL, "tensor_check_live tensor");
    CHECK(t->header_arena != NULL && t->storage_arena != NULL, "tensor arenas set");
    CHECK(t->header_generation == t->header_arena->generation,
          "tensor header generation is live");
    CHECK(t->storage_generation == t->storage_arena->generation,
          "tensor storage generation is live");
    CHECK(tensor_logical_byte_span_from_allocation(t) <= t->allocation_nbytes,
          "tensor logical span inside allocation");
}

static probe_tensor *tensor_empty(probe_arena *arena,
                                  const char *name,
                                  probe_dtype dtype,
                                  int ndim,
                                  const int64_t *sizes)
{
    int i;
    size_t bytes;
    probe_span header;
    probe_span storage;
    probe_tensor *t;
    CHECK(arena != NULL && name != NULL && sizes != NULL, "tensor_empty args");
    CHECK(ndim >= 0 && ndim <= PROBE_MAX_NDIM, "tensor ndim range");
    header = arena_alloc_span(arena, sizeof(*t), 8U);
    t = (probe_tensor *)header.ptr;
    t->name = arena_strdup(arena, name);
    t->dtype = dtype;
    t->ndim = ndim;
    for (i = 0; i < ndim; ++i) {
        t->sizes[i] = sizes[i];
    }
    tensor_make_contiguous_strides(t);
    bytes = (size_t)tensor_numel(ndim, sizes) * dtype_size(dtype);
    storage = arena_alloc_span(arena, bytes, 64U);
    t->header_arena = arena;
    t->header_generation = header.generation;
    t->storage_arena = arena;
    t->allocation_offset = storage.offset;
    t->allocation_nbytes = storage.nbytes;
    t->view_offset = 0U;
    t->storage_generation = storage.generation;
    t->is_view = false;
    t->requires_grad = false;
    t->trainable = false;
    t->base = NULL;
    tensor_check_live(t);
    return t;
}

static probe_tensor *tensor_slice(probe_arena *header_arena,
                                  probe_tensor *base,
                                  int dim,
                                  int64_t start,
                                  int64_t len,
                                  const char *name)
{
    int i;
    probe_span header;
    probe_tensor *t;
    CHECK(header_arena != NULL && base != NULL && name != NULL, "tensor_slice args");
    tensor_check_live(base);
    CHECK(dim >= 0 && dim < base->ndim, "tensor_slice dim");
    CHECK(start >= 0 && len > 0 && start + len <= base->sizes[dim], "tensor_slice range");
    header = arena_alloc_span(header_arena, sizeof(*t), 8U);
    t = (probe_tensor *)header.ptr;
    *t = *base;
    t->header_arena = header_arena;
    t->header_generation = header.generation;
    t->name = arena_strdup(header_arena, name);
    for (i = 0; i < base->ndim; ++i) {
        t->sizes[i] = base->sizes[i];
        t->strides[i] = base->strides[i];
    }
    t->sizes[dim] = len;
    t->view_offset = base->view_offset + (size_t)(start * base->strides[dim]) * dtype_size(base->dtype);
    t->is_view = true;
    t->base = base->base != NULL ? base->base : base;
    tensor_check_live(t);
    return t;
}

static probe_tensor *tensor_contiguous(probe_arena *arena, probe_tensor *src, const char *name)
{
    int i;
    size_t bytes;
    probe_span header;
    probe_span storage;
    probe_tensor *out;
    CHECK(arena != NULL && src != NULL, "tensor_contiguous args");
    tensor_check_live(src);
    header = arena_alloc_span(arena, sizeof(*out), 8U);
    out = (probe_tensor *)header.ptr;
    out->name = arena_strdup(arena, name);
    out->dtype = src->dtype;
    out->ndim = src->ndim;
    for (i = 0; i < src->ndim; ++i) {
        out->sizes[i] = src->sizes[i];
    }
    tensor_make_contiguous_strides(out);
    bytes = (size_t)tensor_numel(out->ndim, out->sizes) * dtype_size(out->dtype);
    storage = arena_alloc_span(arena, bytes, 64U);
    out->header_arena = arena;
    out->header_generation = header.generation;
    out->storage_arena = arena;
    out->allocation_offset = storage.offset;
    out->allocation_nbytes = storage.nbytes;
    out->view_offset = 0U;
    out->storage_generation = storage.generation;
    out->is_view = false;
    out->requires_grad = src->requires_grad;
    out->trainable = false;
    out->base = NULL;
    tensor_check_live(out);
    return out;
}

typedef struct probe_state_object {
    const char *name;
    size_t offset;
    size_t nbytes;
    uint64_t generation;
    uint64_t last_use_fence;
} probe_state_object;

static void state_object_create(probe_context *ctx,
                                probe_state_object *obj,
                                const char *name,
                                size_t nbytes)
{
    probe_span storage;
    CHECK(ctx != NULL && obj != NULL && name != NULL, "state_object_create args");
    memset(obj, 0, sizeof(*obj));
    obj->name = arena_strdup(&ctx->state, name);
    storage = arena_alloc_span(&ctx->state, nbytes, 64U);
    obj->offset = storage.offset;
    obj->nbytes = storage.nbytes;
    obj->generation = storage.generation;
    obj->last_use_fence = 0U;
}

static void state_object_check_live(const probe_context *ctx, const probe_state_object *obj)
{
    CHECK(ctx != NULL && obj != NULL, "state_object_check_live args");
    CHECK(obj->generation == ctx->state.generation, "state object generation is live");
    CHECK(obj->offset <= ctx->state.capacity && obj->nbytes <= ctx->state.capacity - obj->offset,
          "state object span inside state arena");
}

static void state_object_mark_used(probe_state_object *obj, uint64_t fence)
{
    CHECK(obj != NULL && fence != 0U, "state_object_mark_used args");
    obj->last_use_fence = fence;
}

static bool state_object_reset_or_realloc(probe_context *ctx,
                                          probe_state_object *obj,
                                          size_t nbytes,
                                          bool allow_realloc)
{
    CHECK(ctx != NULL && obj != NULL, "state_object_reset args");
    state_object_check_live(ctx, obj);
    if (obj->last_use_fence > ctx->backend.completed_fence) {
        if (allow_realloc) {
            probe_span storage = arena_alloc_span(&ctx->state, nbytes, 64U);
            obj->offset = storage.offset;
            obj->nbytes = storage.nbytes;
            obj->generation = storage.generation;
            obj->last_use_fence = 0U;
            return true;
        }
        fake_backend_wait_until(&ctx->backend, obj->last_use_fence);
    }
    if (nbytes > obj->nbytes) {
        probe_span storage = arena_alloc_span(&ctx->state, nbytes, 64U);
        obj->offset = storage.offset;
        obj->nbytes = storage.nbytes;
        obj->generation = storage.generation;
        obj->last_use_fence = 0U;
        return true;
    }
    obj->last_use_fence = 0U;
    return false;
}

static void state_object_reset(probe_context *ctx, probe_state_object *obj)
{
    (void)state_object_reset_or_realloc(ctx, obj, obj->nbytes, false);
}

typedef struct probe_named_tensor {
    const char *name;
    probe_tensor *tensor;
} probe_named_tensor;

typedef struct probe_named_child {
    const char *name;
    struct probe_module *child;
} probe_named_child;

typedef struct probe_module {
    const char *name;
    bool training;
    probe_named_tensor *params;
    int n_params;
    int cap_params;
    probe_named_tensor *buffers;
    int n_buffers;
    int cap_buffers;
    probe_named_child *children;
    int n_children;
    int cap_children;
} probe_module;

typedef struct probe_module_list {
    probe_module mod;
    probe_module **items;
    int n;
} probe_module_list;

typedef struct probe_module_dict {
    probe_module mod;
} probe_module_dict;

static void module_init(probe_context *ctx,
                        probe_module *mod,
                        const char *name,
                        int cap_params,
                        int cap_buffers,
                        int cap_children)
{
    memset(mod, 0, sizeof(*mod));
    mod->name = arena_strdup(&ctx->params, name);
    mod->training = true;
    mod->cap_params = cap_params;
    mod->cap_buffers = cap_buffers;
    mod->cap_children = cap_children;
    if (cap_params > 0) {
        mod->params = (probe_named_tensor *)arena_alloc(&ctx->params,
            (size_t)cap_params * sizeof(*mod->params), 8U);
    }
    if (cap_buffers > 0) {
        mod->buffers = (probe_named_tensor *)arena_alloc(&ctx->params,
            (size_t)cap_buffers * sizeof(*mod->buffers), 8U);
    }
    if (cap_children > 0) {
        mod->children = (probe_named_child *)arena_alloc(&ctx->params,
            (size_t)cap_children * sizeof(*mod->children), 8U);
    }
}

static void module_add_param(probe_module *mod, const char *name, probe_tensor *tensor)
{
    CHECK(mod->n_params < mod->cap_params, "module param capacity");
    mod->params[mod->n_params].name = name;
    mod->params[mod->n_params].tensor = tensor;
    tensor->requires_grad = true;
    tensor->trainable = true;
    mod->n_params += 1;
}

static void module_add_buffer(probe_module *mod, const char *name, probe_tensor *tensor)
{
    CHECK(mod->n_buffers < mod->cap_buffers, "module buffer capacity");
    mod->buffers[mod->n_buffers].name = name;
    mod->buffers[mod->n_buffers].tensor = tensor;
    mod->n_buffers += 1;
}

static void module_add_child(probe_module *mod, const char *name, probe_module *child)
{
    CHECK(mod->n_children < mod->cap_children, "module child capacity");
    mod->children[mod->n_children].name = name;
    mod->children[mod->n_children].child = child;
    mod->n_children += 1;
}

static void module_set_training(probe_module *mod, bool training)
{
    int i;
    mod->training = training;
    for (i = 0; i < mod->n_children; ++i) {
        module_set_training(mod->children[i].child, training);
    }
}

static void module_list_init(probe_context *ctx, probe_module_list *list, const char *name, int n)
{
    int i;
    module_init(ctx, &list->mod, name, 0, 0, n);
    list->n = n;
    list->items = (probe_module **)arena_alloc(&ctx->params, (size_t)n * sizeof(*list->items), 8U);
    for (i = 0; i < n; ++i) {
        list->items[i] = NULL;
    }
}

static void module_list_set(probe_context *ctx, probe_module_list *list, int index, probe_module *child)
{
    char *name;
    CHECK(index >= 0 && index < list->n, "module_list index");
    CHECK(list->items[index] == NULL, "module_list set once");
    name = arena_strdup_i(&ctx->params, index);
    list->items[index] = child;
    module_add_child(&list->mod, name, child);
}

static void module_dict_init(probe_context *ctx, probe_module_dict *dict, const char *name, int cap)
{
    module_init(ctx, &dict->mod, name, 0, 0, cap);
}

static void module_dict_add(probe_module_dict *dict, const char *name, probe_module *child)
{
    module_add_child(&dict->mod, name, child);
}

typedef struct probe_param_group {
    const char *name;
    const char *match;
    float lr_mult;
    float weight_decay;
} probe_param_group;

typedef struct probe_collected_param {
    char path[PROBE_MAX_PATH];
    char aliases[PROBE_MAX_ALIASES][PROBE_MAX_PATH];
    int n_aliases;
    probe_tensor *tensor;
    const probe_param_group *group;
} probe_collected_param;

typedef struct probe_param_set {
    probe_collected_param items[PROBE_MAX_PARAMS];
    int n;
} probe_param_set;

static bool glob_match_here(const char *pat, const char *text)
{
    if (*pat == '\0') {
        return *text == '\0';
    }
    if (*pat == '*') {
        while (*pat == '*') {
            ++pat;
        }
        if (*pat == '\0') {
            return true;
        }
        while (*text != '\0') {
            if (glob_match_here(pat, text)) {
                return true;
            }
            ++text;
        }
        return glob_match_here(pat, text);
    }
    if (*text != '\0' && *pat == *text) {
        return glob_match_here(pat + 1, text + 1);
    }
    return false;
}

static bool glob_match(const char *pat, const char *text)
{
    return glob_match_here(pat, text);
}

static const probe_param_group *match_group(const char *path,
                                            const probe_param_group *groups,
                                            int n_groups)
{
    int i;
    for (i = 0; i < n_groups; ++i) {
        if (glob_match(groups[i].match, path)) {
            return &groups[i];
        }
    }
    return NULL;
}

static int param_set_find_tensor(const probe_param_set *set, const probe_tensor *tensor)
{
    int i;
    for (i = 0; i < set->n; ++i) {
        if (set->items[i].tensor == tensor) {
            return i;
        }
    }
    return -1;
}

static void param_set_add(probe_param_set *set,
                          const char *path,
                          probe_tensor *tensor,
                          const probe_param_group *groups,
                          int n_groups)
{
    int existing = param_set_find_tensor(set, tensor);
    if (existing >= 0) {
        probe_collected_param *p = &set->items[existing];
        CHECK(p->n_aliases < PROBE_MAX_ALIASES, "param alias capacity");
        snprintf(p->aliases[p->n_aliases], sizeof(p->aliases[p->n_aliases]), "%s", path);
        p->n_aliases += 1;
        return;
    }
    CHECK(set->n < PROBE_MAX_PARAMS, "param set capacity");
    snprintf(set->items[set->n].path, sizeof(set->items[set->n].path), "%s", path);
    set->items[set->n].tensor = tensor;
    set->items[set->n].n_aliases = 0;
    set->items[set->n].group = match_group(path, groups, n_groups);
    set->n += 1;
}

static void join_path(char *out, size_t cap, const char *a, const char *b)
{
    if (a == NULL || a[0] == '\0') {
        snprintf(out, cap, "%s", b);
    } else {
        snprintf(out, cap, "%s.%s", a, b);
    }
}

static void collect_module_params_rec(const probe_module *mod,
                                      const char *path,
                                      const probe_param_group *groups,
                                      int n_groups,
                                      probe_param_set *set)
{
    int i;
    char child_path[PROBE_MAX_PATH];
    for (i = 0; i < mod->n_params; ++i) {
        char ppath[PROBE_MAX_PATH];
        join_path(ppath, sizeof(ppath), path, mod->params[i].name);
        if (mod->params[i].tensor->trainable) {
            param_set_add(set, ppath, mod->params[i].tensor, groups, n_groups);
        }
    }
    for (i = 0; i < mod->n_children; ++i) {
        join_path(child_path, sizeof(child_path), path, mod->children[i].name);
        collect_module_params_rec(mod->children[i].child, child_path, groups, n_groups, set);
    }
}

static void collect_module_params(const probe_module *root,
                                  const probe_param_group *groups,
                                  int n_groups,
                                  probe_param_set *set)
{
    memset(set, 0, sizeof(*set));
    collect_module_params_rec(root, root->name, groups, n_groups, set);
}

static bool param_set_has_path(const probe_param_set *set, const char *path)
{
    int i;
    int j;
    for (i = 0; i < set->n; ++i) {
        if (strcmp(set->items[i].path, path) == 0) {
            return true;
        }
        for (j = 0; j < set->items[i].n_aliases; ++j) {
            if (strcmp(set->items[i].aliases[j], path) == 0) {
                return true;
            }
        }
    }
    return false;
}

static const probe_collected_param *param_set_find_path(const probe_param_set *set,
                                                        const char *path)
{
    int i;
    int j;
    for (i = 0; i < set->n; ++i) {
        if (strcmp(set->items[i].path, path) == 0) {
            return &set->items[i];
        }
        for (j = 0; j < set->items[i].n_aliases; ++j) {
            if (strcmp(set->items[i].aliases[j], path) == 0) {
                return &set->items[i];
            }
        }
    }
    return NULL;
}

/* Fake module types for VLM tree. */
typedef struct fake_linear {
    probe_module mod;
    probe_tensor *weight;
    probe_tensor *bias;
} fake_linear;

typedef struct fake_attn {
    probe_module mod;
    probe_tensor *wq_weight;
} fake_attn;

typedef struct fake_block {
    probe_module mod;
    fake_attn attn;
} fake_block;

typedef struct fake_lm_head {
    probe_module mod;
    probe_tensor *weight;
} fake_lm_head;

typedef struct fake_repr_head {
    probe_module mod;
    fake_linear proj;
} fake_repr_head;

typedef struct fake_backbone {
    probe_module mod;
    fake_linear image_proj;
    probe_tensor *tok_emb_weight;
    probe_tensor *pos_buffer;
    probe_module_list blocks;
    fake_block *block;
} fake_backbone;

typedef struct fake_vlm {
    probe_module mod;
    fake_backbone backbone;
    probe_module_dict heads;
    fake_lm_head lm;
    fake_repr_head repr;
} fake_vlm;

static void fake_linear_init(probe_context *ctx,
                             fake_linear *lin,
                             const char *name,
                             int64_t in,
                             int64_t out)
{
    int64_t ws[2] = {in, out};
    int64_t bs[1] = {out};
    module_init(ctx, &lin->mod, name, 2, 0, 0);
    lin->weight = tensor_empty(&ctx->params, "linear.weight", PROBE_F16, 2, ws);
    lin->bias = tensor_empty(&ctx->params, "linear.bias", PROBE_F16, 1, bs);
    module_add_param(&lin->mod, "weight", lin->weight);
    module_add_param(&lin->mod, "bias", lin->bias);
}

static void fake_attn_init(probe_context *ctx, fake_attn *attn)
{
    int64_t ws[2] = {32, 32};
    module_init(ctx, &attn->mod, "attn", 1, 0, 0);
    attn->wq_weight = tensor_empty(&ctx->params, "wq.weight", PROBE_F16, 2, ws);
    module_add_param(&attn->mod, "wq.weight", attn->wq_weight);
}

static void fake_block_init(probe_context *ctx, fake_block *block, int index)
{
    char name[32];
    snprintf(name, sizeof(name), "block%d", index);
    module_init(ctx, &block->mod, name, 0, 0, 1);
    fake_attn_init(ctx, &block->attn);
    module_add_child(&block->mod, "attn", &block->attn.mod);
}

static void fake_lm_head_init_tied(probe_context *ctx,
                                   fake_lm_head *lm,
                                   probe_tensor *tied_weight)
{
    PROBE_UNUSED(ctx);
    module_init(ctx, &lm->mod, "lm", 1, 0, 0);
    lm->weight = tied_weight;
    module_add_param(&lm->mod, "weight", lm->weight);
}

static void fake_repr_head_init(probe_context *ctx, fake_repr_head *repr)
{
    module_init(ctx, &repr->mod, "repr", 0, 0, 1);
    fake_linear_init(ctx, &repr->proj, "proj", 32, 16);
    module_add_child(&repr->mod, "proj", &repr->proj.mod);
}

static void fake_backbone_init(probe_context *ctx, fake_backbone *bb, int n_layers)
{
    int64_t emb[2] = {128, 32};
    int64_t pos[2] = {64, 32};
    int i;
    module_init(ctx, &bb->mod, "backbone", 1, 1, 2);
    fake_linear_init(ctx, &bb->image_proj, "image_proj", 48, 32);
    module_add_child(&bb->mod, "image_proj", &bb->image_proj.mod);

    bb->tok_emb_weight = tensor_empty(&ctx->params, "tok_emb.weight", PROBE_F16, 2, emb);
    module_add_param(&bb->mod, "tok_emb.weight", bb->tok_emb_weight);

    bb->pos_buffer = tensor_empty(&ctx->params, "rope_freqs", PROBE_F32, 2, pos);
    module_add_buffer(&bb->mod, "rope_freqs", bb->pos_buffer);

    module_list_init(ctx, &bb->blocks, "blocks", n_layers);
    bb->block = (fake_block *)arena_alloc(&ctx->params, (size_t)n_layers * sizeof(*bb->block), 8U);
    for (i = 0; i < n_layers; ++i) {
        fake_block_init(ctx, &bb->block[i], i);
        module_list_set(ctx, &bb->blocks, i, &bb->block[i].mod);
    }
    module_add_child(&bb->mod, "blocks", &bb->blocks.mod);
}

static void fake_vlm_init(probe_context *ctx, fake_vlm *m)
{
    module_init(ctx, &m->mod, "vlm", 0, 0, 2);
    fake_backbone_init(ctx, &m->backbone, 3);
    module_add_child(&m->mod, "backbone", &m->backbone.mod);

    module_dict_init(ctx, &m->heads, "heads", 2);
    fake_lm_head_init_tied(ctx, &m->lm, m->backbone.tok_emb_weight);
    fake_repr_head_init(ctx, &m->repr);
    module_dict_add(&m->heads, "lm", &m->lm.mod);
    module_dict_add(&m->heads, "repr", &m->repr.mod);
    module_add_child(&m->mod, "heads", &m->heads.mod);
}

typedef struct optimizer_manifest {
    int step;
    int n_groups;
} optimizer_manifest;

typedef struct scaler_manifest {
    float scale;
    int growth_tracker;
} scaler_manifest;

typedef struct scheduler_manifest {
    const char *kind;
    int step;
    float last_lr;
} scheduler_manifest;

static void print_checkpoint_manifest(const probe_param_set *set,
                                      const optimizer_manifest *opt,
                                      const scaler_manifest *scaler,
                                      const scheduler_manifest *sched)
{
    int i;
    int j;
    printf("checkpoint_manifest\n");
    printf("  model.params=%d\n", set->n);
    for (i = 0; i < set->n; ++i) {
        const probe_collected_param *p = &set->items[i];
        printf("    param %s dtype=%s group=%s aliases=%d\n",
               p->path,
               dtype_name(p->tensor->dtype),
               p->group != NULL ? p->group->name : "-",
               p->n_aliases);
        for (j = 0; j < p->n_aliases; ++j) {
            printf("      alias %s\n", p->aliases[j]);
        }
    }
    printf("  optimizer.step=%d groups=%d\n", opt->step, opt->n_groups);
    printf("  scaler.scale=%.1f growth_tracker=%d\n", (double)scaler->scale,
           scaler->growth_tracker);
    printf("  lr_scheduler.kind=%s step=%d last_lr=%.6f\n", sched->kind,
           sched->step, (double)sched->last_lr);
}

static void probe_arena_basics(void)
{
    probe_arena arena;
    probe_span a;
    probe_span b;
    probe_span fail;
    uint64_t gen_before;
    arena_init(&arena, PROBE_ARENA_SCRATCH, "probe", 1024U);
    CHECK(strcmp(arena_kind_name(arena.kind), "scratch") == 0, "arena kind name");
    a = arena_alloc_span(&arena, 1U, 64U);
    b = arena_alloc_span(&arena, 7U, 32U);
    CHECK(((uintptr_t)a.ptr % 64U) == 0U && (a.offset % 64U) == 0U,
          "arena 64B aligned alloc and offset");
    CHECK(((uintptr_t)b.ptr % 32U) == 0U && (b.offset % 32U) == 0U,
          "arena 32B aligned alloc and offset");
    CHECK(arena.watermark >= arena.offset, "arena watermark tracks offset");
    {
        size_t offset_before_fail = arena.offset;
        CHECK(arena_try_alloc_span(&arena, 2048U, 8U, &fail) == PROBE_ALLOC_OOM,
              "arena OOM returns status");
        CHECK_EQ_U64(arena.offset, offset_before_fail, "arena OOM does not advance offset");
    }
    gen_before = arena.generation;
    arena_reset(&arena);
    CHECK_EQ_U64(arena.offset, 0U, "arena reset offset");
    CHECK_EQ_U64(arena.generation, gen_before + 1U, "arena reset bumps generation");
    CHECK(arena.watermark > 0U, "arena reset preserves watermark");
    arena_freeze(&arena);
    CHECK(arena_try_alloc_span(&arena, 8U, 8U, &fail) == PROBE_ALLOC_FROZEN,
          "frozen arena rejects allocation");
    CHECK_EQ_U64(arena.offset, 0U, "frozen allocation does not advance offset");
    arena_destroy(&arena);
}

static void probe_scope_ring_and_tensor_views(probe_context *ctx, probe_tensor *persistent_tensor)
{
    int step;
    uint64_t waits_before = ctx->backend.waits;
    int slots[5][2];
    uint64_t generations[5][2];
    for (step = 0; step < 4; ++step) {
        int64_t batch_sizes[2] = {2, 16};
        int64_t hidden_sizes[3] = {2, 14, 32};
        probe_tensor *tokens;
        probe_tensor *hidden;
        probe_tensor *suffix;
        probe_tensor *suffix_compact;
        probe_tensor *persistent_view;
        gd_begin(ctx, PROBE_SCOPE_TRAIN);
        slots[step][0] = ctx->scratch_slot;
        slots[step][1] = ctx->data_slot;
        generations[step][0] = ctx_scratch(ctx)->generation;
        generations[step][1] = ctx_data(ctx)->generation;
        tokens = tensor_empty(ctx_data(ctx), "batch.tokens", PROBE_I32, 2, batch_sizes);
        hidden = tensor_empty(ctx_scratch(ctx), "vlm.hidden", PROBE_F16, 3, hidden_sizes);
        suffix = tensor_slice(ctx_scratch(ctx), hidden, 1, 6, 8, "hidden.text_suffix");
        persistent_view = tensor_slice(ctx_scratch(ctx), persistent_tensor, 0, 1, 2, "persistent.param_view");
        CHECK(tokens != NULL, "data tensor allocated");
        tensor_check_live(tokens);
        tensor_check_live(hidden);
        tensor_check_live(suffix);
        CHECK(tokens->storage_arena->kind == PROBE_ARENA_DATA, "tokens stored in data arena");
        CHECK(hidden->storage_arena->kind == PROBE_ARENA_SCRATCH, "hidden stored in scratch arena");
        CHECK(persistent_view->header_arena->kind == PROBE_ARENA_SCRATCH,
              "persistent view header lives in scratch");
        CHECK(persistent_view->storage_arena->kind == PROBE_ARENA_PARAMS,
              "persistent view storage stays in params");
        CHECK((tensor_first_byte_offset(hidden) % 64U) == 0U, "hidden storage offset aligned");
        CHECK(tensor_is_contiguous(hidden), "hidden is compact");
        CHECK(!tensor_is_contiguous(suffix), "suffix slice is non-contiguous view");
        CHECK(suffix->allocation_offset == hidden->allocation_offset,
              "suffix shares base allocation");
        CHECK(suffix->allocation_nbytes == hidden->allocation_nbytes,
              "suffix shares base allocation span");
        CHECK(tensor_first_byte_offset(suffix) == tensor_first_byte_offset(hidden) + 6U * 32U * dtype_size(PROBE_F16),
              "suffix slice offset");
        suffix_compact = tensor_contiguous(ctx_scratch(ctx), suffix, "hidden.text_suffix.contiguous");
        tensor_check_live(suffix_compact);
        CHECK(tensor_is_contiguous(suffix_compact), "explicit contiguous makes compact tensor");
        CHECK(!suffix_compact->is_view, "contiguous tensor owns arena range");
        CHECK(suffix_compact->allocation_offset != hidden->allocation_offset,
              "contiguous allocates distinct storage");
        gd_end(ctx);
    }
    CHECK(slots[0][0] != slots[1][0], "ring uses different scratch slots before wait");
    CHECK(slots[0][1] != slots[1][1], "ring uses different data slots before wait");
    CHECK(generations[2][0] > generations[0][0], "scratch slot generation bumps before reuse");
    CHECK(generations[2][1] > generations[0][1], "data slot generation bumps before reuse");
    CHECK(ctx->backend.waits >= waits_before + 2U, "ring waits after exhausting scratch/data slots");

    fake_backend_complete_all(&ctx->backend);
    waits_before = ctx->backend.waits;
    gd_begin(ctx, PROBE_SCOPE_INFER);
    gd_end(ctx);
    CHECK_EQ_U64(ctx->backend.waits, waits_before, "completed ring begin does not wait");
}

static void probe_state_fence(probe_context *ctx)
{
    probe_state_object kv;
    size_t first_offset;
    uint64_t waits_before;
    bool relocated;
    state_object_create(ctx, &kv, "kv_cache", 4096U);
    state_object_check_live(ctx, &kv);
    CHECK((kv.offset % 64U) == 0U, "state object offset aligned");
    gd_begin(ctx, PROBE_SCOPE_INFER);
    gd_end(ctx);
    state_object_mark_used(&kv, ctx->last_scope_fence);
    CHECK(kv.last_use_fence > ctx->backend.completed_fence, "state object fence is in-flight");

    first_offset = kv.offset;
    waits_before = ctx->backend.waits;
    relocated = state_object_reset_or_realloc(ctx, &kv, kv.nbytes, true);
    CHECK(relocated, "in-flight state reset can relocate instead of wait");
    CHECK(ctx->backend.waits == waits_before, "state relocation avoids backend wait");
    CHECK(kv.offset != first_offset, "state relocation gets fresh block");
    CHECK_EQ_U64(kv.last_use_fence, 0U, "state relocation clears fence");

    gd_begin(ctx, PROBE_SCOPE_INFER);
    gd_end(ctx);
    state_object_mark_used(&kv, ctx->last_scope_fence);
    waits_before = ctx->backend.waits;
    state_object_reset(ctx, &kv);
    CHECK(ctx->backend.waits == waits_before + 1U, "state reset waits on in-flight fence");
    CHECK_EQ_U64(kv.last_use_fence, 0U, "state reset clears fence");
}

static void probe_module_tree_and_manifest(probe_context *ctx)
{
    fake_vlm *model;
    probe_param_group groups[3] = {
        {.name = "backbone", .match = "vlm.backbone.*", .lr_mult = 0.1f, .weight_decay = 0.01f},
        {.name = "lm", .match = "vlm.heads.lm.*", .lr_mult = 1.0f, .weight_decay = 0.0f},
        {.name = "repr", .match = "vlm.heads.repr.*", .lr_mult = 1.0f, .weight_decay = 0.01f},
    };
    probe_param_set set;
    const probe_collected_param *tied;
    optimizer_manifest opt = {.step = 17, .n_groups = 3};
    scaler_manifest scaler = {.scale = 1024.0f, .growth_tracker = 11};
    scheduler_manifest sched = {.kind = "cosine_warmup", .step = 17, .last_lr = 3.0e-4f};

    model = (fake_vlm *)arena_alloc(&ctx->params, sizeof(*model), 8U);
    fake_vlm_init(ctx, model);
    CHECK(model->backbone.blocks.n == 3, "ModuleList stores block count");
    CHECK(model->backbone.blocks.items[2] == &model->backbone.block[2].mod,
          "ModuleList keeps typed item mapping");

    module_set_training(&model->mod, false);
    CHECK(!model->backbone.block[1].attn.mod.training, "recursive eval mode reaches blocks");
    module_set_training(&model->mod, true);
    CHECK(model->repr.proj.mod.training, "recursive train mode reaches ModuleDict heads");

    collect_module_params(&model->mod, groups, 3, &set);
    CHECK(param_set_has_path(&set, "vlm.backbone.image_proj.weight"),
          "collected image_proj weight path");
    CHECK(param_set_has_path(&set, "vlm.backbone.blocks.0.attn.wq.weight"),
          "collected ModuleList numeric path");
    CHECK(param_set_has_path(&set, "vlm.heads.lm.weight"),
          "collected LM head alias path");
    CHECK(param_set_has_path(&set, "vlm.heads.repr.proj.weight"),
          "collected ModuleDict repr head path");

    tied = param_set_find_path(&set, "vlm.heads.lm.weight");
    CHECK(tied != NULL, "tied LM path resolves");
    CHECK(tied->tensor == model->backbone.tok_emb_weight,
          "LM head weight tied to token embedding tensor");
    CHECK(strcmp(tied->path, "vlm.backbone.tok_emb.weight") == 0,
          "tied weight canonical path is first owner");
    CHECK(tied->n_aliases == 1, "tied weight records alias once");
    CHECK(tied->group != NULL && strcmp(tied->group->name, "backbone") == 0,
          "tied weight group follows canonical owner path");

    print_checkpoint_manifest(&set, &opt, &scaler, &sched);
}

static void probe_params_freeze(probe_context *ctx)
{
    probe_span fail;
    CHECK(ctx != NULL, "probe_params_freeze ctx");
    arena_freeze(&ctx->params);
    CHECK(arena_try_alloc_span(&ctx->params, 8U, 8U, &fail) == PROBE_ALLOC_FROZEN,
          "sealed params arena rejects late allocation");
}

int main(void)
{
    probe_context ctx;
    probe_tensor *persistent_probe_tensor;
    int64_t persistent_sizes[2] = {4, 4};
    printf("v2_foundation_probe: start\n");
    uint64_t heap_allocs_before_hot_path;
    probe_arena_basics();
    ctx_init(&ctx);
    probe_module_tree_and_manifest(&ctx);
    persistent_probe_tensor = tensor_empty(&ctx.params, "debug.persistent_tensor", PROBE_F16, 2, persistent_sizes);
    probe_params_freeze(&ctx);
    heap_allocs_before_hot_path = g_probe_heap_allocs;
    g_probe_heap_forbidden = true;
    probe_scope_ring_and_tensor_views(&ctx, persistent_probe_tensor);
    probe_state_fence(&ctx);
    g_probe_heap_forbidden = false;
    CHECK_EQ_U64(g_probe_heap_allocs, heap_allocs_before_hot_path,
                 "no probe heap allocation in scope hot path");
    printf("arena_watermarks params=%zu state=%zu scratch0=%zu scratch1=%zu data0=%zu data1=%zu waits=%" PRIu64 "\n",
           ctx.params.watermark, ctx.state.watermark,
           ctx.scratch.slots[0].watermark, ctx.scratch.slots[1].watermark,
           ctx.data.slots[0].watermark, ctx.data.slots[1].watermark,
           ctx.backend.waits);
    ctx_destroy(&ctx);
    printf("v2_foundation_probe: ok\n");
    return 0;
}
