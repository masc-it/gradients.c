#include <gradients/memory.h>

#include "backend.h"
#include "memory_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GD_MAX_RING_SLOTS 64U
#define GD_MAX_TOUCHED_STATE 128U
#define GD_MAX_PENDING_FENCES 1024U
#define GD_MAX_SUBALLOC_ALIGNMENT 4096U
#define GD_DEFAULT_ALIGNMENT 256U

typedef struct gd_arena {
    gd_arena_kind kind;
    int32_t slot;
    gd_backend_buffer *buffer;
    unsigned char *base;
    size_t capacity;
    size_t offset;
    size_t watermark;
    size_t default_alignment;
    uint64_t generation;
    bool sealed;
} gd_arena;

typedef struct gd_ring_arena {
    gd_arena *slots;
    uint64_t *slot_fences;
    uint32_t n_slots;
    int32_t current;
    uint64_t waits;
} gd_ring_arena;

typedef struct gd_pending_fence {
    uint64_t sequence;
    gd_backend_fence fence;
} gd_pending_fence;

typedef struct gd_backend_runtime {
    gd_backend *backend;
    uint64_t next_fence;
    uint64_t completed_fence;
    uint64_t waits;
    gd_pending_fence pending[GD_MAX_PENDING_FENCES];
} gd_backend_runtime;

struct gd_context {
    gd_arena params;
    gd_arena state;
    gd_ring_arena scratch;
    gd_ring_arena data;
    gd_backend_runtime backend;
    gd_scope_mode mode;
    bool in_scope;
    uint64_t last_scope_fence;
    gd_state_object *touched_state[GD_MAX_TOUCHED_STATE];
    uint32_t touched_state_count;
    gd_status status;
    char error[160];
};

static uint64_t gd_global_heap_allocs;
static bool gd_global_heap_forbidden;

static bool gd_is_power_of_two(size_t v)
{
    return v != 0U && (v & (v - 1U)) == 0U;
}

static gd_status gd_set_error(gd_context *ctx, gd_status status, const char *message)
{
    return gd_context_set_error(ctx, status, message);
}

gd_status gd_context_set_error(gd_context *ctx, gd_status status, const char *message)
{
    if (ctx != NULL) {
        ctx->status = status;
        snprintf(ctx->error, sizeof(ctx->error), "%s",
                 message != NULL ? message : gd_status_string(status));
    }
    return status;
}

gd_backend *gd_context_backend(gd_context *ctx)
{
    return ctx != NULL ? ctx->backend.backend : NULL;
}

static void *gd_heap_alloc(size_t nbytes)
{
    void *ptr;
    if (gd_global_heap_forbidden || nbytes == 0U) {
        return NULL;
    }
    ptr = calloc(1U, nbytes);
    if (ptr != NULL) {
        gd_global_heap_allocs += 1U;
    }
    return ptr;
}

static bool gd_align_up_size(size_t value, size_t alignment, size_t *out)
{
    if (value > SIZE_MAX - (alignment - 1U)) {
        return false;
    }
    *out = (value + alignment - 1U) & ~(alignment - 1U);
    return true;
}

static size_t gd_normalize_alignment(const gd_arena *arena, size_t alignment)
{
    return alignment == 0U ? arena->default_alignment : alignment;
}

static gd_status gd_arena_init(gd_backend *backend,
                               gd_arena *arena,
                               gd_arena_kind kind,
                               int32_t slot,
                               size_t capacity,
                               size_t default_alignment)
{
    gd_status st;
    memset(arena, 0, sizeof(*arena));
    if (backend == NULL || capacity == 0U || !gd_is_power_of_two(default_alignment) ||
        default_alignment > GD_MAX_SUBALLOC_ALIGNMENT) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_backend_buffer_create(backend, capacity, &arena->buffer);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_backend_buffer_is_host_visible(arena->buffer)) {
        gd_backend_buffer_destroy(arena->buffer);
        memset(arena, 0, sizeof(*arena));
        return GD_ERR_UNSUPPORTED;
    }
    arena->base = (unsigned char *)gd_backend_buffer_host_ptr(arena->buffer);
    if (arena->base == NULL || gd_backend_buffer_nbytes(arena->buffer) < capacity) {
        gd_backend_buffer_destroy(arena->buffer);
        memset(arena, 0, sizeof(*arena));
        return GD_ERR_INTERNAL;
    }
    arena->kind = kind;
    arena->slot = slot;
    arena->capacity = capacity;
    arena->default_alignment = default_alignment;
    arena->generation = 1U;
    return GD_OK;
}

static void gd_arena_destroy(gd_arena *arena)
{
    if (arena == NULL) {
        return;
    }
    gd_backend_buffer_destroy(arena->buffer);
    memset(arena, 0, sizeof(*arena));
}

static void gd_arena_reset(gd_arena *arena)
{
    arena->offset = 0U;
    arena->generation += 1U;
    if (arena->generation == 0U) {
        arena->generation = 1U;
    }
}

static gd_status gd_arena_alloc(gd_context *ctx,
                                gd_arena *arena,
                                size_t nbytes,
                                size_t alignment,
                                gd_span *out)
{
    size_t off;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
        out->slot = -1;
    }
    if (ctx == NULL || arena == NULL || out == NULL || nbytes == 0U) {
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "arena allocation invalid argument");
    }
    if (arena->sealed) {
        return gd_set_error(ctx, GD_ERR_FROZEN, "arena is sealed");
    }
    alignment = gd_normalize_alignment(arena, alignment);
    if (!gd_is_power_of_two(alignment) || alignment > GD_MAX_SUBALLOC_ALIGNMENT) {
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "arena allocation invalid alignment");
    }
    if (!gd_align_up_size(arena->offset, alignment, &off)) {
        return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "arena offset overflow");
    }
    if (off > arena->capacity || nbytes > arena->capacity - off) {
        return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "arena out of memory");
    }
    arena->offset = off + nbytes;
    if (arena->offset > arena->watermark) {
        arena->watermark = arena->offset;
    }
    out->arena = arena->kind;
    out->slot = arena->slot;
    out->offset = off;
    out->nbytes = nbytes;
    out->alignment = alignment;
    out->generation = arena->generation;
    out->buffer = arena->buffer;
    out->host_ptr = arena->base + off;
    return GD_OK;
}

static void gd_pending_release_completed(gd_context *ctx)
{
    uint32_t i;
    for (i = 0; i < GD_MAX_PENDING_FENCES; ++i) {
        gd_pending_fence *pending = &ctx->backend.pending[i];
        if (pending->sequence != 0U && pending->sequence <= ctx->backend.completed_fence) {
            gd_backend_fence_destroy(&pending->fence);
            pending->sequence = 0U;
        }
    }
}

static gd_backend_fence *gd_pending_find(gd_context *ctx, uint64_t sequence)
{
    uint32_t i;
    for (i = 0; i < GD_MAX_PENDING_FENCES; ++i) {
        if (ctx->backend.pending[i].sequence == sequence) {
            return &ctx->backend.pending[i].fence;
        }
    }
    return NULL;
}

static gd_status gd_pending_add(gd_context *ctx, uint64_t sequence, gd_backend_fence *fence)
{
    uint32_t i;
    gd_pending_release_completed(ctx);
    for (i = 0; i < GD_MAX_PENDING_FENCES; ++i) {
        if (ctx->backend.pending[i].sequence == 0U) {
            ctx->backend.pending[i].sequence = sequence;
            ctx->backend.pending[i].fence = *fence;
            fence->handle = NULL;
            return GD_OK;
        }
    }
    return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "pending fence table full");
}

static bool gd_fence_sequence_is_complete(gd_context *ctx, uint64_t sequence)
{
    gd_backend_fence *fence;
    if (sequence == 0U || sequence <= ctx->backend.completed_fence) {
        return true;
    }
    fence = gd_pending_find(ctx, sequence);
    if (fence == NULL) {
        return false;
    }
    if (gd_backend_fence_is_complete(fence)) {
        ctx->backend.completed_fence = sequence;
        gd_pending_release_completed(ctx);
        return true;
    }
    return false;
}

static gd_status gd_backend_wait_until(gd_context *ctx, uint64_t sequence)
{
    gd_backend_fence *fence;
    gd_status st;
    if (sequence == 0U || sequence <= ctx->backend.completed_fence) {
        return GD_OK;
    }
    fence = gd_pending_find(ctx, sequence);
    if (fence == NULL) {
        return gd_set_error(ctx, GD_ERR_INTERNAL, "missing backend fence");
    }
    st = gd_backend_fence_wait(fence);
    if (st != GD_OK) {
        return gd_set_error(ctx, st, "backend fence wait failed");
    }
    ctx->backend.completed_fence = sequence;
    ctx->backend.waits += 1U;
    gd_pending_release_completed(ctx);
    return GD_OK;
}

gd_status gd_context_synchronize(gd_context *ctx)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (ctx->backend.next_fence == 0U ||
        gd_fence_sequence_is_complete(ctx, ctx->backend.next_fence)) {
        return GD_OK;
    }
    return gd_backend_wait_until(ctx, ctx->backend.next_fence);
}

static gd_status gd_backend_record(gd_context *ctx, uint64_t *out_sequence)
{
    gd_backend_fence fence;
    gd_status st;
    uint64_t sequence;
    memset(&fence, 0, sizeof(fence));
    if (ctx == NULL || out_sequence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_backend_record_fence(ctx->backend.backend, &fence);
    if (st != GD_OK) {
        return gd_set_error(ctx, st, "backend fence record failed");
    }
    sequence = ctx->backend.next_fence + 1U;
    if (sequence == 0U) {
        gd_backend_fence_destroy(&fence);
        return gd_set_error(ctx, GD_ERR_INTERNAL, "backend fence sequence overflow");
    }
    st = gd_pending_add(ctx, sequence, &fence);
    if (st != GD_OK) {
        gd_backend_fence_destroy(&fence);
        return st;
    }
    ctx->backend.next_fence = sequence;
    *out_sequence = sequence;
    return GD_OK;
}

static void gd_ring_destroy(gd_ring_arena *ring)
{
    uint32_t i;
    if (ring == NULL) {
        return;
    }
    for (i = 0; i < ring->n_slots; ++i) {
        gd_arena_destroy(&ring->slots[i]);
    }
    free(ring->slots);
    free(ring->slot_fences);
    memset(ring, 0, sizeof(*ring));
    ring->current = -1;
}

static gd_status gd_ring_init(gd_backend *backend,
                              gd_ring_arena *ring,
                              gd_arena_kind kind,
                              uint32_t n_slots,
                              size_t slot_capacity,
                              size_t default_alignment)
{
    uint32_t i;
    memset(ring, 0, sizeof(*ring));
    ring->current = -1;
    if ((kind != GD_ARENA_SCRATCH && kind != GD_ARENA_DATA) || n_slots == 0U ||
        n_slots > GD_MAX_RING_SLOTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    ring->slots = (gd_arena *)gd_heap_alloc((size_t)n_slots * sizeof(*ring->slots));
    ring->slot_fences = (uint64_t *)gd_heap_alloc((size_t)n_slots * sizeof(*ring->slot_fences));
    if (ring->slots == NULL || ring->slot_fences == NULL) {
        gd_ring_destroy(ring);
        return GD_ERR_OUT_OF_MEMORY;
    }
    ring->n_slots = n_slots;
    for (i = 0; i < n_slots; ++i) {
        gd_status st = gd_arena_init(backend, &ring->slots[i], kind, (int32_t)i,
                                     slot_capacity, default_alignment);
        if (st != GD_OK) {
            gd_ring_destroy(ring);
            return st;
        }
    }
    return GD_OK;
}

static int32_t gd_ring_oldest_busy_slot(const gd_ring_arena *ring)
{
    uint32_t i;
    int32_t best = -1;
    uint64_t best_fence = UINT64_MAX;
    for (i = 0; i < ring->n_slots; ++i) {
        uint64_t fence = ring->slot_fences[i];
        if (fence != 0U && fence < best_fence) {
            best_fence = fence;
            best = (int32_t)i;
        }
    }
    return best;
}

static gd_status gd_ring_select(gd_context *ctx, gd_ring_arena *ring)
{
    uint32_t attempt;
    uint32_t start;
    if (ring->current < 0) {
        start = 0U;
    } else {
        start = ((uint32_t)ring->current + 1U) % ring->n_slots;
    }
    for (attempt = 0; attempt < ring->n_slots; ++attempt) {
        uint32_t idx = (start + attempt) % ring->n_slots;
        if (gd_fence_sequence_is_complete(ctx, ring->slot_fences[idx])) {
            ring->current = (int32_t)idx;
            ring->slot_fences[idx] = 0U;
            gd_arena_reset(&ring->slots[idx]);
            return GD_OK;
        }
    }
    {
        int32_t oldest = gd_ring_oldest_busy_slot(ring);
        gd_status st;
        if (oldest < 0) {
            return gd_set_error(ctx, GD_ERR_INTERNAL, "ring exhausted without busy slot");
        }
        st = gd_backend_wait_until(ctx, ring->slot_fences[oldest]);
        if (st != GD_OK) {
            return st;
        }
        ring->waits += 1U;
        ring->current = oldest;
        ring->slot_fences[oldest] = 0U;
        gd_arena_reset(&ring->slots[oldest]);
    }
    return GD_OK;
}

static gd_arena *gd_ring_current_arena(gd_ring_arena *ring)
{
    if (ring == NULL || ring->current < 0) {
        return NULL;
    }
    return &ring->slots[ring->current];
}

static const gd_arena *gd_context_span_arena(const gd_context *ctx, const gd_span *span)
{
    if (ctx == NULL || span == NULL) {
        return NULL;
    }
    switch (span->arena) {
    case GD_ARENA_PARAMS:
        return span->slot == -1 ? &ctx->params : NULL;
    case GD_ARENA_STATE:
        return span->slot == -1 ? &ctx->state : NULL;
    case GD_ARENA_SCRATCH:
        if (span->slot < 0 || (uint32_t)span->slot >= ctx->scratch.n_slots) {
            return NULL;
        }
        return &ctx->scratch.slots[span->slot];
    case GD_ARENA_DATA:
        if (span->slot < 0 || (uint32_t)span->slot >= ctx->data.n_slots) {
            return NULL;
        }
        return &ctx->data.slots[span->slot];
    default:
        return NULL;
    }
}

bool gd_context_span_is_live(const gd_context *ctx, const gd_span *span)
{
    const gd_arena *arena = gd_context_span_arena(ctx, span);
    if (arena == NULL || span == NULL || span->nbytes == 0U) {
        return false;
    }
    if (span->generation != arena->generation || span->buffer != arena->buffer) {
        return false;
    }
    if (span->offset > arena->capacity || span->nbytes > arena->capacity - span->offset) {
        return false;
    }
    if (arena->base != NULL && span->host_ptr != arena->base + span->offset) {
        return false;
    }
    return true;
}

gd_status gd_context_validate_span(gd_context *ctx, const gd_span *span, const char *message)
{
    if (ctx == NULL || span == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_context_span_is_live(ctx, span)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE,
                            message != NULL ? message : "span is stale");
    }
    return GD_OK;
}

gd_status gd_context_wait_for_span(gd_context *ctx, const gd_span *span)
{
    gd_status st;
    uint64_t fence = 0U;
    if (ctx == NULL || span == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_context_validate_span(ctx, span, "span is stale");
    if (st != GD_OK) {
        return st;
    }
    if (span->arena == GD_ARENA_SCRATCH) {
        fence = ctx->scratch.slot_fences[(uint32_t)span->slot];
    } else if (span->arena == GD_ARENA_DATA) {
        fence = ctx->data.slot_fences[(uint32_t)span->slot];
    } else {
        return gd_context_synchronize(ctx);
    }
    if (fence == 0U || gd_fence_sequence_is_complete(ctx, fence)) {
        return GD_OK;
    }
    return gd_backend_wait_until(ctx, fence);
}

static bool gd_state_object_live(const gd_context *ctx, const gd_state_object *object)
{
    if (ctx == NULL || object == NULL || object->span.arena != GD_ARENA_STATE ||
        object->span.generation != ctx->state.generation || object->span.buffer != ctx->state.buffer) {
        return false;
    }
    return object->span.offset <= ctx->state.capacity &&
           object->span.nbytes <= ctx->state.capacity - object->span.offset;
}

static gd_status gd_alloc_from_ring(gd_context *ctx,
                                    gd_ring_arena *ring,
                                    size_t nbytes,
                                    size_t alignment,
                                    gd_span *out,
                                    const char *message)
{
    gd_arena *arena;
    if (ctx == NULL || !ctx->in_scope) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, message);
    }
    arena = gd_ring_current_arena(ring);
    if (arena == NULL) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "ring has no current slot");
    }
    return gd_arena_alloc(ctx, arena, nbytes, alignment, out);
}

gd_memory_config gd_memory_config_default(void)
{
    gd_memory_config config;
    config.params_bytes = 128U * 1024U * 1024U;
    config.state_bytes = 64U * 1024U * 1024U;
    config.scratch_slot_bytes = 64U * 1024U * 1024U;
    config.data_slot_bytes = 8U * 1024U * 1024U;
    config.scratch_slots = 3U;
    config.data_slots = 3U;
    config.default_alignment = GD_DEFAULT_ALIGNMENT;
    return config;
}

gd_status gd_context_create(const gd_memory_config *config, gd_context **out_ctx)
{
    gd_memory_config cfg;
    gd_context *ctx;
    gd_backend *backend = NULL;
    gd_status st;
    size_t default_alignment;
    if (out_ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_ctx = NULL;
    cfg = config != NULL ? *config : gd_memory_config_default();
    default_alignment = cfg.default_alignment == 0U ? GD_DEFAULT_ALIGNMENT : cfg.default_alignment;
    if (cfg.params_bytes == 0U || cfg.state_bytes == 0U || cfg.scratch_slot_bytes == 0U ||
        cfg.data_slot_bytes == 0U || cfg.scratch_slots == 0U || cfg.data_slots == 0U ||
        cfg.scratch_slots > GD_MAX_RING_SLOTS || cfg.data_slots > GD_MAX_RING_SLOTS ||
        !gd_is_power_of_two(default_alignment) || default_alignment > GD_MAX_SUBALLOC_ALIGNMENT) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_backend_create_default(&backend);
    if (st != GD_OK) {
        return st;
    }
    ctx = (gd_context *)gd_heap_alloc(sizeof(*ctx));
    if (ctx == NULL) {
        gd_backend_destroy(backend);
        return GD_ERR_OUT_OF_MEMORY;
    }
    ctx->backend.backend = backend;
    ctx->scratch.current = -1;
    ctx->data.current = -1;
    ctx->mode = GD_SCOPE_NONE;
    st = gd_arena_init(backend, &ctx->params, GD_ARENA_PARAMS, -1,
                       cfg.params_bytes, default_alignment);
    if (st != GD_OK) {
        gd_context_destroy(ctx);
        return st;
    }
    st = gd_arena_init(backend, &ctx->state, GD_ARENA_STATE, -1,
                       cfg.state_bytes, default_alignment);
    if (st != GD_OK) {
        gd_context_destroy(ctx);
        return st;
    }
    st = gd_ring_init(backend, &ctx->scratch, GD_ARENA_SCRATCH, cfg.scratch_slots,
                      cfg.scratch_slot_bytes, default_alignment);
    if (st != GD_OK) {
        gd_context_destroy(ctx);
        return st;
    }
    st = gd_ring_init(backend, &ctx->data, GD_ARENA_DATA, cfg.data_slots,
                      cfg.data_slot_bytes, default_alignment);
    if (st != GD_OK) {
        gd_context_destroy(ctx);
        return st;
    }
    *out_ctx = ctx;
    return GD_OK;
}

void gd_context_destroy(gd_context *ctx)
{
    uint32_t i;
    if (ctx == NULL) {
        return;
    }
    gd_debug_complete_all(ctx);
    for (i = 0; i < GD_MAX_PENDING_FENCES; ++i) {
        gd_backend_fence_destroy(&ctx->backend.pending[i].fence);
    }
    gd_ring_destroy(&ctx->scratch);
    gd_ring_destroy(&ctx->data);
    gd_arena_destroy(&ctx->params);
    gd_arena_destroy(&ctx->state);
    gd_backend_destroy(ctx->backend.backend);
    free(ctx);
}

gd_status gd_context_status(const gd_context *ctx)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return ctx->status;
}

const char *gd_context_error(const gd_context *ctx)
{
    if (ctx == NULL) {
        return "no context";
    }
    return ctx->error[0] != '\0' ? ctx->error : gd_status_string(ctx->status);
}

void gd_context_clear_error(gd_context *ctx)
{
    if (ctx != NULL) {
        ctx->status = GD_OK;
        ctx->error[0] = '\0';
    }
}

gd_status gd_context_seal_params(gd_context *ctx)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    ctx->params.sealed = true;
    return GD_OK;
}

gd_status gd_begin(gd_context *ctx, gd_scope_mode mode)
{
    gd_status st;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (ctx->in_scope || (mode != GD_SCOPE_TRAIN && mode != GD_SCOPE_EVAL && mode != GD_SCOPE_INFER)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "invalid gd_begin state");
    }
    st = gd_ring_select(ctx, &ctx->scratch);
    if (st != GD_OK) {
        return st;
    }
    st = gd_ring_select(ctx, &ctx->data);
    if (st != GD_OK) {
        return st;
    }
    ctx->mode = mode;
    ctx->in_scope = true;
    ctx->touched_state_count = 0U;
    return GD_OK;
}

gd_status gd_end(gd_context *ctx)
{
    uint32_t i;
    uint64_t fence;
    gd_status st;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->in_scope || ctx->scratch.current < 0 || ctx->data.current < 0) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "invalid gd_end state");
    }
    st = gd_backend_record(ctx, &fence);
    if (st != GD_OK) {
        return st;
    }
    ctx->scratch.slot_fences[ctx->scratch.current] = fence;
    ctx->data.slot_fences[ctx->data.current] = fence;
    for (i = 0; i < ctx->touched_state_count; ++i) {
        ctx->touched_state[i]->last_use_fence = fence;
    }
    ctx->touched_state_count = 0U;
    ctx->last_scope_fence = fence;
    ctx->mode = GD_SCOPE_NONE;
    ctx->in_scope = false;
    return GD_OK;
}

gd_status gd_alloc_params(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_arena_alloc(ctx, &ctx->params, nbytes, alignment, out);
}

gd_status gd_alloc_state(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_arena_alloc(ctx, &ctx->state, nbytes, alignment, out);
}

gd_status gd_alloc_scratch(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    return gd_alloc_from_ring(ctx, &ctx->scratch, nbytes, alignment, out,
                              "scratch allocation requires active scope");
}

gd_status gd_alloc_data(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    return gd_alloc_from_ring(ctx, &ctx->data, nbytes, alignment, out,
                              "data allocation requires active scope");
}

gd_status gd_context_alloc_span(gd_context *ctx,
                                gd_arena_kind arena,
                                size_t nbytes,
                                size_t alignment,
                                gd_span *out)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    switch (arena) {
    case GD_ARENA_PARAMS:
        return gd_alloc_params(ctx, nbytes, alignment, out);
    case GD_ARENA_STATE:
        return gd_alloc_state(ctx, nbytes, alignment, out);
    case GD_ARENA_SCRATCH:
        return gd_alloc_scratch(ctx, nbytes, alignment, out);
    case GD_ARENA_DATA:
        return gd_alloc_data(ctx, nbytes, alignment, out);
    default:
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid arena kind");
    }
}

gd_status gd_state_object_create(gd_context *ctx,
                                 size_t nbytes,
                                 size_t alignment,
                                 gd_state_object *out)
{
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_alloc_state(ctx, nbytes, alignment, &out->span);
    if (st != GD_OK) {
        return st;
    }
    out->last_use_fence = 0U;
    return GD_OK;
}

gd_status gd_state_touch(gd_context *ctx, gd_state_object *object)
{
    uint32_t i;
    if (ctx == NULL || object == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->in_scope) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "state touch requires active scope");
    }
    if (!gd_state_object_live(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "state object is stale");
    }
    for (i = 0; i < ctx->touched_state_count; ++i) {
        if (ctx->touched_state[i] == object) {
            return GD_OK;
        }
    }
    if (ctx->touched_state_count >= GD_MAX_TOUCHED_STATE) {
        return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "too many touched state objects");
    }
    ctx->touched_state[ctx->touched_state_count] = object;
    ctx->touched_state_count += 1U;
    return GD_OK;
}

gd_status gd_state_object_reset(gd_context *ctx,
                                gd_state_object *object,
                                size_t nbytes,
                                size_t alignment,
                                bool allow_realloc,
                                bool *out_relocated)
{
    gd_status st;
    bool need_fresh;
    gd_span fresh;
    if (out_relocated != NULL) {
        *out_relocated = false;
    }
    if (ctx == NULL || object == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_state_object_live(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "state object is stale");
    }
    if (nbytes == 0U) {
        nbytes = object->span.nbytes;
    }
    if (alignment == 0U) {
        alignment = object->span.alignment;
    }
    need_fresh = nbytes > object->span.nbytes ||
                 (alignment > object->span.alignment && (object->span.offset % alignment) != 0U);
    if (object->last_use_fence != 0U &&
        !gd_fence_sequence_is_complete(ctx, object->last_use_fence)) {
        if (allow_realloc) {
            st = gd_alloc_state(ctx, nbytes, alignment, &fresh);
            if (st != GD_OK) {
                return st;
            }
            object->span = fresh;
            object->last_use_fence = 0U;
            if (out_relocated != NULL) {
                *out_relocated = true;
            }
            return GD_OK;
        }
        st = gd_backend_wait_until(ctx, object->last_use_fence);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_fresh) {
        st = gd_alloc_state(ctx, nbytes, alignment, &fresh);
        if (st != GD_OK) {
            return st;
        }
        object->span = fresh;
        if (out_relocated != NULL) {
            *out_relocated = true;
        }
    }
    object->last_use_fence = 0U;
    return GD_OK;
}

static void gd_arena_stats_fill(const gd_arena *arena, gd_arena_stats *out)
{
    out->capacity = arena->capacity;
    out->offset = arena->offset;
    out->watermark = arena->watermark;
    out->generation = arena->generation;
    out->sealed = arena->sealed;
}

static void gd_ring_stats_fill(const gd_ring_arena *ring, gd_ring_stats *out)
{
    uint32_t i;
    memset(out, 0, sizeof(*out));
    out->slots = ring->n_slots;
    out->current_slot = ring->current;
    out->waits = ring->waits;
    for (i = 0; i < ring->n_slots; ++i) {
        out->total_capacity += ring->slots[i].capacity;
        out->total_watermark += ring->slots[i].watermark;
        if (ring->slots[i].watermark > out->max_slot_watermark) {
            out->max_slot_watermark = ring->slots[i].watermark;
        }
    }
}

gd_status gd_memory_stats_query(const gd_context *ctx, gd_memory_stats *out)
{
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    gd_arena_stats_fill(&ctx->params, &out->params);
    gd_arena_stats_fill(&ctx->state, &out->state);
    gd_ring_stats_fill(&ctx->scratch, &out->scratch);
    gd_ring_stats_fill(&ctx->data, &out->data);
    out->backend_waits = ctx->backend.waits;
    out->last_scope_fence = ctx->last_scope_fence;
    return GD_OK;
}

void gd_debug_complete_all(gd_context *ctx)
{
    uint32_t i;
    if (ctx == NULL) {
        return;
    }
    for (i = 0; i < GD_MAX_PENDING_FENCES; ++i) {
        if (ctx->backend.pending[i].sequence != 0U) {
            (void)gd_backend_fence_wait(&ctx->backend.pending[i].fence);
        }
    }
    ctx->backend.completed_fence = ctx->backend.next_fence;
    gd_pending_release_completed(ctx);
}

uint64_t gd_debug_heap_alloc_count(void)
{
    return gd_global_heap_allocs;
}

void gd_debug_set_heap_guard(bool forbid_heap_allocations)
{
    gd_global_heap_forbidden = forbid_heap_allocations;
}

int32_t gd_debug_current_ring_slot(const gd_context *ctx, gd_arena_kind arena)
{
    if (ctx == NULL) {
        return -1;
    }
    if (arena == GD_ARENA_SCRATCH) {
        return ctx->scratch.current;
    }
    if (arena == GD_ARENA_DATA) {
        return ctx->data.current;
    }
    return -1;
}

uint64_t gd_debug_ring_slot_generation(const gd_context *ctx,
                                       gd_arena_kind arena,
                                       uint32_t slot)
{
    const gd_ring_arena *ring;
    if (ctx == NULL) {
        return 0U;
    }
    ring = arena == GD_ARENA_SCRATCH ? &ctx->scratch :
           arena == GD_ARENA_DATA ? &ctx->data : NULL;
    if (ring == NULL || slot >= ring->n_slots) {
        return 0U;
    }
    return ring->slots[slot].generation;
}

uint64_t gd_debug_ring_slot_fence(const gd_context *ctx, gd_arena_kind arena, uint32_t slot)
{
    const gd_ring_arena *ring;
    if (ctx == NULL) {
        return 0U;
    }
    ring = arena == GD_ARENA_SCRATCH ? &ctx->scratch :
           arena == GD_ARENA_DATA ? &ctx->data : NULL;
    if (ring == NULL || slot >= ring->n_slots) {
        return 0U;
    }
    return ring->slot_fences[slot];
}
