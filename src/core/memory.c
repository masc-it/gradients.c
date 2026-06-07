#include <gradients/memory.h>

#include "backend.h"
#include "memory_internal.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GD_MAX_RING_SLOTS 64U
#define GD_MAX_TOUCHED_STATE 128U
#define GD_MAX_PENDING_FENCES 1024U
#define GD_MAX_SUBALLOC_ALIGNMENT 4096U
#define GD_DEFAULT_ALIGNMENT 256U
#define GD_ARENA_MAX_FREE_BLOCKS 16384U

typedef struct gd_free_block {
    size_t offset;
    size_t nbytes;
} gd_free_block;

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
    gd_free_block *free_blocks;
    uint32_t free_count;
    uint32_t free_capacity;
    bool sealed;
} gd_arena;

typedef enum gd_data_slot_state_internal {
    GD_DATA_SLOT_FREE_INTERNAL = 0,
    GD_DATA_SLOT_FILLING_INTERNAL = 1,
    GD_DATA_SLOT_READY_INTERNAL = 2,
    GD_DATA_SLOT_IN_STEP_INTERNAL = 3,
    GD_DATA_SLOT_RETIRED_INTERNAL = 4,
} gd_data_slot_state_internal;

typedef struct gd_ring_arena {
    gd_arena *slots;
    uint64_t *slot_fences;
    uint8_t *slot_states; /* Only used by the data ring. */
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

struct gd_state_object {
    gd_context *owner;
    gd_span span;
    uint64_t last_use_fence;
    uint64_t epoch;
    gd_state_object *prev;
    gd_state_object *next;
};

typedef struct gd_touched_state {
    gd_state_object *object;
    gd_span span;
    uint64_t epoch;
    gd_state_access access;
} gd_touched_state;

struct gd_context {
    gd_arena params;
    gd_arena state;
    gd_ring_arena scratch;
    gd_ring_arena data;
    gd_backend_runtime backend;
    gd_scope_mode mode;
    bool in_scope;
    uint64_t last_scope_fence;
    gd_state_object *state_objects;
    gd_touched_state touched_state[GD_MAX_TOUCHED_STATE];
    uint32_t touched_state_count;
    gd_autograd_state *autograd;
    uint64_t next_tensor_id;
    gd_status status;
    char error[160];
    pthread_mutex_t tensor_id_mutex;
    pthread_mutex_t backend_mutex;
    pthread_mutex_t data_mutex;
    pthread_cond_t data_cv;
    bool sync_ready;
    gd_batch *active_batch;
    int32_t active_data_slot;
    uint64_t active_data_generation;
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

gd_autograd_state *gd_context_autograd(gd_context *ctx)
{
    return ctx != NULL ? ctx->autograd : NULL;
}

const gd_autograd_state *gd_context_autograd_const(const gd_context *ctx)
{
    return ctx != NULL ? ctx->autograd : NULL;
}

gd_scope_mode gd_context_scope_mode(const gd_context *ctx)
{
    return ctx != NULL ? ctx->mode : GD_SCOPE_NONE;
}

bool gd_context_in_scope(const gd_context *ctx)
{
    return ctx != NULL && ctx->in_scope;
}

gd_status gd_context_next_tensor_id(gd_context *ctx, uint64_t *out_id)
{
    gd_status st = GD_OK;
    if (ctx == NULL || out_id == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&ctx->tensor_id_mutex);
    if (ctx->next_tensor_id == 0U || ctx->next_tensor_id == UINT64_MAX) {
        st = gd_set_error(ctx, GD_ERR_INTERNAL, "tensor id counter overflow");
    } else {
        *out_id = ctx->next_tensor_id;
        ctx->next_tensor_id += 1U;
    }
    pthread_mutex_unlock(&ctx->tensor_id_mutex);
    return st;
}

uint64_t gd_context_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0U;
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
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
    if (kind == GD_ARENA_SCRATCH) {
        arena->free_blocks = (gd_free_block *)gd_heap_alloc(GD_ARENA_MAX_FREE_BLOCKS *
                                                            sizeof(arena->free_blocks[0]));
        if (arena->free_blocks == NULL) {
            gd_backend_buffer_destroy(arena->buffer);
            memset(arena, 0, sizeof(*arena));
            return GD_ERR_OUT_OF_MEMORY;
        }
        arena->free_capacity = GD_ARENA_MAX_FREE_BLOCKS;
    }
    return GD_OK;
}

static void gd_arena_destroy(gd_arena *arena)
{
    if (arena == NULL) {
        return;
    }
    gd_backend_buffer_destroy(arena->buffer);
    free(arena->free_blocks);
    memset(arena, 0, sizeof(*arena));
}

static void gd_arena_reset(gd_arena *arena)
{
    arena->offset = 0U;
    arena->free_count = 0U;
    arena->generation += 1U;
    if (arena->generation == 0U) {
        arena->generation = 1U;
    }
}

static void gd_arena_remove_free_block(gd_arena *arena, uint32_t index)
{
    if (arena == NULL || index >= arena->free_count) {
        return;
    }
    if (index + 1U < arena->free_count) {
        memmove(&arena->free_blocks[index],
                &arena->free_blocks[index + 1U],
                (size_t)(arena->free_count - index - 1U) * sizeof(arena->free_blocks[0]));
    }
    arena->free_count -= 1U;
}

static gd_status gd_arena_alloc_from_free(gd_arena *arena,
                                          size_t nbytes,
                                          size_t alignment,
                                          size_t *out_offset)
{
    uint32_t i;
    uint32_t best = UINT32_MAX;
    size_t best_waste = SIZE_MAX;
    size_t best_aligned = 0U;
    if (arena == NULL || out_offset == NULL || arena->free_count == 0U) {
        return GD_ERR_BAD_STATE;
    }
    for (i = 0U; i < arena->free_count; ++i) {
        const gd_free_block *block = &arena->free_blocks[i];
        size_t aligned;
        size_t padding;
        size_t usable;
        size_t waste;
        if (!gd_align_up_size(block->offset, alignment, &aligned) || aligned < block->offset) {
            continue;
        }
        padding = aligned - block->offset;
        if (padding > block->nbytes) {
            continue;
        }
        usable = block->nbytes - padding;
        if (nbytes > usable) {
            continue;
        }
        waste = usable - nbytes;
        if (waste < best_waste) {
            best = i;
            best_waste = waste;
            best_aligned = aligned;
            if (waste == 0U) {
                break;
            }
        }
    }
    if (best == UINT32_MAX) {
        return GD_ERR_BAD_STATE;
    }
    {
        gd_free_block *block = &arena->free_blocks[best];
        const size_t block_start = block->offset;
        const size_t block_end = block->offset + block->nbytes;
        const size_t alloc_end = best_aligned + nbytes;
        const size_t prefix = best_aligned - block_start;
        const size_t suffix = block_end - alloc_end;
        if (prefix != 0U && suffix != 0U) {
            block->nbytes = prefix;
            if (arena->free_count < arena->free_capacity) {
                if (best + 1U < arena->free_count) {
                    memmove(&arena->free_blocks[best + 2U],
                            &arena->free_blocks[best + 1U],
                            (size_t)(arena->free_count - best - 1U) * sizeof(arena->free_blocks[0]));
                }
                arena->free_blocks[best + 1U].offset = alloc_end;
                arena->free_blocks[best + 1U].nbytes = suffix;
                arena->free_count += 1U;
            }
        } else if (prefix != 0U) {
            block->nbytes = prefix;
        } else if (suffix != 0U) {
            block->offset = alloc_end;
            block->nbytes = suffix;
        } else {
            gd_arena_remove_free_block(arena, best);
        }
    }
    *out_offset = best_aligned;
    return GD_OK;
}

static gd_status gd_arena_free(gd_context *ctx, gd_arena *arena, const gd_span *span)
{
    uint32_t index;
    size_t start;
    size_t end;
    if (ctx == NULL || arena == NULL || span == NULL || span->nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (arena->free_blocks == NULL || arena->free_capacity == 0U) {
        return gd_set_error(ctx, GD_ERR_UNSUPPORTED, "arena does not support suballocation free");
    }
    if (span->offset > arena->capacity || span->nbytes > arena->capacity - span->offset) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "free span outside arena");
    }
    start = span->offset;
    end = span->offset + span->nbytes;
    for (index = 0U; index < arena->free_count && arena->free_blocks[index].offset < start; ++index) {
    }
    if (index > 0U) {
        const gd_free_block *prev = &arena->free_blocks[index - 1U];
        if (prev->offset + prev->nbytes > start) {
            return gd_set_error(ctx, GD_ERR_BAD_STATE, "double free or overlapping free span");
        }
    }
    if (index < arena->free_count && end > arena->free_blocks[index].offset) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "double free or overlapping free span");
    }
    {
        const bool merge_prev = index > 0U &&
                                arena->free_blocks[index - 1U].offset +
                                        arena->free_blocks[index - 1U].nbytes == start;
        const bool merge_next = index < arena->free_count && end == arena->free_blocks[index].offset;
        if (merge_prev && merge_next) {
            arena->free_blocks[index - 1U].nbytes += span->nbytes + arena->free_blocks[index].nbytes;
            gd_arena_remove_free_block(arena, index);
        } else if (merge_prev) {
            arena->free_blocks[index - 1U].nbytes += span->nbytes;
        } else if (merge_next) {
            arena->free_blocks[index].offset = start;
            arena->free_blocks[index].nbytes += span->nbytes;
        } else {
            if (arena->free_count >= arena->free_capacity) {
                return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "arena free block table full");
            }
            if (index < arena->free_count) {
                memmove(&arena->free_blocks[index + 1U],
                        &arena->free_blocks[index],
                        (size_t)(arena->free_count - index) * sizeof(arena->free_blocks[0]));
            }
            arena->free_blocks[index].offset = start;
            arena->free_blocks[index].nbytes = span->nbytes;
            arena->free_count += 1U;
        }
    }
    return GD_OK;
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
    if (arena->free_count != 0U &&
        gd_arena_alloc_from_free(arena, nbytes, alignment, &off) == GD_OK) {
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

static bool gd_fence_sequence_is_complete_locked(gd_context *ctx, uint64_t sequence)
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

static bool gd_fence_sequence_is_complete(gd_context *ctx, uint64_t sequence)
{
    bool complete;
    pthread_mutex_lock(&ctx->backend_mutex);
    complete = gd_fence_sequence_is_complete_locked(ctx, sequence);
    pthread_mutex_unlock(&ctx->backend_mutex);
    return complete;
}

static gd_status gd_backend_wait_until(gd_context *ctx, uint64_t sequence)
{
    gd_backend_fence *fence;
    gd_status st = GD_OK;
    pthread_mutex_lock(&ctx->backend_mutex);
    if (sequence == 0U || sequence <= ctx->backend.completed_fence) {
        pthread_mutex_unlock(&ctx->backend_mutex);
        return GD_OK;
    }
    fence = gd_pending_find(ctx, sequence);
    if (fence == NULL) {
        st = gd_set_error(ctx, GD_ERR_INTERNAL, "missing backend fence");
        pthread_mutex_unlock(&ctx->backend_mutex);
        return st;
    }
    st = gd_backend_fence_wait(fence);
    if (st != GD_OK) {
        st = gd_set_error(ctx, st, "backend fence wait failed");
        pthread_mutex_unlock(&ctx->backend_mutex);
        return st;
    }
    ctx->backend.completed_fence = sequence;
    ctx->backend.waits += 1U;
    gd_pending_release_completed(ctx);
    pthread_mutex_unlock(&ctx->backend_mutex);
    return GD_OK;
}

gd_status gd_context_flush_backend(gd_context *ctx)
{
    gd_status st;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_backend_flush(ctx->backend.backend);
    if (st != GD_OK) {
        return gd_set_error(ctx, st, "backend flush failed");
    }
    return GD_OK;
}

gd_status gd_context_synchronize(gd_context *ctx)
{
    gd_status st;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_context_flush_backend(ctx);
    if (st != GD_OK) {
        return st;
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
    pthread_mutex_lock(&ctx->backend_mutex);
    st = gd_backend_record_fence(ctx->backend.backend, &fence);
    if (st != GD_OK) {
        pthread_mutex_unlock(&ctx->backend_mutex);
        return gd_set_error(ctx, st, "backend fence record failed");
    }
    sequence = ctx->backend.next_fence + 1U;
    if (sequence == 0U) {
        gd_backend_fence_destroy(&fence);
        pthread_mutex_unlock(&ctx->backend_mutex);
        return gd_set_error(ctx, GD_ERR_INTERNAL, "backend fence sequence overflow");
    }
    st = gd_pending_add(ctx, sequence, &fence);
    if (st != GD_OK) {
        gd_backend_fence_destroy(&fence);
        pthread_mutex_unlock(&ctx->backend_mutex);
        return st;
    }
    ctx->backend.next_fence = sequence;
    *out_sequence = sequence;
    pthread_mutex_unlock(&ctx->backend_mutex);
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
    free(ring->slot_states);
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
    if (kind == GD_ARENA_DATA) {
        ring->slot_states = (uint8_t *)gd_heap_alloc((size_t)n_slots * sizeof(*ring->slot_states));
    }
    if (ring->slots == NULL || ring->slot_fences == NULL ||
        (kind == GD_ARENA_DATA && ring->slot_states == NULL)) {
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

gd_status gd_context_free_span(gd_context *ctx, const gd_span *span)
{
    gd_arena *arena;
    if (ctx == NULL || span == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (span->arena != GD_ARENA_SCRATCH) {
        return GD_OK;
    }
    if (!gd_context_span_is_live(ctx, span)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "free span is stale");
    }
    if (span->slot < 0 || (uint32_t)span->slot >= ctx->scratch.n_slots) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "invalid scratch free slot");
    }
    arena = &ctx->scratch.slots[(uint32_t)span->slot];
    return gd_arena_free(ctx, arena, span);
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

static bool gd_state_access_valid(gd_state_access access)
{
    return access == GD_STATE_READ || access == GD_STATE_WRITE ||
           access == GD_STATE_READ_WRITE;
}

static bool gd_state_object_registered(const gd_context *ctx,
                                       const gd_state_object *object)
{
    const gd_state_object *cursor;
    if (ctx == NULL || object == NULL) {
        return false;
    }
    for (cursor = ctx->state_objects; cursor != NULL; cursor = cursor->next) {
        if (cursor == object) {
            return true;
        }
    }
    return false;
}

static bool gd_state_object_live(const gd_context *ctx, const gd_state_object *object)
{
    if (ctx == NULL || object == NULL || object->owner != ctx ||
        !gd_state_object_registered(ctx, object) || object->span.arena != GD_ARENA_STATE ||
        object->span.generation != ctx->state.generation || object->span.buffer != ctx->state.buffer) {
        return false;
    }
    return object->span.offset <= ctx->state.capacity &&
           object->span.nbytes <= ctx->state.capacity - object->span.offset;
}

static int32_t gd_state_touched_find(const gd_context *ctx,
                                     const gd_state_object *object)
{
    uint32_t i;
    if (ctx == NULL || object == NULL) {
        return -1;
    }
    for (i = 0; i < ctx->touched_state_count; ++i) {
        if (ctx->touched_state[i].object == object) {
            return (int32_t)i;
        }
    }
    return -1;
}

static bool gd_state_object_touched(const gd_context *ctx,
                                    const gd_state_object *object)
{
    return gd_state_touched_find(ctx, object) >= 0;
}

static void gd_state_object_bump_epoch(gd_state_object *object)
{
    object->epoch += 1U;
    if (object->epoch == 0U) {
        object->epoch = 1U;
    }
}

static void gd_state_object_link(gd_context *ctx, gd_state_object *object)
{
    object->owner = ctx;
    object->prev = NULL;
    object->next = ctx->state_objects;
    if (ctx->state_objects != NULL) {
        ctx->state_objects->prev = object;
    }
    ctx->state_objects = object;
}

static void gd_state_object_unlink(gd_context *ctx, gd_state_object *object)
{
    if (object->prev != NULL) {
        object->prev->next = object->next;
    } else if (ctx->state_objects == object) {
        ctx->state_objects = object->next;
    }
    if (object->next != NULL) {
        object->next->prev = object->prev;
    }
    object->owner = NULL;
    object->prev = NULL;
    object->next = NULL;
}

static gd_status gd_alloc_from_ring(gd_context *ctx,
                                    gd_ring_arena *ring,
                                    size_t nbytes,
                                    size_t alignment,
                                    gd_span *out,
                                    const char *message)
{
    gd_arena *arena;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
        out->slot = -1;
    }
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
    ctx->active_data_slot = -1;
    ctx->next_tensor_id = 1U;
    if (pthread_mutex_init(&ctx->tensor_id_mutex, NULL) != 0 ||
        pthread_mutex_init(&ctx->backend_mutex, NULL) != 0 ||
        pthread_mutex_init(&ctx->data_mutex, NULL) != 0 ||
        pthread_cond_init(&ctx->data_cv, NULL) != 0) {
        gd_backend_destroy(backend);
        free(ctx);
        return GD_ERR_INTERNAL;
    }
    ctx->sync_ready = true;
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
    st = gd_autograd_state_create(ctx, &ctx->autograd);
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
    while (ctx->state_objects != NULL) {
        gd_state_object *object = ctx->state_objects;
        ctx->state_objects = object->next;
        free(object);
    }
    gd_autograd_state_destroy(ctx->autograd);
    ctx->autograd = NULL;
    gd_ring_destroy(&ctx->scratch);
    gd_ring_destroy(&ctx->data);
    gd_arena_destroy(&ctx->params);
    gd_arena_destroy(&ctx->state);
    gd_backend_destroy(ctx->backend.backend);
    if (ctx->sync_ready) {
        pthread_cond_destroy(&ctx->data_cv);
        pthread_mutex_destroy(&ctx->data_mutex);
        pthread_mutex_destroy(&ctx->backend_mutex);
        pthread_mutex_destroy(&ctx->tensor_id_mutex);
    }
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

static bool gd_scope_mode_is_valid(gd_scope_mode mode)
{
    return mode == GD_SCOPE_TRAIN || mode == GD_SCOPE_EVAL || mode == GD_SCOPE_INFER;
}

static gd_status gd_data_shape_nbytes(gd_dtype dtype, gd_shape shape, size_t *out)
{
    size_t elem_size;
    size_t n = 1U;
    uint32_t i;
    if (out == NULL || shape.rank > GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_dtype_size(dtype);
    if (elem_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < shape.rank; ++i) {
        if (shape.dims[i] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((uint64_t)shape.dims[i] > (uint64_t)(SIZE_MAX / n)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        n *= (size_t)shape.dims[i];
    }
    if (n > SIZE_MAX / elem_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = n * elem_size;
    return GD_OK;
}

static void gd_data_tensor_set_strides(gd_tensor *tensor)
{
    uint32_t i;
    int64_t stride = 1;
    for (i = tensor->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        tensor->strides[dim] = stride;
        stride *= tensor->shape[dim];
    }
}

gd_status gd_context_data_slot_acquire(gd_context *ctx,
                                       int32_t *out_slot,
                                       uint64_t *out_generation)
{
    if (out_slot != NULL) {
        *out_slot = -1;
    }
    if (out_generation != NULL) {
        *out_generation = 0U;
    }
    if (ctx == NULL || out_slot == NULL || out_generation == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (;;) {
        int32_t retired = -1;
        uint64_t retired_fence = 0U;
        uint32_t i;
        pthread_mutex_lock(&ctx->data_mutex);
        for (i = 0U; i < ctx->data.n_slots; ++i) {
            if (ctx->data.slot_states[i] == (uint8_t)GD_DATA_SLOT_FREE_INTERNAL) {
                gd_arena_reset(&ctx->data.slots[i]);
                ctx->data.slot_fences[i] = 0U;
                ctx->data.slot_states[i] = (uint8_t)GD_DATA_SLOT_FILLING_INTERNAL;
                *out_slot = (int32_t)i;
                *out_generation = ctx->data.slots[i].generation;
                pthread_mutex_unlock(&ctx->data_mutex);
                return GD_OK;
            }
        }
        for (i = 0U; i < ctx->data.n_slots; ++i) {
            if (ctx->data.slot_states[i] == (uint8_t)GD_DATA_SLOT_RETIRED_INTERNAL) {
                retired = (int32_t)i;
                retired_fence = ctx->data.slot_fences[i];
                break;
            }
        }
        if (retired < 0) {
            pthread_cond_wait(&ctx->data_cv, &ctx->data_mutex);
            pthread_mutex_unlock(&ctx->data_mutex);
            continue;
        }
        pthread_mutex_unlock(&ctx->data_mutex);
        {
            gd_status st = gd_backend_wait_until(ctx, retired_fence);
            if (st != GD_OK) {
                return st;
            }
        }
        pthread_mutex_lock(&ctx->data_mutex);
        if (ctx->data.slot_states[(uint32_t)retired] ==
                (uint8_t)GD_DATA_SLOT_RETIRED_INTERNAL &&
            ctx->data.slot_fences[(uint32_t)retired] == retired_fence) {
            ctx->data.slot_states[(uint32_t)retired] =
                (uint8_t)GD_DATA_SLOT_FREE_INTERNAL;
            ctx->data.slot_fences[(uint32_t)retired] = 0U;
            pthread_cond_broadcast(&ctx->data_cv);
        }
        pthread_mutex_unlock(&ctx->data_mutex);
    }
}

gd_status gd_context_data_slot_tensor(gd_context *ctx,
                                      int32_t slot,
                                      uint64_t generation,
                                      gd_dtype dtype,
                                      gd_shape shape,
                                      size_t alignment,
                                      gd_tensor *out)
{
    gd_tensor tensor;
    size_t nbytes = 0U;
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || out == NULL || slot < 0 || (uint32_t)slot >= ctx->data.n_slots) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_data_shape_nbytes(dtype, shape, &nbytes);
    if (st != GD_OK) {
        return gd_set_error(ctx, st, "invalid data tensor shape");
    }
    memset(&tensor, 0, sizeof(tensor));
    st = gd_context_next_tensor_id(ctx, &tensor.id);
    if (st != GD_OK) {
        return st;
    }
    tensor.version = 0U;
    tensor.dtype = dtype;
    tensor.device = GD_DEVICE_GPU;
    tensor.layout = GD_LAYOUT_STRIDED;
    tensor.rank = shape.rank;
    if (shape.rank > 0U) {
        memcpy(tensor.shape, shape.dims, (size_t)shape.rank * sizeof(tensor.shape[0]));
    }
    gd_data_tensor_set_strides(&tensor);
    pthread_mutex_lock(&ctx->data_mutex);
    if (ctx->data.slot_states[(uint32_t)slot] !=
            (uint8_t)GD_DATA_SLOT_FILLING_INTERNAL ||
        ctx->data.slots[(uint32_t)slot].generation != generation) {
        pthread_mutex_unlock(&ctx->data_mutex);
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "data slot is not fillable");
    }
    st = gd_arena_alloc(ctx, &ctx->data.slots[(uint32_t)slot], nbytes,
                        alignment, &tensor.storage);
    pthread_mutex_unlock(&ctx->data_mutex);
    if (st != GD_OK) {
        return st;
    }
    tensor.view_offset = 0U;
    tensor.is_view = false;
    tensor.requires_grad = false;
    tensor.is_leaf = true;
    *out = tensor;
    return GD_OK;
}

gd_status gd_context_data_slot_publish(gd_context *ctx,
                                       int32_t slot,
                                       uint64_t generation)
{
    if (ctx == NULL || slot < 0 || (uint32_t)slot >= ctx->data.n_slots) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&ctx->data_mutex);
    if (ctx->data.slot_states[(uint32_t)slot] !=
            (uint8_t)GD_DATA_SLOT_FILLING_INTERNAL ||
        ctx->data.slots[(uint32_t)slot].generation != generation) {
        pthread_mutex_unlock(&ctx->data_mutex);
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "data slot publish state mismatch");
    }
    ctx->data.slot_states[(uint32_t)slot] = (uint8_t)GD_DATA_SLOT_READY_INTERNAL;
    pthread_cond_broadcast(&ctx->data_cv);
    pthread_mutex_unlock(&ctx->data_mutex);
    return GD_OK;
}

gd_status gd_context_data_slot_abort(gd_context *ctx,
                                     int32_t slot,
                                     uint64_t generation)
{
    if (ctx == NULL || slot < 0 || (uint32_t)slot >= ctx->data.n_slots) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&ctx->data_mutex);
    if (ctx->data.slots[(uint32_t)slot].generation == generation &&
        (ctx->data.slot_states[(uint32_t)slot] ==
             (uint8_t)GD_DATA_SLOT_FILLING_INTERNAL ||
         ctx->data.slot_states[(uint32_t)slot] ==
             (uint8_t)GD_DATA_SLOT_READY_INTERNAL)) {
        ctx->data.slot_states[(uint32_t)slot] = (uint8_t)GD_DATA_SLOT_FREE_INTERNAL;
        ctx->data.slot_fences[(uint32_t)slot] = 0U;
        pthread_cond_broadcast(&ctx->data_cv);
    }
    pthread_mutex_unlock(&ctx->data_mutex);
    return GD_OK;
}

static gd_status gd_context_begin_on_data_slot(gd_context *ctx,
                                               int32_t slot,
                                               uint64_t generation)
{
    if (slot < 0 || (uint32_t)slot >= ctx->data.n_slots) {
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid step data slot");
    }
    pthread_mutex_lock(&ctx->data_mutex);
    if (ctx->data.slot_states[(uint32_t)slot] != (uint8_t)GD_DATA_SLOT_READY_INTERNAL ||
        ctx->data.slots[(uint32_t)slot].generation != generation) {
        pthread_mutex_unlock(&ctx->data_mutex);
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "step batch is not ready");
    }
    ctx->data.slot_states[(uint32_t)slot] = (uint8_t)GD_DATA_SLOT_IN_STEP_INTERNAL;
    ctx->data.current = slot;
    ctx->active_data_slot = slot;
    ctx->active_data_generation = generation;
    pthread_mutex_unlock(&ctx->data_mutex);
    return GD_OK;
}

static void gd_context_rollback_data_begin(gd_context *ctx, bool empty_batch)
{
    if (ctx == NULL || ctx->active_data_slot < 0) {
        return;
    }
    pthread_mutex_lock(&ctx->data_mutex);
    if ((uint32_t)ctx->active_data_slot < ctx->data.n_slots &&
        ctx->data.slot_states[(uint32_t)ctx->active_data_slot] ==
            (uint8_t)GD_DATA_SLOT_IN_STEP_INTERNAL &&
        ctx->data.slots[(uint32_t)ctx->active_data_slot].generation ==
            ctx->active_data_generation) {
        ctx->data.slot_states[(uint32_t)ctx->active_data_slot] =
            empty_batch ? (uint8_t)GD_DATA_SLOT_FREE_INTERNAL
                        : (uint8_t)GD_DATA_SLOT_READY_INTERNAL;
        pthread_cond_broadcast(&ctx->data_cv);
    }
    ctx->data.current = -1;
    ctx->active_data_slot = -1;
    ctx->active_data_generation = 0U;
    pthread_mutex_unlock(&ctx->data_mutex);
}

gd_status gd_begin_step(gd_context *ctx, gd_scope_mode mode, gd_batch *batch)
{
    gd_status st;
    int32_t slot = -1;
    uint64_t generation = 0U;
    bool empty_batch;
    if (ctx == NULL || batch == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    empty_batch = _gd_batch_is_empty(batch) != 0;
    if (ctx->in_scope || !gd_scope_mode_is_valid(mode)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "invalid gd_begin_step state");
    }
    if (empty_batch) {
        st = gd_context_data_slot_acquire(ctx, &slot, &generation);
        if (st == GD_OK) {
            st = gd_context_data_slot_publish(ctx, slot, generation);
        }
        if (st != GD_OK) {
            if (slot >= 0) {
                (void)gd_context_data_slot_abort(ctx, slot, generation);
            }
            return st;
        }
    } else {
        st = _gd_batch_begin_step(batch, ctx, &slot, &generation);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_context_begin_on_data_slot(ctx, slot, generation);
    if (st != GD_OK) {
        if (empty_batch) {
            (void)gd_context_data_slot_abort(ctx, slot, generation);
        } else {
            _gd_batch_abort_begin_step(batch);
        }
        return st;
    }
    st = gd_ring_select(ctx, &ctx->scratch);
    if (st != GD_OK) {
        gd_context_rollback_data_begin(ctx, empty_batch);
        if (!empty_batch) {
            _gd_batch_abort_begin_step(batch);
        }
        return st;
    }
    st = gd_backend_scope_begin(ctx->backend.backend);
    if (st != GD_OK) {
        gd_context_rollback_data_begin(ctx, empty_batch);
        if (!empty_batch) {
            _gd_batch_abort_begin_step(batch);
        }
        return gd_set_error(ctx, st, "backend scope begin failed");
    }
    ctx->mode = mode;
    ctx->in_scope = true;
    ctx->touched_state_count = 0U;
    ctx->active_batch = empty_batch ? NULL : batch;
    st = gd_autograd_on_begin(ctx);
    if (st != GD_OK) {
        ctx->active_batch = NULL;
        ctx->mode = GD_SCOPE_NONE;
        ctx->in_scope = false;
        gd_context_rollback_data_begin(ctx, empty_batch);
        if (!empty_batch) {
            _gd_batch_abort_begin_step(batch);
        }
        return st;
    }
    return GD_OK;
}

gd_status gd_end_step(gd_context *ctx)
{
    uint32_t i;
    uint64_t fence;
    gd_status st;
    int32_t data_slot;
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->in_scope || ctx->scratch.current < 0 || ctx->active_data_slot < 0) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "invalid gd_end_step state");
    }
    for (i = 0; i < ctx->touched_state_count; ++i) {
        if (!gd_state_object_live(ctx, ctx->touched_state[i].object) ||
            ctx->touched_state[i].object->epoch != ctx->touched_state[i].epoch) {
            return gd_set_error(ctx, GD_ERR_BAD_STATE,
                                "touched state object changed during step");
        }
    }
    st = gd_backend_record(ctx, &fence);
    if (st != GD_OK) {
        return st;
    }
    ctx->scratch.slot_fences[ctx->scratch.current] = fence;
    data_slot = ctx->active_data_slot;
    pthread_mutex_lock(&ctx->data_mutex);
    if ((uint32_t)data_slot < ctx->data.n_slots &&
        ctx->data.slot_states[(uint32_t)data_slot] ==
            (uint8_t)GD_DATA_SLOT_IN_STEP_INTERNAL) {
        ctx->data.slot_fences[(uint32_t)data_slot] = fence;
        ctx->data.slot_states[(uint32_t)data_slot] =
            (uint8_t)GD_DATA_SLOT_RETIRED_INTERNAL;
        pthread_cond_broadcast(&ctx->data_cv);
    }
    ctx->data.current = -1;
    pthread_mutex_unlock(&ctx->data_mutex);
    for (i = 0; i < ctx->touched_state_count; ++i) {
        ctx->touched_state[i].object->last_use_fence = fence;
    }
    ctx->touched_state_count = 0U;
    ctx->last_scope_fence = fence;
    gd_autograd_on_end(ctx);
    ctx->mode = GD_SCOPE_NONE;
    ctx->in_scope = false;
    if (ctx->active_batch != NULL) {
        _gd_batch_end_step(ctx->active_batch, fence);
    }
    ctx->active_batch = NULL;
    ctx->active_data_slot = -1;
    ctx->active_data_generation = 0U;
    return GD_OK;
}

gd_status gd_alloc_params(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
        out->slot = -1;
    }
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_arena_alloc(ctx, &ctx->params, nbytes, alignment, out);
}

gd_status gd_alloc_state(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out)
{
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
        out->slot = -1;
    }
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
                                 gd_state_object **out)
{
    gd_status st;
    gd_state_object *object;
    gd_span span;
    if (out != NULL) {
        *out = NULL;
    }
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    object = (gd_state_object *)gd_heap_alloc(sizeof(*object));
    if (object == NULL) {
        return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "state object allocation failed");
    }
    st = gd_alloc_state(ctx, nbytes, alignment, &span);
    if (st != GD_OK) {
        free(object);
        return st;
    }
    memset(object, 0, sizeof(*object));
    object->span = span;
    object->epoch = 1U;
    gd_state_object_link(ctx, object);
    *out = object;
    return GD_OK;
}

gd_status gd_state_object_destroy(gd_context *ctx, gd_state_object *object)
{
    gd_status st;
    if (ctx == NULL || object == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_state_object_live(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "state object is stale");
    }
    if (ctx->in_scope && gd_state_object_touched(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BUSY,
                            "cannot destroy state object touched in active scope");
    }
    if (object->last_use_fence != 0U &&
        !gd_fence_sequence_is_complete(ctx, object->last_use_fence)) {
        st = gd_backend_wait_until(ctx, object->last_use_fence);
        if (st != GD_OK) {
            return st;
        }
    }
    gd_state_object_unlink(ctx, object);
    memset(object, 0, sizeof(*object));
    free(object);
    return GD_OK;
}

gd_status gd_state_object_acquire_span(gd_context *ctx,
                                       gd_state_object *object,
                                       gd_state_access access,
                                       gd_span *out)
{
    int32_t touched;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
        out->slot = -1;
    }
    if (ctx == NULL || object == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!ctx->in_scope) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE,
                            "state object acquire requires active scope");
    }
    if (!gd_state_access_valid(access)) {
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid state access mode");
    }
    if (!gd_state_object_live(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BAD_STATE, "state object is stale");
    }
    touched = gd_state_touched_find(ctx, object);
    if (touched >= 0) {
        gd_touched_state *entry = &ctx->touched_state[(uint32_t)touched];
        entry->access = (gd_state_access)((uint32_t)entry->access | (uint32_t)access);
        *out = entry->span;
        return GD_OK;
    }
    if (ctx->touched_state_count >= GD_MAX_TOUCHED_STATE) {
        return gd_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "too many touched state objects");
    }
    ctx->touched_state[ctx->touched_state_count].object = object;
    ctx->touched_state[ctx->touched_state_count].span = object->span;
    ctx->touched_state[ctx->touched_state_count].epoch = object->epoch;
    ctx->touched_state[ctx->touched_state_count].access = access;
    ctx->touched_state_count += 1U;
    *out = object->span;
    return GD_OK;
}

gd_status gd_state_object_reset(gd_context *ctx,
                                gd_state_object *object,
                                size_t nbytes,
                                size_t alignment,
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
    if (ctx->in_scope && gd_state_object_touched(ctx, object)) {
        return gd_set_error(ctx, GD_ERR_BUSY,
                            "cannot reset state object touched in active scope");
    }
    if (nbytes == 0U) {
        nbytes = object->span.nbytes;
    }
    if (alignment == 0U) {
        alignment = object->span.alignment;
    } else if (!gd_is_power_of_two(alignment) || alignment > GD_MAX_SUBALLOC_ALIGNMENT) {
        return gd_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "state object reset invalid alignment");
    }
    need_fresh = nbytes > object->span.nbytes ||
                 (alignment > object->span.alignment && (object->span.offset % alignment) != 0U);
    if (object->last_use_fence != 0U &&
        !gd_fence_sequence_is_complete(ctx, object->last_use_fence)) {
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
    gd_state_object_bump_epoch(object);
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

uint64_t gd_debug_state_object_last_use_fence(const gd_state_object *object)
{
    return object != NULL ? object->last_use_fence : 0U;
}
