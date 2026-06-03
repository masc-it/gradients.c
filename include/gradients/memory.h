#ifndef GRADIENTS_MEMORY_H
#define GRADIENTS_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_context gd_context;

typedef enum gd_arena_kind {
    GD_ARENA_PARAMS = 0,
    GD_ARENA_STATE = 1,
    GD_ARENA_SCRATCH = 2,
    GD_ARENA_DATA = 3,
} gd_arena_kind;

typedef enum gd_scope_mode {
    GD_SCOPE_NONE = 0,
    GD_SCOPE_TRAIN = 1,
    GD_SCOPE_EVAL = 2,
    GD_SCOPE_INFER = 3,
} gd_scope_mode;

typedef struct gd_memory_config {
    size_t params_bytes;
    size_t state_bytes;
    size_t scratch_slot_bytes;
    size_t data_slot_bytes;
    uint32_t scratch_slots;
    uint32_t data_slots;
    size_t default_alignment;
} gd_memory_config;

typedef struct gd_span {
    gd_arena_kind arena;
    int32_t slot;
    size_t offset;
    size_t nbytes;
    size_t alignment;
    uint64_t generation;
    void *buffer;
    void *host_ptr;
} gd_span;

typedef struct gd_arena_stats {
    size_t capacity;
    size_t offset;
    size_t watermark;
    uint64_t generation;
    bool sealed;
} gd_arena_stats;

typedef struct gd_ring_stats {
    uint32_t slots;
    int32_t current_slot;
    size_t total_capacity;
    size_t total_watermark;
    size_t max_slot_watermark;
    uint64_t waits;
} gd_ring_stats;

typedef struct gd_memory_stats {
    gd_arena_stats params;
    gd_arena_stats state;
    gd_ring_stats scratch;
    gd_ring_stats data;
    uint64_t backend_waits;
    uint64_t last_scope_fence;
} gd_memory_stats;

typedef struct gd_state_object {
    gd_span span;
    uint64_t last_use_fence;
} gd_state_object;

gd_memory_config gd_memory_config_default(void);

gd_status gd_context_create(const gd_memory_config *config, gd_context **out_ctx);
void gd_context_destroy(gd_context *ctx);

gd_status gd_context_status(const gd_context *ctx);
const char *gd_context_error(const gd_context *ctx);
void gd_context_clear_error(gd_context *ctx);

gd_status gd_context_seal_params(gd_context *ctx);

gd_status gd_begin(gd_context *ctx, gd_scope_mode mode);
gd_status gd_end(gd_context *ctx);

gd_status gd_alloc_params(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out);
gd_status gd_alloc_state(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out);
gd_status gd_alloc_scratch(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out);
gd_status gd_alloc_data(gd_context *ctx, size_t nbytes, size_t alignment, gd_span *out);

gd_status gd_state_object_create(gd_context *ctx,
                                 size_t nbytes,
                                 size_t alignment,
                                 gd_state_object *out);
gd_status gd_state_touch(gd_context *ctx, gd_state_object *object);
gd_status gd_state_object_reset(gd_context *ctx,
                                gd_state_object *object,
                                size_t nbytes,
                                size_t alignment,
                                bool allow_realloc,
                                bool *out_relocated);

gd_status gd_memory_stats_query(const gd_context *ctx, gd_memory_stats *out);

void gd_debug_complete_all(gd_context *ctx);
uint64_t gd_debug_heap_alloc_count(void);
void gd_debug_set_heap_guard(bool forbid_heap_allocations);
int32_t gd_debug_current_ring_slot(const gd_context *ctx, gd_arena_kind arena);
uint64_t gd_debug_ring_slot_generation(const gd_context *ctx,
                                       gd_arena_kind arena,
                                       uint32_t slot);
uint64_t gd_debug_ring_slot_fence(const gd_context *ctx,
                                  gd_arena_kind arena,
                                  uint32_t slot);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_MEMORY_H */
