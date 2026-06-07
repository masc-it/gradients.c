#ifndef GD_CORE_MEMORY_INTERNAL_H
#define GD_CORE_MEMORY_INTERNAL_H

#include <stdbool.h>

#include <gradients/memory.h>
#include <gradients/tensor.h>

typedef struct gd_backend gd_backend;
typedef struct gd_autograd_state gd_autograd_state;
typedef struct gd_batch gd_batch;

gd_backend *gd_context_backend(gd_context *ctx);
gd_autograd_state *gd_context_autograd(gd_context *ctx);
const gd_autograd_state *gd_context_autograd_const(const gd_context *ctx);
gd_scope_mode gd_context_scope_mode(const gd_context *ctx);
bool gd_context_in_scope(const gd_context *ctx);
gd_status gd_context_next_tensor_id(gd_context *ctx, uint64_t *out_id);
uint64_t gd_context_now_ns(void);

gd_status gd_context_data_slot_acquire(gd_context *ctx,
                                       int32_t *out_slot,
                                       uint64_t *out_generation);
gd_status gd_context_data_slot_tensor(gd_context *ctx,
                                      int32_t slot,
                                      uint64_t generation,
                                      gd_dtype dtype,
                                      gd_shape shape,
                                      size_t alignment,
                                      gd_tensor *out);
gd_status gd_context_data_slot_publish(gd_context *ctx,
                                       int32_t slot,
                                       uint64_t generation);
gd_status gd_context_data_slot_abort(gd_context *ctx,
                                     int32_t slot,
                                     uint64_t generation);

gd_status _gd_batch_begin_step(gd_batch *batch,
                               gd_context *ctx,
                               int32_t *out_slot,
                               uint64_t *out_generation);
void _gd_batch_end_step(gd_batch *batch, uint64_t fence);
void _gd_batch_abort_begin_step(gd_batch *batch);
int _gd_batch_is_empty(const gd_batch *batch);
gd_batch *gd_batch_empty(void);

gd_status gd_autograd_state_create(gd_context *ctx, gd_autograd_state **out_state);
void gd_autograd_state_destroy(gd_autograd_state *state);
gd_status gd_autograd_on_begin(gd_context *ctx);
void gd_autograd_on_end(gd_context *ctx);

gd_status gd_context_set_error(gd_context *ctx, gd_status status, const char *message);
gd_status gd_context_flush_backend(gd_context *ctx);
gd_status gd_context_synchronize(gd_context *ctx);
gd_status gd_context_wait_for_span(gd_context *ctx, const gd_span *span);

gd_status gd_context_alloc_span(gd_context *ctx,
                                gd_arena_kind arena,
                                size_t nbytes,
                                size_t alignment,
                                gd_span *out);

gd_status gd_context_free_span(gd_context *ctx, const gd_span *span);

bool gd_context_span_is_live(const gd_context *ctx, const gd_span *span);

gd_status gd_context_validate_span(gd_context *ctx,
                                   const gd_span *span,
                                   const char *message);

#endif /* GD_CORE_MEMORY_INTERNAL_H */
