#ifndef GD_CORE_MEMORY_INTERNAL_H
#define GD_CORE_MEMORY_INTERNAL_H

#include <stdbool.h>

#include <gradients/memory.h>

typedef struct gd_backend gd_backend;

gd_backend *gd_context_backend(gd_context *ctx);

gd_status gd_context_set_error(gd_context *ctx, gd_status status, const char *message);
gd_status gd_context_flush_backend(gd_context *ctx);
gd_status gd_context_synchronize(gd_context *ctx);
gd_status gd_context_wait_for_span(gd_context *ctx, const gd_span *span);

gd_status gd_context_alloc_span(gd_context *ctx,
                                gd_arena_kind arena,
                                size_t nbytes,
                                size_t alignment,
                                gd_span *out);

bool gd_context_span_is_live(const gd_context *ctx, const gd_span *span);

gd_status gd_context_validate_span(gd_context *ctx,
                                   const gd_span *span,
                                   const char *message);

#endif /* GD_CORE_MEMORY_INTERNAL_H */
