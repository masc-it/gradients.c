#ifndef GRADIENTS_CORE_INTERNAL_H
#define GRADIENTS_CORE_INTERNAL_H

#include "gradients/context.h"
#include "gradients/device.h"
#include "gradients/status.h"

typedef struct gd_graph gd_graph;
typedef struct _gd_backend _gd_backend;
typedef struct _gd_backend_vtable _gd_backend_vtable;

void _gd_set_last_error(gd_status status, const char *message);
gd_status _gd_error(gd_status status, const char *message);

gd_status _gd_device_validate_available(const gd_context *ctx, gd_device device);

gd_graph *_gd_context_active_graph(const gd_context *ctx);
gd_status _gd_context_set_active_graph(gd_context *ctx, gd_graph *graph);

const char *_gd_context_scope(const gd_context *ctx);
gd_status _gd_context_scope_push(gd_context *ctx, const char *name);
gd_status _gd_context_scope_pop(gd_context *ctx);

_gd_backend *_gd_context_backend(const gd_context *ctx, gd_device device);
gd_status _gd_context_register_backend(gd_context *ctx, const _gd_backend_vtable *vt);

#endif /* GRADIENTS_CORE_INTERNAL_H */
