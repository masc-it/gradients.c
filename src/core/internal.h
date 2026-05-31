#ifndef GRADIENTS_CORE_INTERNAL_H
#define GRADIENTS_CORE_INTERNAL_H

#include "gradients/context.h"
#include "gradients/device.h"
#include "gradients/status.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct gd_graph gd_graph;
typedef struct _gd_backend _gd_backend;
typedef struct _gd_backend_vtable _gd_backend_vtable;
typedef struct _gd_node _gd_node;

typedef enum _gd_profile_event {
    _GD_PROFILE_EVENT_STAGE_LEAVES = 0,
    _GD_PROFILE_EVENT_ENCODE,
    _GD_PROFILE_EVENT_WAIT,
    _GD_PROFILE_EVENT_WRITEBACK,
    _GD_PROFILE_EVENT_COPY_ALIAS,
    _GD_PROFILE_EVENT_COUNT
} _gd_profile_event;

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

bool _gd_profile_enabled(const gd_context *ctx);
bool _gd_profile_trace_enabled(const gd_context *ctx);
uint64_t _gd_profile_now_ns(void);
void _gd_profile_record_op_time(gd_context *ctx, const _gd_backend *backend,
                                int op, uint64_t elapsed_ns);
void _gd_profile_record_compile(gd_context *ctx, const _gd_backend *backend,
                                uint64_t elapsed_ns, const _gd_node *nodes, int n_nodes);
void _gd_profile_record_run(gd_context *ctx, const _gd_backend *backend,
                            uint64_t elapsed_ns, const _gd_node *nodes, int n_nodes);
void _gd_profile_record_sync(gd_context *ctx, const _gd_backend *backend,
                             uint64_t elapsed_ns, bool explicit_call);
void _gd_profile_record_upload(gd_context *ctx, const _gd_backend *backend,
                               uint64_t elapsed_ns, size_t nbytes);
void _gd_profile_record_download(gd_context *ctx, const _gd_backend *backend,
                                 uint64_t elapsed_ns, size_t nbytes, bool blocking_read);
void _gd_profile_record_alloc(gd_context *ctx, const _gd_backend *backend, size_t nbytes);
void _gd_profile_record_free(gd_context *ctx, const _gd_backend *backend, size_t nbytes);
void _gd_profile_record_event(gd_context *ctx, const _gd_backend *backend,
                              _gd_profile_event event, uint64_t elapsed_ns,
                              size_t nbytes, uint64_t count);

#endif /* GRADIENTS_CORE_INTERNAL_H */
