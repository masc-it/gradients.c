#ifndef GRADIENTS_BACKEND_H
#define GRADIENTS_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

#include "gradients/context.h"
#include "gradients/device.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations: backends see the IR/graph opaquely. */
typedef struct gd_graph gd_graph;
typedef struct gd_graph_runner gd_graph_runner;
typedef struct _gd_node _gd_node;
typedef struct _gd_executable _gd_executable;
typedef struct _gd_backend _gd_backend;

typedef struct _gd_backend_caps {
    bool host_visible;          /* can expose CPU-accessible pointers for its storage */
    bool supports_cpu_ref;      /* is itself the correctness/reference backend */
    gd_memory_kind default_memory; /* preferred memory kind for fresh tensors */
} _gd_backend_caps;

typedef struct _gd_backend_vtable {
    gd_device_type type;
    const char *name;

    gd_status (*init)(_gd_backend *self, gd_context *ctx, int device_index);
    void (*shutdown)(_gd_backend *self);

    /* Storage (P1). */
    gd_status (*storage_alloc)(_gd_backend *self, const gd_storage_desc *desc, void **handle_out);
    void (*storage_free)(_gd_backend *self, void *handle);
    gd_status (*storage_host_ptr)(_gd_backend *self, void *handle, void **ptr_out);

    /* Host transfers (P2), blocking in v1. */
    gd_status (*upload)(_gd_backend *self, void *dst_handle, size_t dst_off,
                        const void *src, size_t nbytes);
    gd_status (*download)(_gd_backend *self, void *src_handle, size_t src_off,
                          void *dst, size_t nbytes);

    /* Execution (P3). */
    gd_status (*compile)(_gd_backend *self, gd_graph *graph, _gd_executable **out);
    gd_status (*execute)(_gd_backend *self, _gd_executable *exe);
    gd_status (*execute_bound)(_gd_backend *self,
                               _gd_executable *exe,
                               const gd_graph_runner *runner); /* nullable */
    gd_status (*execute_until)(_gd_backend *self, _gd_executable *exe, int node_id); /* nullable */
    void (*executable_free)(_gd_backend *self, _gd_executable *exe);
    /* Storage backing a compiled value (for host transfer / materialization). */
    gd_status (*value_storage)(_gd_backend *self, _gd_executable *exe, int value_id,
                               gd_storage **storage_out, size_t *offset_out);

    /* Capability preflight. Must return the same decision compile would make
     * for this finalized graph node, including dtype/layout/shape/attrs. */
    gd_status (*check_node)(_gd_backend *self, const gd_graph *graph, const _gd_node *node);

    /* Ordering (P4). */
    gd_status (*synchronize)(_gd_backend *self);

    /* Lazy writeback (P4): flush any pending device->host writeback associated
     * with `cookie` (an opaque backend handle, e.g. a compiled executable),
     * making the affected host-visible storages current. Invoked by the core
     * when a host read hits a storage the backend marked pending. Nullable for
     * backends that never defer writeback (e.g. CPU_REF). */
    gd_status (*flush_pending)(_gd_backend *self, void *cookie);
} _gd_backend_vtable;

struct _gd_backend {
    const _gd_backend_vtable *vt;
    _gd_backend_caps caps;
    gd_context *ctx;
    int device_index;
    void *impl; /* backend-private state */
};

/* Backend registration entry points (one per backend). */
gd_status _gd_backend_check_node(_gd_backend *backend,
                                 const gd_graph *graph,
                                 const _gd_node *node);
gd_status _gd_backend_check_graph(_gd_backend *backend,
                                  const gd_graph *graph,
                                  int *bad_node_out);

gd_status _gd_cpu_backend_register(gd_context *ctx);
#if defined(GD_ENABLE_METAL)
/* Best-effort: returns GD_ERR_UNSUPPORTED (without registering) when no Metal
 * device or shader library is available, so callers can ignore the result. */
gd_status _gd_metal_backend_register(gd_context *ctx);
#endif

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_BACKEND_H */
