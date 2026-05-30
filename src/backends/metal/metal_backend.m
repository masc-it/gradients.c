/* Metal backend (M0): F32, one compute kernel per IR node, unified memory.
 *
 * Pinned to Apple's Metal Programming Guide semantics (see data/metal):
 *  - storageModeShared buffers are CPU-addressable, but coherency is bounded by
 *    the command buffer: CPU writes must precede commit; CPU reads must follow
 *    completion. So leaf staging happens before commit, and download/synchronize
 *    block on the in-flight command buffer.
 *  - device/queue/library/pipelines are long-lived; command buffers/encoders are
 *    transient and created per execute.
 *  - one MTLComputeCommandEncoder with one dispatch per node is ordered under the
 *    default serial dispatch, so no manual barriers are needed in v1.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string.h>

#include "../backend.h"
#include "../../core/internal.h"
#include "../../core/storage_internal.h"
#include "../../core/tensor_internal.h"
#include "../../graph/graph_internal.h"
#include "metal_kernel_types.h"

/* ---- Long-lived backend state (ARC) -------------------------------------- */

@interface GDMetalState : NSObject
@property (strong) id<MTLDevice> device;
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLLibrary> library;
/* Compute pipelines keyed by _gd_op_kind (NSNumber). */
@property (strong) NSMutableDictionary<NSNumber *, id<MTLComputePipelineState>> *pipelines;
/* Last committed command buffer, or nil once synchronized. */
@property (strong) id<MTLCommandBuffer> inFlight;
@end

@implementation GDMetalState
@end

/* op kind -> kernel function name in kernels.metal. Extend per phase. */
typedef struct {
    _gd_op_kind op;
    const char *fn;
} gd_metal_kernel_entry;

static const gd_metal_kernel_entry g_metal_kernels[] = {
    {_GD_OP_ADD, "gd_add"},
};

/* ---- Compiled executable: per-value Metal buffer plan -------------------- */

typedef struct gd_metal_value {
    gd_storage *storage;   /* owned METAL buffer storage, one per value */
    gd_tensor *external;   /* borrowed CPU leaf source, NULL for produced values */
    size_t leaf_offset;    /* byte offset of the leaf data inside its CPU storage */
} gd_metal_value;

struct _gd_executable {
    const gd_graph *graph;
    int n_values;
    gd_metal_value *values;
};

/* ---- Helpers ------------------------------------------------------------- */

static GDMetalState *state_of(_gd_backend *self)
{
    return (__bridge GDMetalState *)self->impl;
}

static id<MTLComputePipelineState> pipeline_for(GDMetalState *st, _gd_op_kind op)
{
    return st.pipelines[@((int)op)];
}

static int64_t desc_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

/* ---- Storage + transfers ------------------------------------------------- */

static gd_status metal_storage_alloc(_gd_backend *self, const gd_storage_desc *desc,
                                     void **handle_out)
{
    GDMetalState *st = state_of(self);

    *handle_out = NULL;
    if (desc->device.type != GD_DEVICE_METAL) {
        return _gd_error(GD_ERR_DEVICE, "metal storage requires a Metal device");
    }
    if (desc->memory != GD_MEM_UNIFIED && desc->memory != GD_MEM_HOST) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal storage supports unified/host memory in v1");
    }

    id<MTLBuffer> buffer = [st.device newBufferWithLength:desc->nbytes
                                                  options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "MTLBuffer allocation failed");
    }
    memset(buffer.contents, 0, desc->nbytes);
    *handle_out = (__bridge_retained void *)buffer;
    return GD_OK;
}

static void metal_storage_free(_gd_backend *self, void *handle)
{
    (void)self;
    if (handle != NULL) {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle; /* ARC releases */
        (void)buffer;
    }
}

static gd_status metal_storage_host_ptr(_gd_backend *self, void *handle, void **ptr_out)
{
    (void)self;
    if (handle == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "metal storage handle is NULL");
    }
    {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
        *ptr_out = buffer.contents;
    }
    return GD_OK;
}

static gd_status metal_synchronize(_gd_backend *self)
{
    GDMetalState *st = state_of(self);
    id<MTLCommandBuffer> cmd = st.inFlight;

    if (cmd == nil) {
        return GD_OK;
    }
    [cmd waitUntilCompleted];
    st.inFlight = nil;
    if (cmd.status == MTLCommandBufferStatusError) {
        const char *reason = cmd.error != nil
                                 ? cmd.error.localizedDescription.UTF8String
                                 : "command buffer failed";
        return _gd_error(GD_ERR_BACKEND, reason);
    }
    return GD_OK;
}

static gd_status metal_upload(_gd_backend *self, void *dst_handle, size_t dst_off,
                              const void *src, size_t nbytes)
{
    (void)self;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)dst_handle;
    memcpy((unsigned char *)buffer.contents + dst_off, src, nbytes);
    return GD_OK;
}

static gd_status metal_download(_gd_backend *self, void *src_handle, size_t src_off,
                                void *dst, size_t nbytes)
{
    /* Blocking read (P4): the CPU only observes GPU writes after completion. */
    gd_status status = metal_synchronize(self);
    if (status != GD_OK) {
        return status;
    }
    {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)src_handle;
        memcpy(dst, (const unsigned char *)buffer.contents + src_off, nbytes);
    }
    return GD_OK;
}

/* ---- Compile ------------------------------------------------------------- */

static void metal_executable_free(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;

    if (exe == NULL) {
        return;
    }
    /* Ensure no in-flight command buffer still references these buffers. */
    (void)metal_synchronize(self);
    if (exe->values != NULL) {
        for (i = 0; i < exe->n_values; ++i) {
            gd_storage_release(exe->values[i].storage);
        }
        free(exe->values);
    }
    free(exe);
}

static gd_status metal_compile(_gd_backend *self, gd_graph *graph, _gd_executable **out)
{
    gd_status status = GD_OK;
    _gd_executable *exe = NULL;
    int n = graph->n_values;
    int i = 0;

    *out = NULL;
    exe = calloc(1U, sizeof(*exe));
    if (exe == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal executable");
    }
    exe->graph = graph;
    exe->n_values = n;
    exe->values = calloc((size_t)(n > 0 ? n : 1), sizeof(*exe->values));
    if (exe->values == NULL) {
        free(exe);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal value table");
    }

    for (i = 0; i < n; ++i) {
        const _gd_value *value = &graph->values[i];
        gd_storage_desc sdesc;
        gd_storage *storage = NULL;
        size_t nbytes = 0U;
        size_t alignment = 0U;

        if (value->external != NULL && !_gd_tensor_is_contiguous(value->external)) {
            status = _gd_error(GD_ERR_UNSUPPORTED, "metal requires contiguous leaf inputs");
            goto fail;
        }
        status = gd_tensor_desc_nbytes(&value->desc, &nbytes, &alignment);
        if (status != GD_OK) {
            goto fail;
        }
        sdesc = (gd_storage_desc){{GD_DEVICE_METAL, self->device_index}, GD_MEM_UNIFIED,
                                  nbytes, alignment};
        status = gd_storage_create(graph->ctx, &sdesc, &storage);
        if (status != GD_OK) {
            goto fail;
        }
        exe->values[i].storage = storage;
        exe->values[i].external = value->external; /* borrowed; graph retains it */
        exe->values[i].leaf_offset =
            value->external != NULL ? (size_t)value->desc.storage_offset_bytes : 0U;
    }

    *out = exe;
    return GD_OK;

fail:
    metal_executable_free(self, exe);
    return status;
}

/* ---- Execute ------------------------------------------------------------- */

/* Copies each external-leaf's host bytes into its Metal buffer. Must run before
 * the command buffer is committed (coherency). */
static gd_status stage_leaves(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *src = NULL;
        void *dst = NULL;
        gd_status status = GD_OK;

        if (v->external == NULL) {
            continue;
        }
        src = gd_tensor_storage(v->external);
        if (src == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "metal leaf input has no storage");
        }
        status = gd_storage_data_cpu(v->storage, &dst);
        if (status != GD_OK) {
            return status;
        }
        status = gd_storage_copy_to_cpu(self->ctx, src, v->leaf_offset, dst,
                                        gd_storage_nbytes(v->storage));
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static void build_ew_params(gd_metal_ew_params *p,
                            const gd_tensor_desc *out_desc,
                            const gd_tensor_desc *a_desc,
                            const gd_tensor_desc *b_desc)
{
    int i = 0;

    memset(p, 0, sizeof(*p));
    p->ndim = out_desc->ndim;
    p->numel = (int)desc_numel(out_desc);
    p->a_ndim = a_desc->ndim;
    p->b_ndim = b_desc->ndim;
    for (i = 0; i < out_desc->ndim; ++i) {
        p->out_sizes[i] = (int)out_desc->sizes[i];
    }
    for (i = 0; i < a_desc->ndim; ++i) {
        p->a_sizes[i] = (int)a_desc->sizes[i];
    }
    for (i = 0; i < b_desc->ndim; ++i) {
        p->b_sizes[i] = (int)b_desc->sizes[i];
    }
}

static id<MTLBuffer> value_buffer(_gd_executable *exe, int value_id)
{
    return (__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[value_id].storage);
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        NSUInteger numel)
{
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;

    if (tg > numel) {
        tg = numel;
    }
    if (tg == 0) {
        tg = 1;
    }
    [enc dispatchThreads:MTLSizeMake(numel, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static gd_status encode_node(_gd_backend *self,
                             id<MTLComputeCommandEncoder> enc,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    GDMetalState *st = state_of(self);
    id<MTLComputePipelineState> pso = pipeline_for(st, node->op);
    const gd_tensor_desc *out_desc = NULL;
    int64_t numel = 0;

    if (pso == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op");
    }

    switch (node->op) {
    case _GD_OP_ADD: {
        gd_metal_ew_params params;
        const gd_tensor_desc *a_desc = &exe->graph->values[node->inputs[0]].desc;
        const gd_tensor_desc *b_desc = &exe->graph->values[node->inputs[1]].desc;

        out_desc = &exe->graph->values[node->outputs[0]].desc;
        numel = desc_numel(out_desc);
        build_ew_params(&params, out_desc, a_desc, b_desc);

        [enc setComputePipelineState:pso];
        [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        if (numel > 0) {
            dispatch_1d(enc, pso, (NSUInteger)numel);
        }
        return GD_OK;
    }
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "metal op not implemented in M0");
    }
}

static gd_status metal_execute_range(_gd_backend *self, _gd_executable *exe, int last_node)
{
    GDMetalState *st = state_of(self);
    gd_status status = GD_OK;
    int i = 0;

    /* About to mutate input buffers via CPU writes: any prior command buffer
     * must complete first (coherency). */
    status = metal_synchronize(self);
    if (status != GD_OK) {
        return status;
    }
    status = stage_leaves(self, exe);
    if (status != GD_OK) {
        return status;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (i = 0; i <= last_node && i < exe->graph->n_nodes; ++i) {
            status = encode_node(self, enc, exe, &exe->graph->nodes[i]);
            if (status != GD_OK) {
                [enc endEncoding];
                return status;
            }
        }
        [enc endEncoding];
        [cmd commit];
        st.inFlight = cmd;
    }
    return GD_OK;
}

static gd_status metal_execute(_gd_backend *self, _gd_executable *exe)
{
    return metal_execute_range(self, exe, exe->graph->n_nodes - 1);
}

static gd_status metal_execute_until(_gd_backend *self, _gd_executable *exe, int node_id)
{
    gd_status status = metal_execute_range(self, exe, node_id);
    if (status != GD_OK) {
        return status;
    }
    /* Partial execution is a debugging aid; make results immediately readable. */
    return metal_synchronize(self);
}

static gd_status metal_value_storage(_gd_backend *self, _gd_executable *exe, int value_id,
                                     gd_storage **storage_out, size_t *offset_out)
{
    (void)self;
    if (value_id < 0 || value_id >= exe->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    *storage_out = exe->values[value_id].storage;
    *offset_out = 0U;
    return GD_OK;
}

static bool metal_supports_node(_gd_backend *self, const _gd_node *node)
{
    return pipeline_for(state_of(self), node->op) != nil;
}

/* ---- Init / shutdown / registration -------------------------------------- */

static NSString *resolve_metallib_path(void)
{
    const char *env = getenv("GRADIENTS_METALLIB");

    if (env != NULL && env[0] != '\0') {
        return [NSString stringWithUTF8String:env];
    }
    return @"build/gradients.metallib";
}

static gd_status metal_init(_gd_backend *self, gd_context *ctx, int device_index)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = nil;
        id<MTLLibrary> library = nil;
        NSError *error = nil;
        NSString *path = resolve_metallib_path();
        GDMetalState *st = nil;
        size_t k = 0;

        (void)device_index;
        if (device == nil) {
            return _gd_error(GD_ERR_UNSUPPORTED, "no Metal device available");
        }
        queue = [device newCommandQueue];
        if (queue == nil) {
            return _gd_error(GD_ERR_BACKEND, "failed to create Metal command queue");
        }
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (library == nil) {
            return _gd_error(GD_ERR_UNSUPPORTED,
                             "failed to load gradients.metallib (set GRADIENTS_METALLIB)");
        }

        st = [GDMetalState new];
        st.device = device;
        st.queue = queue;
        st.library = library;
        st.pipelines = [NSMutableDictionary dictionary];
        st.inFlight = nil;

        for (k = 0; k < sizeof(g_metal_kernels) / sizeof(g_metal_kernels[0]); ++k) {
            id<MTLFunction> fn =
                [library newFunctionWithName:[NSString stringWithUTF8String:g_metal_kernels[k].fn]];
            id<MTLComputePipelineState> pso = nil;

            if (fn == nil) {
                return _gd_error(GD_ERR_BACKEND, "metallib is missing a required kernel");
            }
            pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "failed to create compute pipeline state");
            }
            st.pipelines[@((int)g_metal_kernels[k].op)] = pso;
        }

        self->impl = (__bridge_retained void *)st;
        self->caps.host_visible = true;
        self->caps.supports_cpu_ref = false;
        self->caps.default_memory = GD_MEM_UNIFIED;
        (void)ctx;
        return GD_OK;
    }
}

static void metal_shutdown(_gd_backend *self)
{
    if (self->impl != NULL) {
        GDMetalState *st = (__bridge_transfer GDMetalState *)self->impl; /* ARC releases */
        st.inFlight = nil;
        self->impl = NULL;
    }
}

static const _gd_backend_vtable metal_backend_vtable = {
    .type = GD_DEVICE_METAL,
    .name = "metal",
    .init = metal_init,
    .shutdown = metal_shutdown,
    .storage_alloc = metal_storage_alloc,
    .storage_free = metal_storage_free,
    .storage_host_ptr = metal_storage_host_ptr,
    .upload = metal_upload,
    .download = metal_download,
    .compile = metal_compile,
    .execute = metal_execute,
    .execute_until = metal_execute_until,
    .executable_free = metal_executable_free,
    .value_storage = metal_value_storage,
    .supports_node = metal_supports_node,
    .synchronize = metal_synchronize,
};

gd_status _gd_metal_backend_register(gd_context *ctx)
{
    return _gd_context_register_backend(ctx, &metal_backend_vtable);
}
