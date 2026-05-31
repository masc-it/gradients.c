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
/* Pipelines keyed by function name, for ops needing multiple kernels (SDPA bwd). */
@property (strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelinesByName;
/* Last committed command buffer, or nil once synchronized. */
@property (strong) id<MTLCommandBuffer> inFlight;
/* Executables with deferred CPU-backed external write-back, pending until the
 * next synchronize/flush (P4 lazy writeback). Holds boxed _gd_executable*
 * pointers; deduped. Empty for fully device-resident graphs. */
@property (strong) NSMutableArray<NSValue *> *pendingExes;
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
    {_GD_OP_MUL, "gd_mul"},
    {_GD_OP_SCALE, "gd_scale"},
    {_GD_OP_RELU, "gd_relu"},
    {_GD_OP_SILU, "gd_silu"},
    {_GD_OP_COPY, "gd_copy"},
    {_GD_OP_CAST, "gd_cast"},
    {_GD_OP_MATMUL, "gd_matmul_tiled"},
    {_GD_OP_LINEAR, "gd_linear_tiled"},
    {_GD_OP_SUM, "gd_reduce"},
    {_GD_OP_MEAN, "gd_reduce"},
    {_GD_OP_SOFTMAX, "gd_softmax"},
    {_GD_OP_RMS_NORM, "gd_rms_norm"},
    {_GD_OP_CROSS_ENTROPY, "gd_cross_entropy"},
    {_GD_OP_RELU_BWD, "gd_relu_bwd"},
    {_GD_OP_SILU_BWD, "gd_silu_bwd"},
    {_GD_OP_SOFTMAX_BWD, "gd_softmax_bwd"},
    {_GD_OP_SUM_BWD, "gd_sum_bwd"},
    {_GD_OP_MEAN_BWD, "gd_sum_bwd"},
    {_GD_OP_CROSS_ENTROPY_BWD, "gd_cross_entropy_bwd"},
    {_GD_OP_REDUCE_TO, "gd_reduce_to"},
    {_GD_OP_STEP_INC, "gd_step_inc"},
    {_GD_OP_ADAMW_STEP, "gd_adamw"},
    {_GD_OP_GELU, "gd_gelu"},
    {_GD_OP_GELU_BWD, "gd_gelu_bwd"},
    {_GD_OP_TRANSPOSE, "gd_transpose"},
    {_GD_OP_EMBEDDING, "gd_embedding"},
    {_GD_OP_EMBEDDING_BWD, "gd_embedding_bwd"},
    {_GD_OP_ROPE, "gd_rope"},
    {_GD_OP_ROPE_BWD, "gd_rope"},
    {_GD_OP_SDPA, "gd_sdpa"},
    {_GD_OP_SDPA_BWD, "gd_sdpa_bwd_dq"},
    {_GD_OP_RMS_NORM_BWD, "gd_rms_norm_bwd"},
    {_GD_OP_RMS_NORM_WBWD, "gd_rms_norm_wbwd"},
};

/* Kernels not mapped 1:1 to an op (looked up by name during encode). */
static const char *const g_metal_extra_kernels[] = {
    "gd_sdpa_bwd_dkv",
};

static gd_status metal_dtype_code(gd_dtype dtype, int *out)
{
    switch (dtype) {
    case GD_DTYPE_F32:
        *out = GD_METAL_DT_F32;
        return GD_OK;
    case GD_DTYPE_I32:
        *out = GD_METAL_DT_I32;
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cast supports F32/I32 only in v1");
    }
}

/* ---- Compiled executable: per-value Metal buffer plan -------------------- */

typedef struct gd_metal_value {
    gd_storage *storage;   /* retained METAL buffer storage; copy/reshape values may alias */
    gd_tensor *external;   /* borrowed leaf source, NULL for produced values */
    size_t leaf_offset;    /* byte offset of the leaf data inside its source storage */
    bool external_alias;   /* true when storage directly aliases a Metal external leaf */
    bool needs_writeback;  /* CPU-backed external value is mutated by this executable */
    bool has_staged;
    uint64_t staged_version;
} gd_metal_value;

struct _gd_executable {
    const gd_graph *graph;
    int n_values;
    gd_metal_value *values;
    bool needs_stage;
    bool needs_writeback;
    /* Per-node compiled encode plan (P8): pipeline state(s) resolved once at
     * compile so the per-run encode loop does no dictionary lookups. Stored as
     * unretained __bridge pointers; the owning GDMetalState retains the
     * pipelines for the context lifetime, which outlives every executable. */
    int n_plan;
    void **node_pso;   /* primary pipeline per node id, or NULL */
    void **node_pso2;  /* secondary pipeline (e.g. sdpa_bwd dkv), or NULL */
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

static id<MTLComputePipelineState> pipeline_named(GDMetalState *st, const char *name)
{
    return st.pipelinesByName[[NSString stringWithUTF8String:name]];
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

/* Persist any external-leaf mutations (optimizer in-place updates, gradient
 * slots written via emit_to) from their Metal buffers back to the owning CPU
 * tensor storage. CPU_REF gets this for free by borrowing leaf storage; Metal
 * stages copies, so it must copy back after the work completes. */
static gd_status writeback_externals(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;
    uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
    size_t bytes = 0U;
    uint64_t count = 0U;

    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *dst = NULL;
        void *src = NULL;
        size_t nbytes = 0U;
        gd_status status = GD_OK;

        if (v->external == NULL || v->external_alias || !v->needs_writeback) {
            continue;
        }
        dst = gd_tensor_storage(v->external);
        if (dst == NULL) {
            continue;
        }
        nbytes = gd_storage_nbytes(v->storage);
        bytes += nbytes;
        count += 1U;
        status = gd_storage_data_cpu(v->storage, &src);
        if (status != GD_OK) {
            return status;
        }
        status = gd_storage_copy_from_cpu(self->ctx, dst, v->leaf_offset, src, nbytes);
        if (status != GD_OK) {
            return status;
        }
    }
    if (start != 0U) {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_WRITEBACK,
                                 _gd_profile_now_ns() - start, bytes, count);
    }
    return GD_OK;
}

/* Writes back every executable with deferred CPU-backed external mutations and
 * clears the pending set. Called after the in-flight command buffer completes. */
static gd_status flush_pending_writebacks(_gd_backend *self)
{
    GDMetalState *st = state_of(self);
    gd_status status = GD_OK;

    if (st.pendingExes.count == 0) {
        return GD_OK;
    }
    for (NSValue *boxed in st.pendingExes) {
        _gd_executable *exe = (_gd_executable *)boxed.pointerValue;
        gd_status s = writeback_externals(self, exe);
        if (s != GD_OK && status == GD_OK) {
            status = s;
        }
    }
    [st.pendingExes removeAllObjects];
    return status;
}

/* Registers an executable's CPU-backed mutated externals for deferred writeback.
 * The host storages are marked pending so a later read resolves the flush; the
 * executable is added to the pending set (deduped) for the next synchronize. */
static void register_pending_writeback(_gd_backend *self, _gd_executable *exe)
{
    GDMetalState *st = state_of(self);
    NSValue *boxed = [NSValue valueWithPointer:exe];
    int i = 0;

    if (![st.pendingExes containsObject:boxed]) {
        [st.pendingExes addObject:boxed];
    }
    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *dst = NULL;

        if (v->external == NULL || v->external_alias || !v->needs_writeback) {
            continue;
        }
        dst = gd_tensor_storage(v->external);
        if (dst != NULL) {
            _gd_storage_set_pending_flush(dst, self, exe);
        }
    }
}

static gd_status metal_synchronize(_gd_backend *self)
{
    GDMetalState *st = state_of(self);
    id<MTLCommandBuffer> cmd = st.inFlight;

    if (cmd == nil) {
        /* No GPU work in flight, but a prior sync may have left writebacks queued
         * (it never happens today, but keep the set authoritative). */
        return flush_pending_writebacks(self);
    }
    {
        uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
        [cmd waitUntilCompleted];
        if (start != 0U) {
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_WAIT,
                                     _gd_profile_now_ns() - start, 0U, 1U);
        }
    }
    st.inFlight = nil;
    if (cmd.status == MTLCommandBufferStatusError) {
        const char *reason = cmd.error != nil
                                 ? cmd.error.localizedDescription.UTF8String
                                 : "command buffer failed";
        [st.pendingExes removeAllObjects];
        return _gd_error(GD_ERR_BACKEND, reason);
    }
    return flush_pending_writebacks(self);
}

/* P4 flush hook: a host read hit a storage this backend marked pending. The
 * command buffer is serial, so syncing the latest in-flight buffer completes all
 * earlier ones; we then write back every pending executable. `cookie` is
 * advisory (the registering executable); we flush the whole pending set. */
static gd_status metal_flush_pending(_gd_backend *self, void *cookie)
{
    (void)cookie;
    return metal_synchronize(self);
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
    /* node_pso/node_pso2 hold unretained __bridge pointers; just free the arrays. */
    free(exe->node_pso);
    free(exe->node_pso2);
    free(exe);
}

static bool can_alias_copy_value(const gd_graph *graph, const _gd_node *node)
{
    const _gd_value *in = NULL;
    const _gd_value *out = NULL;
    size_t in_bytes = 0U;
    size_t out_bytes = 0U;
    size_t alignment = 0U;

    if (graph == NULL || node == NULL || node->op != _GD_OP_COPY ||
        node->n_inputs != 1 || node->n_outputs != 1) {
        return false;
    }
    if (node->inputs[0] < 0 || node->inputs[0] >= graph->n_values ||
        node->outputs[0] < 0 || node->outputs[0] >= graph->n_values) {
        return false;
    }
    in = &graph->values[node->inputs[0]];
    out = &graph->values[node->outputs[0]];
    if (out->external != NULL) {
        return false; /* emit_to copy must write the external target. */
    }
    if (in->desc.dtype != out->desc.dtype || !gd_device_equal(in->desc.device, out->desc.device) ||
        in->desc.quant != out->desc.quant ||
        in->desc.storage_offset_bytes != 0 || out->desc.storage_offset_bytes != 0 ||
        in->desc.layout != GD_LAYOUT_CONTIGUOUS || out->desc.layout != GD_LAYOUT_CONTIGUOUS) {
        return false;
    }
    if (gd_tensor_desc_nbytes(&in->desc, &in_bytes, &alignment) != GD_OK ||
        gd_tensor_desc_nbytes(&out->desc, &out_bytes, &alignment) != GD_OK) {
        _gd_set_last_error(GD_OK, NULL);
        return false;
    }
    return in_bytes == out_bytes;
}

static bool node_mutates_input(const _gd_node *node, int input_index)
{
    switch (node->op) {
    case _GD_OP_STEP_INC:
        return input_index == 0;
    case _GD_OP_ADAMW_STEP:
        return input_index == 0 || input_index == 2 || input_index == 3;
    default:
        return false;
    }
}

static void mark_external_writebacks(_gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->graph->n_nodes; ++i) {
        const _gd_node *node = &exe->graph->nodes[i];
        int j = 0;

        for (j = 0; j < node->n_outputs; ++j) {
            int value_id = node->outputs[j];
            if (value_id >= 0 && value_id < exe->n_values &&
                exe->values[value_id].external != NULL && !exe->values[value_id].external_alias) {
                exe->values[value_id].needs_writeback = true;
            }
        }
        for (j = 0; j < node->n_inputs; ++j) {
            int value_id = node->inputs[j];
            if (node_mutates_input(node, j) && value_id >= 0 && value_id < exe->n_values &&
                exe->values[value_id].external != NULL && !exe->values[value_id].external_alias) {
                exe->values[value_id].needs_writeback = true;
            }
        }
    }
}

static void refresh_external_flags(_gd_executable *exe)
{
    int i = 0;

    exe->needs_stage = false;
    exe->needs_writeback = false;
    for (i = 0; i < exe->n_values; ++i) {
        if (exe->values[i].external != NULL && !exe->values[i].external_alias) {
            exe->needs_stage = true;
        }
        if (exe->values[i].needs_writeback) {
            exe->needs_writeback = true;
        }
    }
}

static gd_status alias_copy_values(_gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->graph->n_nodes; ++i) {
        const _gd_node *node = &exe->graph->nodes[i];
        int input = 0;
        int output = 0;
        gd_status status = GD_OK;

        if (!can_alias_copy_value(exe->graph, node)) {
            continue;
        }
        input = node->inputs[0];
        output = node->outputs[0];
        if (exe->values[input].storage == exe->values[output].storage) {
            continue;
        }
        status = gd_storage_retain(exe->values[input].storage);
        if (status != GD_OK) {
            return status;
        }
        gd_storage_release(exe->values[output].storage);
        exe->values[output].storage = exe->values[input].storage;
        exe->values[output].leaf_offset = exe->values[input].leaf_offset;
    }
    return GD_OK;
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
        if (value->external != NULL &&
            gd_tensor_storage(value->external) != NULL &&
            gd_storage_device(gd_tensor_storage(value->external)).type == GD_DEVICE_METAL &&
            gd_storage_device(gd_tensor_storage(value->external)).index == self->device_index &&
            value->desc.storage_offset_bytes == 0) {
            storage = gd_tensor_storage(value->external);
            status = gd_storage_retain(storage);
            if (status != GD_OK) {
                goto fail;
            }
            exe->values[i].external_alias = true;
        } else {
            sdesc = (gd_storage_desc){{GD_DEVICE_METAL, self->device_index}, GD_MEM_UNIFIED,
                                      nbytes, alignment};
            status = gd_storage_create(graph->ctx, &sdesc, &storage);
            if (status != GD_OK) {
                goto fail;
            }
        }
        exe->values[i].storage = storage;
        exe->values[i].external = value->external; /* borrowed; graph retains it */
        exe->values[i].leaf_offset =
            value->external != NULL ? (size_t)value->desc.storage_offset_bytes : 0U;
    }

    status = alias_copy_values(exe);
    if (status != GD_OK) {
        goto fail;
    }
    mark_external_writebacks(exe);
    refresh_external_flags(exe);

    /* Resolve each node's pipeline once (P8): the per-run encode loop then does
     * no NSDictionary lookups. sdpa_bwd needs a second pipeline (dkv). */
    exe->n_plan = graph->n_nodes;
    if (graph->n_nodes > 0) {
        GDMetalState *st = state_of(self);
        int j = 0;

        exe->node_pso = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso));
        exe->node_pso2 = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso2));
        if (exe->node_pso == NULL || exe->node_pso2 == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal encode plan");
            goto fail;
        }
        for (j = 0; j < graph->n_nodes; ++j) {
            _gd_op_kind op = graph->nodes[j].op;
            exe->node_pso[j] = (__bridge void *)pipeline_for(st, op);
            if (op == _GD_OP_SDPA_BWD) {
                exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_sdpa_bwd_dkv");
            }
        }
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
static bool metal_needs_stage_now(const _gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->n_values; ++i) {
        const gd_metal_value *v = &exe->values[i];
        gd_storage *src = NULL;

        if (v->external == NULL || v->external_alias) {
            continue;
        }
        src = gd_tensor_storage(v->external);
        if (src == NULL || !v->has_staged || v->staged_version != _gd_storage_version(src)) {
            return true;
        }
    }
    return false;
}

static gd_status stage_leaves(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;
    uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
    size_t bytes = 0U;
    uint64_t count = 0U;

    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *src = NULL;
        void *dst = NULL;
        size_t nbytes = 0U;
        gd_status status = GD_OK;

        if (v->external == NULL || v->external_alias) {
            continue;
        }
        src = gd_tensor_storage(v->external);
        if (src == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "metal leaf input has no storage");
        }
        nbytes = gd_storage_nbytes(v->storage);
        if (v->has_staged && v->staged_version == _gd_storage_version(src)) {
            continue;
        }
        bytes += nbytes;
        count += 1U;
        status = gd_storage_data_cpu(v->storage, &dst);
        if (status != GD_OK) {
            return status;
        }
        status = gd_storage_copy_to_cpu(self->ctx, src, v->leaf_offset, dst, nbytes);
        if (status != GD_OK) {
            return status;
        }
        v->staged_version = _gd_storage_version(src);
        v->has_staged = true;
    }
    if (start != 0U) {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_STAGE_LEAVES,
                                 _gd_profile_now_ns() - start, bytes, count);
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

/* Tiled GEMM dispatch: one threadgroup per GD_METAL_GEMM_TILE-square output
 * block, with the K-tiling loop inside the kernel. Uses fixed-size threadgroups
 * (required for the threadgroup-memory tiles) and rounds the grid up. */
static void dispatch_gemm_tiles(id<MTLComputeCommandEncoder> enc,
                                NSUInteger cols,
                                NSUInteger rows,
                                NSUInteger batch)
{
    NSUInteger tile = GD_METAL_GEMM_TILE;
    NSUInteger gx = (cols + tile - 1) / tile;
    NSUInteger gy = (rows + tile - 1) / tile;

    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, batch)
        threadsPerThreadgroup:MTLSizeMake(tile, tile, 1)];
}

/* Threadgroup size for the RMSNorm-family reductions; must match GD_RMS_TG in
 * kernels.metal (power of two for the tree reduction). */
#define GD_METAL_RMS_TG 256

/* Dispatches `groups` threadgroups of GD_METAL_RMS_TG threads (one threadgroup
 * per reduced row, or per channel tile). */
static void dispatch_reduce_groups(id<MTLComputeCommandEncoder> enc, NSUInteger groups)
{
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_TG, 1, 1)];
}

static void encode_binary(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *a_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &exe->graph->values[node->inputs[1]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_ew_params params;

    build_ew_params(&params, out_desc, a_desc, b_desc);
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_unary(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         _gd_executable *exe,
                         const _gd_node *node,
                         float scale)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_unary_params params;

    params.numel = (int)numel;
    params.scale = scale;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static gd_status encode_cast(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_cast_params params;
    gd_status status = GD_OK;

    params.numel = (int)numel;
    status = metal_dtype_code(in_desc->dtype, &params.src_dtype);
    if (status != GD_OK) {
        return status;
    }
    status = metal_dtype_code(out_desc->dtype, &params.dst_dtype);
    if (status != GD_OK) {
        return status;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

static void encode_matmul(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *a_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &exe->graph->values[node->inputs[1]].desc;
    int a_rows = (int)a_desc->sizes[a_desc->ndim - 2];
    int a_cols = (int)a_desc->sizes[a_desc->ndim - 1];
    int b_rows = (int)b_desc->sizes[b_desc->ndim - 2];
    int b_cols = (int)b_desc->sizes[b_desc->ndim - 1];
    gd_metal_matmul_params p;
    int batch_total = 1;
    int i = 0;

    memset(&p, 0, sizeof(p));
    p.m = (int)out_desc->sizes[out_desc->ndim - 2];
    p.n = (int)out_desc->sizes[out_desc->ndim - 1];
    p.k = node->attrs.trans_a ? a_rows : a_cols;
    p.a_cols = a_cols;
    p.b_cols = b_cols;
    p.a_mat = a_rows * a_cols;
    p.b_mat = b_rows * b_cols;
    p.out_mat = p.m * p.n;
    p.trans_a = node->attrs.trans_a ? 1 : 0;
    p.trans_b = node->attrs.trans_b ? 1 : 0;
    p.batch_ndim = out_desc->ndim - 2;
    p.a_batch_ndim = a_desc->ndim - 2;
    p.b_batch_ndim = b_desc->ndim - 2;
    for (i = 0; i < p.batch_ndim; ++i) {
        p.out_batch_sizes[i] = (int)out_desc->sizes[i];
        batch_total *= p.out_batch_sizes[i];
    }
    for (i = 0; i < p.a_batch_ndim; ++i) {
        p.a_batch_sizes[i] = (int)a_desc->sizes[i];
    }
    for (i = 0; i < p.b_batch_ndim; ++i) {
        p.b_batch_sizes[i] = (int)b_desc->sizes[i];
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.n > 0 && p.m > 0 && batch_total > 0) {
        dispatch_gemm_tiles(enc, (NSUInteger)p.n, (NSUInteger)p.m, (NSUInteger)batch_total);
    }
}

static void encode_linear(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_linear_params p;
    int bias_input = node->attrs.has_bias ? node->inputs[2] : node->inputs[0];

    p.in_features = (int)x_desc->sizes[x_desc->ndim - 1];
    p.out_features = (int)out_desc->sizes[out_desc->ndim - 1];
    p.rows = p.in_features > 0 ? (int)(desc_numel(x_desc) / p.in_features) : 0;
    p.trans_w = node->attrs.trans_b ? 1 : 0;
    p.has_bias = node->attrs.has_bias ? 1 : 0;

    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    /* bias is always bound (placeholder when absent; the kernel guards reads). */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.out_features > 0 && p.rows > 0) {
        dispatch_gemm_tiles(enc, (NSUInteger)p.out_features, (NSUInteger)p.rows, 1);
    }
}

/* Splits a tensor around `dim` into [outer, d, inner] extents. */
static void split_around_dim(const gd_tensor_desc *desc, int dim,
                             int *outer, int *d, int *inner)
{
    int i = 0;
    *outer = 1;
    *inner = 1;
    *d = (int)desc->sizes[dim];
    for (i = 0; i < dim; ++i) {
        *outer *= (int)desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        *inner *= (int)desc->sizes[i];
    }
}

static void encode_reduce(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_params p;

    split_around_dim(x_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = (node->op == _GD_OP_MEAN) ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static void encode_softmax(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           _gd_executable *exe,
                           const _gd_node *node)
{
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    split_around_dim(desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static void encode_rms_norm(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pso,
                            _gd_executable *exe,
                            const _gd_node *node)
{
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(desc_numel(desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.rows > 0) {
        dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

static gd_status encode_cross_entropy(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      _gd_executable *exe,
                                      const _gd_node *node)
{
    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy needs I32 targets in v1");
    }
    split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    /* Single threadgroup reduction over positions (see gd_cross_entropy). The
     * kernel's threadgroup `partial` array is sized GD_CE_TG, so clamp here. */
    {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > 256) {
            tg = 256;
        }
        if (tg == 0) {
            tg = 1;
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    return GD_OK;
}

/* relu_bwd/silu_bwd: dx from (x, go). */
static void encode_unary_bwd(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_unary_params p;

    p.numel = (int)numel;
    p.scale = 0.0F;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

/* softmax_bwd: dx from (y, go) along dim; one thread per (outer,inner). */
static void encode_softmax_bwd(id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pso,
                               _gd_executable *exe,
                               const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.outer * p.inner > 0) {
        dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

/* sum_bwd/mean_bwd: broadcast go back over dim into dx (the output/x shape). */
static void encode_sum_bwd(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           _gd_executable *exe,
                           const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_reduce_params p;

    split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = (node->op == _GD_OP_MEAN_BWD) ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static gd_status encode_cross_entropy_bwd(id<MTLComputeCommandEncoder> enc,
                                          id<MTLComputePipelineState> pso,
                                          _gd_executable *exe,
                                          const _gd_node *node)
{
    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy_bwd needs I32 targets");
    }
    split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.outer * p.inner > 0) {
        dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
    return GD_OK;
}

/* reduce_to: sum go (broadcast shape) into the target output shape. */
static void encode_reduce_to(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *target = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_to_params p;
    int i = 0;

    memset(&p, 0, sizeof(p));
    p.target_ndim = target->ndim;
    p.target_numel = (int)desc_numel(target);
    p.go_ndim = go->ndim;
    p.go_numel = (int)desc_numel(go);
    for (i = 0; i < target->ndim; ++i) {
        p.target_sizes[i] = (int)target->sizes[i];
    }
    for (i = 0; i < go->ndim; ++i) {
        p.go_sizes[i] = (int)go->sizes[i];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.target_numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)p.target_numel);
    }
}

/* step_inc: in-place ++ on a scalar buffer (no produced value). */
static void encode_step_inc(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pso,
                            _gd_executable *exe,
                            const _gd_node *node)
{
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    dispatch_1d(enc, pso, 1);
}

/* adamw: in-place update of (param, m, v) using grad and step. */
static void encode_adamw(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         _gd_executable *exe,
                         const _gd_node *node)
{
    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = desc_numel(param);
    gd_metal_adamw_params p;

    p.numel = (int)numel;
    p.lr = node->attrs.lr;
    p.beta1 = node->attrs.beta1;
    p.beta2 = node->attrs.beta2;
    p.eps = node->attrs.eps;
    p.weight_decay = node->attrs.weight_decay;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

/* gelu / gelu_bwd. fwd: (x)->out; bwd: (x, go)->dx. */
static void encode_gelu(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        _gd_executable *exe,
                        const _gd_node *node,
                        bool is_bwd)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_gelu_params p;
    int idx = 0;

    p.numel = (int)numel;
    p.tanh_approx = node->attrs.gelu_tanh ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    idx = 1;
    if (is_bwd) {
        [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        idx = 2;
    }
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:(NSUInteger)idx];
    [enc setBytes:&p length:sizeof(p) atIndex:(NSUInteger)(idx + 1)];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_transpose(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_transpose_params p;
    int64_t stride = 1;
    int k = 0;

    memset(&p, 0, sizeof(p));
    p.ndim = out_desc->ndim;
    p.numel = (int)numel;
    for (k = out_desc->ndim - 1; k >= 0; --k) {
        p.in_strides[k] = (int)stride;
        stride *= in_desc->sizes[k];
    }
    for (k = 0; k < out_desc->ndim; ++k) {
        p.out_sizes[k] = (int)out_desc->sizes[k];
        p.perm[k] = node->attrs.perm[k];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static gd_status encode_embedding(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  _gd_executable *exe,
                                  const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *table = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(numel / p.dim) : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

static gd_status encode_embedding_bwd(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      _gd_executable *exe,
                                      const _gd_node *node)
{
    const gd_tensor_desc *table = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding_bwd needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(desc_numel(go) / p.dim) : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    /* One thread per dtable element (vocab*dim). */
    {
        int total = p.vocab * p.dim;
        if (total > 0) {
            dispatch_1d(enc, pso, (NSUInteger)total);
        }
    }
    return GD_OK;
}

static gd_status encode_rope(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node,
                             float sin_sign)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *pos = &exe->graph->values[node->inputs[1]].desc;
    int64_t head_dim = out_desc->sizes[out_desc->ndim - 1];
    gd_metal_rope_params p;

    if (pos->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope needs I32 position ids in v1");
    }
    p.head_dim = (int)head_dim;
    p.heads = (int)out_desc->sizes[out_desc->ndim - 2];
    p.rows = p.head_dim > 0 ? (int)(desc_numel(out_desc) / head_dim) : 0;
    p.n_dims = node->attrs.rope_n_dims;
    p.interleaved = node->attrs.rope_interleaved;
    p.theta = node->attrs.rope_theta;
    p.sin_sign = sin_sign;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.rows > 0) {
        dispatch_1d(enc, pso, (NSUInteger)p.rows);
    }
    return GD_OK;
}

static void fill_sdpa_params(gd_metal_sdpa_params *p,
                             const gd_tensor_desc *q_desc,
                             const gd_tensor_desc *k_desc,
                             const gd_tensor_desc *bias_desc,
                             const _gd_node *node)
{
    p->B = (int)q_desc->sizes[0];
    p->Tq = (int)q_desc->sizes[1];
    p->Hq = (int)q_desc->sizes[2];
    p->Dh = (int)q_desc->sizes[3];
    p->Tk = (int)k_desc->sizes[1];
    p->Hkv = (int)k_desc->sizes[2];
    p->scale = node->attrs.attn_scale;
    p->causal = node->attrs.causal;
    p->window = node->attrs.sliding_window;
    p->has_bias = node->attrs.has_bias ? 1 : 0;
    p->Bb = bias_desc != NULL ? (int)bias_desc->sizes[0] : 1;
    p->Hb = bias_desc != NULL ? (int)bias_desc->sizes[1] : 1;
    p->Tqb = bias_desc != NULL ? (int)bias_desc->sizes[2] : 1;
    p->Tkb = bias_desc != NULL ? (int)bias_desc->sizes[3] : 1;
}

static void encode_sdpa(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        _gd_executable *exe,
                        const _gd_node *node)
{
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *bias = node->attrs.has_bias
                                     ? &exe->graph->values[node->inputs[3]].desc
                                     : NULL;
    /* bias is always bound; the kernel only reads it when has_bias. Use q as a
     * valid placeholder buffer when absent. */
    int bias_input = node->attrs.has_bias ? node->inputs[3] : node->inputs[0];
    gd_metal_sdpa_params p;
    int grid = 0;

    fill_sdpa_params(&p, q, k, bias, node);
    grid = p.B * p.Hq * p.Tq;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:3];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (grid > 0) {
        dispatch_1d(enc, pso, (NSUInteger)grid);
    }
}

/* SDPA backward: dq kernel over query rows, then dk/dv kernel over kv rows. */
static gd_status encode_sdpa_bwd(_gd_backend *self,
                                 id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> dq_pso,
                                 id<MTLComputePipelineState> dkv_pso,
                                 _gd_executable *exe,
                                 const _gd_node *node)
{
    /* inputs: go(0), q(1), k(2), v(3); outputs: dq(0), dk(1), dv(2). */
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[2]].desc;
    const gd_tensor_desc *bias = node->attrs.has_bias
                                     ? &exe->graph->values[node->inputs[4]].desc
                                     : NULL;
    int bias_input = node->attrs.has_bias ? node->inputs[4] : node->inputs[1];
    gd_metal_sdpa_params p;
    int dq_grid = 0;
    int dkv_grid = 0;

    if (dkv_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd_dkv pipeline missing");
    }
    fill_sdpa_params(&p, q, k, bias, node);
    dq_grid = p.B * p.Hq * p.Tq;
    dkv_grid = p.B * p.Hkv * p.Tk;

    [enc setComputePipelineState:dq_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:5]; /* dq */
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    if (dq_grid > 0) {
        dispatch_1d(enc, dq_pso, (NSUInteger)dq_grid);
    }

    [enc setComputePipelineState:dkv_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:value_buffer(exe, node->outputs[1]) offset:0 atIndex:5]; /* dk */
    [enc setBuffer:value_buffer(exe, node->outputs[2]) offset:0 atIndex:6]; /* dv */
    [enc setBytes:&p length:sizeof(p) atIndex:7];
    if (dkv_grid > 0) {
        dispatch_1d(enc, dkv_pso, (NSUInteger)dkv_grid);
    }
    return GD_OK;
}

static void encode_rms_norm_bwd(id<MTLComputeCommandEncoder> enc,
                                id<MTLComputePipelineState> pso,
                                _gd_executable *exe,
                                const _gd_node *node)
{
    /* dx: inputs x(0), weight(1), go(2); output dx (x shape). */
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(desc_numel(desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.rows > 0) {
        dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

static void encode_rms_norm_wbwd(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> pso,
                                 _gd_executable *exe,
                                 const _gd_node *node)
{
    /* dweight: inputs x(0), go(1); output dweight [last]. dims from x. */
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)x_desc->sizes[x_desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(desc_numel(x_desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.last > 0) {
        /* One threadgroup per channel tile of GD_METAL_RMS_TG channels. */
        NSUInteger groups = ((NSUInteger)p.last + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        dispatch_reduce_groups(enc, groups);
    }
}

static gd_status encode_node(_gd_backend *self,
                             id<MTLComputeCommandEncoder> enc,
                             _gd_executable *exe,
                             const _gd_node *node,
                             id<MTLComputePipelineState> pso,
                             id<MTLComputePipelineState> pso2)
{
    if (pso == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op");
    }

    switch (node->op) {
    case _GD_OP_ADD:
    case _GD_OP_MUL:
        encode_binary(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SCALE:
        encode_unary(enc, pso, exe, node, node->attrs.scale);
        return GD_OK;
    case _GD_OP_RELU:
    case _GD_OP_SILU:
        encode_unary(enc, pso, exe, node, 0.0F);
        return GD_OK;
    case _GD_OP_COPY:
        if (node->n_inputs == 1 && node->n_outputs == 1 &&
            exe->values[node->inputs[0]].storage == exe->values[node->outputs[0]].storage) {
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_COPY_ALIAS,
                                     0U, 0U, 1U);
            return GD_OK;
        }
        encode_unary(enc, pso, exe, node, 0.0F);
        return GD_OK;
    case _GD_OP_CAST:
        return encode_cast(enc, pso, exe, node);
    case _GD_OP_MATMUL:
        encode_matmul(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_LINEAR:
        encode_linear(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SUM:
    case _GD_OP_MEAN:
        encode_reduce(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SOFTMAX:
        encode_softmax(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_RMS_NORM:
        encode_rms_norm(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_CROSS_ENTROPY:
        return encode_cross_entropy(enc, pso, exe, node);
    case _GD_OP_RELU_BWD:
    case _GD_OP_SILU_BWD:
        encode_unary_bwd(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SOFTMAX_BWD:
        encode_softmax_bwd(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SUM_BWD:
    case _GD_OP_MEAN_BWD:
        encode_sum_bwd(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_CROSS_ENTROPY_BWD:
        return encode_cross_entropy_bwd(enc, pso, exe, node);
    case _GD_OP_REDUCE_TO:
        encode_reduce_to(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_STEP_INC:
        encode_step_inc(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_ADAMW_STEP:
        encode_adamw(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_GELU:
        encode_gelu(enc, pso, exe, node, false);
        return GD_OK;
    case _GD_OP_GELU_BWD:
        encode_gelu(enc, pso, exe, node, true);
        return GD_OK;
    case _GD_OP_TRANSPOSE:
        encode_transpose(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_EMBEDDING:
        return encode_embedding(enc, pso, exe, node);
    case _GD_OP_EMBEDDING_BWD:
        return encode_embedding_bwd(enc, pso, exe, node);
    case _GD_OP_ROPE:
        return encode_rope(enc, pso, exe, node, 1.0F);
    case _GD_OP_ROPE_BWD:
        return encode_rope(enc, pso, exe, node, -1.0F);
    case _GD_OP_SDPA:
        encode_sdpa(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_SDPA_BWD:
        return encode_sdpa_bwd(self, enc, pso, pso2, exe, node);
    case _GD_OP_RMS_NORM_BWD:
        encode_rms_norm_bwd(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_RMS_NORM_WBWD:
        encode_rms_norm_wbwd(enc, pso, exe, node);
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "metal op not implemented yet");
    }
}

static gd_status metal_execute_range(_gd_backend *self, _gd_executable *exe, int last_node)
{
    GDMetalState *st = state_of(self);
    gd_status status = GD_OK;
    int i = 0;

    /* If CPU-backed leaves changed, any prior command buffer using their shadow
     * buffers must complete before the CPU mutates those buffers (Metal coherency).
     * If nothing changed, the next command buffer can be queued immediately. */
    if (exe->needs_stage && metal_needs_stage_now(exe)) {
        status = metal_synchronize(self);
        if (status != GD_OK) {
            return status;
        }
        status = stage_leaves(self, exe);
        if (status != GD_OK) {
            return status;
        }
    } else {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_STAGE_LEAVES,
                                 0U, 0U, 0U);
    }

    /* Trace mode: encode one node per command buffer and wait on each, so the
     * host-measured time is attributed to that op kind. Serializing like this is
     * far slower than normal execution and is for profiling only. */
    if (_gd_profile_trace_enabled(self->ctx)) {
        status = metal_synchronize(self);
        if (status != GD_OK) {
            return status;
        }
        for (i = 0; i <= last_node && i < exe->graph->n_nodes; ++i) {
            const _gd_node *node = &exe->graph->nodes[i];
            id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
            id<MTLComputePipelineState> pso2 = (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
            uint64_t op_start = _gd_profile_now_ns();
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                status = encode_node(self, enc, exe, node, pso, pso2);
                if (status != GD_OK) {
                    [enc endEncoding];
                    return status;
                }
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                if (cmd.status == MTLCommandBufferStatusError) {
                    return _gd_error(GD_ERR_BACKEND, "metal trace command buffer failed");
                }
            }
            _gd_profile_record_op_time(self->ctx, self, (int)node->op,
                                       _gd_profile_now_ns() - op_start);
        }
        st.inFlight = nil;
        if (exe->needs_writeback) {
            return writeback_externals(self, exe);
        }
        return GD_OK;
    }

    {
        uint64_t encode_start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            for (i = 0; i <= last_node && i < exe->graph->n_nodes; ++i) {
                id<MTLComputePipelineState> pso =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
                id<MTLComputePipelineState> pso2 =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
                status = encode_node(self, enc, exe, &exe->graph->nodes[i], pso, pso2);
                if (status != GD_OK) {
                    [enc endEncoding];
                    return status;
                }
            }
            [enc endEncoding];
            [cmd commit];
            st.inFlight = cmd;
        }
        if (encode_start != 0U) {
            uint64_t count = 0U;
            if (last_node >= 0) {
                int capped = last_node < exe->graph->n_nodes ? last_node : exe->graph->n_nodes - 1;
                count = capped >= 0 ? (uint64_t)capped + 1U : 0U;
            }
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_ENCODE,
                                     _gd_profile_now_ns() - encode_start, 0U, count);
        }
    }
    if (exe->needs_writeback) {
        /* P4: defer the wait + device->host writeback. The mutated CPU-backed
         * leaves are marked pending; a host read (gd_tensor_copy_to_cpu,
         * gd_storage_data_cpu) or gd_synchronize triggers the flush. P5's
         * skip-unchanged staging keeps the shadow buffers authoritative across
         * runs, so deferring does not lose in-place updates. */
        register_pending_writeback(self, exe);
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
        st.pipelinesByName = [NSMutableDictionary dictionary];
        st.inFlight = nil;
        st.pendingExes = [NSMutableArray array];

        for (k = 0; k < sizeof(g_metal_kernels) / sizeof(g_metal_kernels[0]); ++k) {
            NSString *name = [NSString stringWithUTF8String:g_metal_kernels[k].fn];
            id<MTLFunction> fn = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pso = nil;

            if (fn == nil) {
                return _gd_error(GD_ERR_BACKEND, "metallib is missing a required kernel");
            }
            pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "failed to create compute pipeline state");
            }
            st.pipelines[@((int)g_metal_kernels[k].op)] = pso;
            st.pipelinesByName[name] = pso;
        }
        for (k = 0; k < sizeof(g_metal_extra_kernels) / sizeof(g_metal_extra_kernels[0]); ++k) {
            NSString *name = [NSString stringWithUTF8String:g_metal_extra_kernels[k]];
            id<MTLFunction> fn = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pso = nil;

            if (fn == nil) {
                return _gd_error(GD_ERR_BACKEND, "metallib is missing a required kernel");
            }
            pso = [device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "failed to create compute pipeline state");
            }
            st.pipelinesByName[name] = pso;
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
        [st.pendingExes removeAllObjects];
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
    .flush_pending = metal_flush_pending,
};

gd_status _gd_metal_backend_register(gd_context *ctx)
{
    return _gd_context_register_backend(ctx, &metal_backend_vtable);
}
