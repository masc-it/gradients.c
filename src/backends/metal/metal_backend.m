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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <limits.h>
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
/* Opt-in MPSMatrixMultiplication path (`GD_METAL_MPS=1`). */
@property BOOL useMPS;
@end

@implementation GDMetalState
@end

@interface GDMPSGemmPlan : NSObject
@property (strong) MPSMatrixMultiplication *kernel;
@property (strong) MPSMatrix *left;
@property (strong) MPSMatrix *right;
@property (strong) MPSMatrix *result;
@end

@implementation GDMPSGemmPlan
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
    {_GD_OP_SDPA, "gd_sdpa_tiled"},
    {_GD_OP_SDPA_BWD, "gd_sdpa_bwd_dq"},
    {_GD_OP_RMS_NORM_BWD, "gd_rms_norm_bwd"},
    {_GD_OP_RMS_NORM_WBWD, "gd_rms_norm_wbwd"},
};

/* Kernels not mapped 1:1 to an op (looked up by name during encode). */
static const char *const g_metal_extra_kernels[] = {
    "gd_sdpa_bwd_dkv",
    "gd_sdpa_bwd_stats",
    "gd_sdpa", /* reference forward; fallback when head_dim > GD_METAL_SDPA_DHT */
    "gd_sdpa_splitk",  /* split-K forward (long context) */
    "gd_sdpa_combine", /* split-K combine pass */
    "gd_sdpa_bwd_stats_dq_split",
    "gd_sdpa_bwd_stats_dq_combine",
    "gd_sdpa_bwd_dkv_split",
    "gd_sdpa_bwd_dkv_reduce",
    "gd_silu_mul", /* F1: fused SwiGLU activation */
    "gd_cross_entropy_reduce",
    "gd_lmce_fwd_chunk",
    "gd_lmce_loss_rows",
    "gd_lmce_dlogits_chunk",
    "gd_embedding_bwd_scatter",
    "gd_rms_norm_wbwd_reduce",
};

/* Threadgroup size for the RMSNorm-family reductions; must match GD_RMS_TG in
 * kernels.metal (power of two for the tree reduction). */
#define GD_METAL_RMS_TG 256

/* Backward split-K scratch is one buffer carved into four regions (float counts
 * and offsets). When n_splits==1 only the stats region is used (the original
 * single-pass path). Keep in sync with the kernel partial layouts. */
typedef struct {
    int64_t stats_off;      /* final (m,l,D): B*Hq*Tq*3 */
    int64_t stats_part_off; /* per-split (m,l,raw): B*Hq*Tq*S*3 */
    int64_t dq_part_off;    /* per-split dq fused partials (acc,ks): B*Hq*Tq*S*2*Dh */
    int64_t dkv_part_off;   /* per-split dk,dv: B*Hkv*Tk*S*2*Dh */
    int64_t total;          /* total floats */
} sdpa_bwd_layout;

static sdpa_bwd_layout sdpa_bwd_scratch_layout(int B, int Hq, int Hkv,
                                               int Tq, int Tk, int Dh, int S)
{
    sdpa_bwd_layout o;
    int64_t off = 0;
    o.stats_off = off;
    off += (int64_t)B * Hq * Tq * 3;
    o.stats_part_off = off;
    off += (int64_t)B * Hq * Tq * S * 3;
    o.dq_part_off = off;
    off += (int64_t)B * Hq * Tq * S * 2 * Dh;
    o.dkv_part_off = off;
    off += (int64_t)B * Hkv * Tk * S * 2 * Dh;
    o.total = off;
    return o;
}

/* Split-K key partitioning for the forward SDPA, derived purely from Tk so the
 * compile-time scratch size and the encode-time dispatch agree. Returns 1 (no
 * split) for short sequences, keeping them on the single-pass tiled kernel. */
static int sdpa_num_splits(int Tk)
{
    int s = (Tk + GD_METAL_SDPA_SPLIT_MIN - 1) / GD_METAL_SDPA_SPLIT_MIN;
    if (s < 1) {
        s = 1;
    }
    if (s > GD_METAL_SDPA_SPLIT_MAX) {
        s = GD_METAL_SDPA_SPLIT_MAX;
    }
    return s;
}

/* Keys per split, rounded up to a GD_METAL_SDPA_BK multiple so each partition
 * starts on a key-tile boundary. */
static int sdpa_split_len(int Tk, int n_splits)
{
    int len = (Tk + n_splits - 1) / n_splits;
    len = ((len + GD_METAL_SDPA_BK - 1) / GD_METAL_SDPA_BK) * GD_METAL_SDPA_BK;
    if (len < GD_METAL_SDPA_BK) {
        len = GD_METAL_SDPA_BK;
    }
    return len;
}

typedef struct lmce_scratch_layout {
    size_t logits_off;
    size_t m_off;
    size_t l_off;
    size_t target_logit_off;
    size_t losses_off;
    size_t total;
} lmce_scratch_layout;

static lmce_scratch_layout lmce_scratch_layout_for(int rows, int chunk)
{
    lmce_scratch_layout L;
    size_t off = 0U;
    L.logits_off = off;
    off += (size_t)rows * (size_t)chunk * sizeof(float);
    L.m_off = off;
    off += (size_t)rows * sizeof(float);
    L.l_off = off;
    off += (size_t)rows * sizeof(float);
    L.target_logit_off = off;
    off += (size_t)rows * sizeof(float);
    L.losses_off = off;
    off += (size_t)rows * sizeof(float);
    L.total = off;
    return L;
}

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
    void **node_pso3;  /* tertiary pipeline (e.g. sdpa_bwd stats), or NULL */
    void **node_mps;   /* optional per-node MPS kernel object, retained, or NULL */
    size_t *node_scratch_bytes; /* per-node scratch need inside shared arena */
    gd_storage *scratch_arena;  /* max-sized executable scratch, reused per node */
    size_t scratch_arena_bytes;
    /* Fusion plan (GPU_FUSED, Option A): a node marked absorbed is encoded by its
     * group head, so execute skips it. Set by metal_plan_fusions at compile;
     * all-zero means no fusion (the identity plan). */
    uint8_t *node_absorbed;
    /* For a fused head node, the node id of the producer it absorbs (or -1).
     * encode routes such heads to their fused kernel; the fused pipeline lives
     * in node_pso2[head]. */
    int *node_fused_src;
};

/* White-box counter so tests can assert a fusion actually engaged (a silent
 * fallback to the unfused path would otherwise pass parity unnoticed). Declared
 * here (no public header); tests reference it with a local extern. */
static unsigned long g_metal_fusions_applied = 0UL;
unsigned long _gd_metal_fusions_applied(void);
unsigned long _gd_metal_fusions_applied(void) { return g_metal_fusions_applied; }

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
    /* node_pso* hold unretained __bridge pointers; just free the arrays. */
    free(exe->node_pso);
    free(exe->node_pso2);
    free(exe->node_pso3);
    if (exe->node_mps != NULL) {
        for (i = 0; i < exe->n_plan; ++i) {
            if (exe->node_mps[i] != NULL) {
                CFRelease(exe->node_mps[i]);
            }
        }
        free(exe->node_mps);
    }
    free(exe->node_absorbed);
    free(exe->node_fused_src);
    free(exe->node_scratch_bytes);
    gd_storage_release(exe->scratch_arena);
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

/* Fusion planner (GPU_FUSED, Option A). Recognizes fusible node groups in the
 * (op-granular, shared) IR and marks absorbed nodes so `execute` skips them,
 * binding each group head to a fused kernel. CPU_REF still runs the unfused
 * nodes as the oracle; parity holds at group boundaries.
 *
 * Phase F0 ships this hook with NO patterns: node_absorbed stays all-zero, so
 * the executable runs every node exactly as the unfused plan (identity pass).
 * Patterns are added incrementally; see docs/metal_gpu_fuse.md. */
/* Returns the lowest node index that consumes `value_id`, or INT_MAX if none.
 * Used as the legality gate when a fused head relocates an intermediate's
 * production to its own (later) position: every consumer must run at or after
 * the head, else it would read the value before the fused kernel writes it. */
static int value_min_consumer(const gd_graph *graph, int value_id)
{
    int best = INT_MAX;
    int i = 0;
    for (i = 0; i < graph->n_nodes; ++i) {
        const _gd_node *node = &graph->nodes[i];
        int j = 0;
        for (j = 0; j < node->n_inputs; ++j) {
            if (node->inputs[j] == value_id) {
                if (i < best) {
                    best = i;
                }
                break;
            }
        }
    }
    return best;
}

/* F1 pattern: silu(gate) -> mul(., up). The mul (head) absorbs the silu and is
 * encoded by gd_silu_mul, which writes both hh = silu(gate)*up and act =
 * silu(gate). act stays materialized (the unfused backward reads it), so the
 * win is one fewer dispatch plus the mul no longer reloading act from DRAM.
 * Legality: head is MUL; its first input is produced by a single-output SILU;
 * silu's act is not consumed by any node before the head (so relocating its
 * production to the head is safe); inputs share the head's shape (no
 * broadcast). */
static bool try_fuse_silu_mul(GDMetalState *st, const gd_graph *graph,
                              _gd_executable *exe, int head_idx)
{
    const _gd_node *head = &graph->nodes[head_idx];
    int act_id = 0;
    int silu_idx = 0;
    const _gd_node *silu = NULL;
    const gd_tensor_desc *out_d = NULL;
    const gd_tensor_desc *up_d = NULL;

    if (head->op != _GD_OP_MUL || head->n_inputs != 2 || head->n_outputs != 1) {
        return false;
    }
    act_id = head->inputs[0];
    if (act_id < 0 || act_id >= exe->n_values) {
        return false;
    }
    silu_idx = exe->graph->values[act_id].producer_node_id;
    if (silu_idx < 0 || silu_idx >= graph->n_nodes) {
        return false;
    }
    silu = &graph->nodes[silu_idx];
    if (silu->op != _GD_OP_SILU || silu->n_inputs != 1 || silu->n_outputs != 1 ||
        silu->outputs[0] != act_id) {
        return false;
    }
    /* No node may read `act` before the head (production is relocated to head). */
    if (value_min_consumer(graph, act_id) < head_idx) {
        return false;
    }
    /* Equal shapes: the fused kernel is contiguous numel, no broadcast. */
    out_d = &exe->graph->values[head->outputs[0]].desc;
    up_d = &exe->graph->values[head->inputs[1]].desc;
    if (desc_numel(out_d) != desc_numel(up_d) ||
        desc_numel(out_d) != desc_numel(&exe->graph->values[silu->inputs[0]].desc)) {
        return false;
    }
    exe->node_pso2[head_idx] = (__bridge void *)pipeline_named(st, "gd_silu_mul");
    if (exe->node_pso2[head_idx] == NULL) {
        return false; /* kernel missing: fall back to unfused */
    }
    exe->node_fused_src[head_idx] = silu_idx;
    exe->node_absorbed[silu_idx] = 1;
    g_metal_fusions_applied++;
    return true;
}

static void metal_plan_fusions(_gd_backend *self, const gd_graph *graph,
                               _gd_executable *exe)
{
    GDMetalState *st = state_of(self);
    int i = 0;
    for (i = 0; i < graph->n_nodes; ++i) {
        if (exe->node_absorbed[i]) {
            continue;
        }
        (void)try_fuse_silu_mul(st, graph, exe, i);
    }
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
        size_t scratch_max_bytes = 0U;

        exe->node_pso = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso));
        exe->node_pso2 = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso2));
        exe->node_pso3 = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso3));
        exe->node_mps = calloc((size_t)graph->n_nodes, sizeof(*exe->node_mps));
        exe->node_scratch_bytes = calloc((size_t)graph->n_nodes,
                                         sizeof(*exe->node_scratch_bytes));
        exe->node_absorbed = calloc((size_t)graph->n_nodes, sizeof(*exe->node_absorbed));
        exe->node_fused_src = calloc((size_t)graph->n_nodes, sizeof(*exe->node_fused_src));
        if (exe->node_pso == NULL || exe->node_pso2 == NULL || exe->node_pso3 == NULL ||
            exe->node_mps == NULL || exe->node_scratch_bytes == NULL ||
            exe->node_absorbed == NULL || exe->node_fused_src == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal encode plan");
            goto fail;
        }
        memset(exe->node_fused_src, 0xFF, (size_t)graph->n_nodes * sizeof(*exe->node_fused_src)); /* -1 */
        for (j = 0; j < graph->n_nodes; ++j) {
            const _gd_node *node = &graph->nodes[j];
            _gd_op_kind op = node->op;
            exe->node_pso[j] = (__bridge void *)pipeline_for(st, op);
            if (st.useMPS && (op == _GD_OP_LINEAR || op == _GD_OP_MATMUL)) {
                const gd_tensor_desc *a_desc = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *b_desc = &graph->values[node->inputs[1]].desc;
                const gd_tensor_desc *out_desc = &graph->values[node->outputs[0]].desc;
                if (a_desc->dtype == GD_DTYPE_F32 && b_desc->dtype == GD_DTYPE_F32 &&
                    out_desc->dtype == GD_DTYPE_F32 && a_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    b_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    out_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    a_desc->storage_offset_bytes == 0 && b_desc->storage_offset_bytes == 0 &&
                    out_desc->storage_offset_bytes == 0) {
                    bool ok = false;
                    bool trans_left = false;
                    bool trans_right = node->attrs.trans_b ? true : false;
                    int result_rows = 0;
                    int result_cols = 0;
                    int inner = 0;
                    NSUInteger batch = 1U;
                    NSUInteger a_rows_desc = 0U;
                    NSUInteger a_cols_desc = 0U;
                    NSUInteger b_rows_desc = 0U;
                    NSUInteger b_cols_desc = 0U;
                    NSUInteger out_rows_desc = 0U;
                    NSUInteger out_cols_desc = 0U;
                    if (op == _GD_OP_LINEAR && !node->attrs.has_bias) {
                        inner = (int)a_desc->sizes[a_desc->ndim - 1];
                        result_cols = (int)out_desc->sizes[out_desc->ndim - 1];
                        result_rows = inner > 0 ? (int)(desc_numel(a_desc) / inner) : 0;
                        a_rows_desc = (NSUInteger)result_rows;
                        a_cols_desc = (NSUInteger)inner;
                        b_rows_desc = (NSUInteger)b_desc->sizes[0];
                        b_cols_desc = (NSUInteger)b_desc->sizes[1];
                        out_rows_desc = (NSUInteger)result_rows;
                        out_cols_desc = (NSUInteger)result_cols;
                        ok = true;
                    } else if (op == _GD_OP_MATMUL && !node->attrs.trans_a && b_desc->ndim == 2) {
                        int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                        int b_rows_i = (int)b_desc->sizes[0];
                        int b_cols_i = (int)b_desc->sizes[1];
                        inner = a_cols_i;
                        result_cols = trans_right ? b_rows_i : b_cols_i;
                        result_rows = inner > 0 ? (int)(desc_numel(a_desc) / inner) : 0;
                        a_rows_desc = (NSUInteger)result_rows;
                        a_cols_desc = (NSUInteger)inner;
                        b_rows_desc = (NSUInteger)b_rows_i;
                        b_cols_desc = (NSUInteger)b_cols_i;
                        out_rows_desc = (NSUInteger)result_rows;
                        out_cols_desc = (NSUInteger)result_cols;
                        ok = (out_desc->sizes[out_desc->ndim - 1] == result_cols);
                    } else if (op == _GD_OP_MATMUL && out_desc->ndim >= 3 &&
                               a_desc->ndim == out_desc->ndim &&
                               b_desc->ndim == out_desc->ndim) {
                        bool same_batch = true;
                        int batch_ndim = out_desc->ndim - 2;
                        int a_rows_i = (int)a_desc->sizes[a_desc->ndim - 2];
                        int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                        int b_rows_i = (int)b_desc->sizes[b_desc->ndim - 2];
                        int b_cols_i = (int)b_desc->sizes[b_desc->ndim - 1];
                        int i_batch = 0;
                        for (i_batch = 0; i_batch < batch_ndim; ++i_batch) {
                            if (a_desc->sizes[i_batch] != out_desc->sizes[i_batch] ||
                                b_desc->sizes[i_batch] != out_desc->sizes[i_batch]) {
                                same_batch = false;
                                break;
                            }
                            batch *= (NSUInteger)out_desc->sizes[i_batch];
                        }
                        trans_left = node->attrs.trans_a ? true : false;
                        inner = trans_left ? a_rows_i : a_cols_i;
                        result_rows = trans_left ? a_cols_i : a_rows_i;
                        result_cols = trans_right ? b_rows_i : b_cols_i;
                        a_rows_desc = (NSUInteger)a_rows_i;
                        a_cols_desc = (NSUInteger)a_cols_i;
                        b_rows_desc = (NSUInteger)b_rows_i;
                        b_cols_desc = (NSUInteger)b_cols_i;
                        out_rows_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 2];
                        out_cols_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 1];
                        ok = same_batch && inner == (trans_right ? b_cols_i : b_rows_i) &&
                             out_desc->sizes[out_desc->ndim - 2] == result_rows &&
                             out_desc->sizes[out_desc->ndim - 1] == result_cols;
                    }
                    if (ok && result_rows > 0 && result_cols > 0 && inner > 0 && batch > 0U) {
                        NSUInteger a_row_bytes = a_cols_desc * sizeof(float);
                        NSUInteger b_row_bytes = b_cols_desc * sizeof(float);
                        NSUInteger out_row_bytes = out_cols_desc * sizeof(float);
                        MPSMatrixDescriptor *ad = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:a_rows_desc
                                             columns:a_cols_desc
                                            matrices:batch
                                            rowBytes:a_row_bytes
                                         matrixBytes:a_rows_desc * a_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor *bd = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:b_rows_desc
                                             columns:b_cols_desc
                                            matrices:batch
                                            rowBytes:b_row_bytes
                                         matrixBytes:b_rows_desc * b_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor *od = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:out_rows_desc
                                             columns:out_cols_desc
                                            matrices:batch
                                            rowBytes:out_row_bytes
                                         matrixBytes:out_rows_desc * out_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        GDMPSGemmPlan *plan = [GDMPSGemmPlan new];
                        plan.kernel = [[MPSMatrixMultiplication alloc]
                            initWithDevice:st.device
                            transposeLeft:trans_left
                            transposeRight:trans_right
                            resultRows:(NSUInteger)result_rows
                            resultColumns:(NSUInteger)result_cols
                            interiorColumns:(NSUInteger)inner
                            alpha:1.0
                            beta:0.0];
                        plan.kernel.batchStart = 0U;
                        plan.kernel.batchSize = batch;
                        plan.left = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[0]].storage)
                                    offset:0
                                descriptor:ad];
                        plan.right = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[1]].storage)
                                    offset:0
                                descriptor:bd];
                        plan.result = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->outputs[0]].storage)
                                    offset:0
                                descriptor:od];
                        if (plan.kernel != nil && plan.left != nil && plan.right != nil &&
                            plan.result != nil) {
                            exe->node_mps[j] = (void *)CFBridgingRetain(plan);
                        }
                    }
                }
            }
            if (op == _GD_OP_RMS_NORM_WBWD) {
                const gd_tensor_desc *x_desc = &graph->values[node->inputs[0]].desc;
                int last = (int)x_desc->sizes[x_desc->ndim - 1];
                int rows = last > 0 ? (int)(desc_numel(x_desc) / last) : 0;
                int row_blocks = (rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
                exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_rms_norm_wbwd_reduce");
                if (row_blocks > 0 && last > 0) {
                    exe->node_scratch_bytes[j] = (size_t)row_blocks * (size_t)last *
                                                 sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_EMBEDDING_BWD) {
                exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_embedding_bwd_scatter");
            } else if (op == _GD_OP_CROSS_ENTROPY) {
                const gd_tensor_desc *logits = &graph->values[node->inputs[0]].desc;
                int dim = node->attrs.dim;
                int64_t positions = 0;
                if (dim < 0) {
                    dim += logits->ndim;
                }
                if (dim >= 0 && dim < logits->ndim && logits->sizes[dim] > 0) {
                    positions = desc_numel(logits) / logits->sizes[dim];
                }
                exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_cross_entropy_reduce");
                if (positions > 0) {
                    exe->node_scratch_bytes[j] = (size_t)positions * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_LM_CROSS_ENTROPY ||
                       op == _GD_OP_LM_CROSS_ENTROPY_BWD) {
                const gd_tensor_desc *x_desc = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *w_desc = &graph->values[node->inputs[1]].desc;
                int D = (int)x_desc->sizes[x_desc->ndim - 1];
                int V = (int)w_desc->sizes[0];
                int rows = D > 0 ? (int)(desc_numel(x_desc) / D) : 0;
                int chunk = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
                if (rows > 0 && chunk > 0) {
                    exe->node_scratch_bytes[j] = lmce_scratch_layout_for(rows, chunk).total;
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_SDPA) {
                /* Long-context forward uses split-K: scratch holds per-split
                 * partial (acc[Dh], m, l) for each (b, hq, query). Only allocated
                 * when the tiled path is eligible (Dh <= DHT) and >1 split. */
                const gd_tensor_desc *qd = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *kd = &graph->values[node->inputs[1]].desc;
                int Dh = (int)qd->sizes[3];
                int Tk = (int)kd->sizes[1];
                int nsplit = sdpa_num_splits(Tk);
                if (Dh <= GD_METAL_SDPA_DHT && nsplit > 1) {
                    int64_t npart = qd->sizes[0] * qd->sizes[2] * qd->sizes[1]
                                    * nsplit * (Dh + 2);
                    exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_sdpa_splitk");
                    exe->node_pso3[j] = (__bridge void *)pipeline_named(st, "gd_sdpa_combine");
                    exe->node_scratch_bytes[j] = (size_t)npart * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_SDPA_BWD) {
                exe->node_pso2[j] = (__bridge void *)pipeline_named(st, "gd_sdpa_bwd_dkv");
                exe->node_pso3[j] = (__bridge void *)pipeline_named(st, "gd_sdpa_bwd_stats");
                /* Scratch holds the per-query stats [m, l, D] (B*Hq*Tq*3). For
                 * long context the split-K path additionally carves per-split
                 * partial regions out of the same buffer; size it accordingly.
                 * q is input 1: [B, Tq, Hq, Dh]; k is input 2: [B, Tk, Hkv, Dh]. */
                {
                    const gd_tensor_desc *qd = &graph->values[node->inputs[1]].desc;
                    const gd_tensor_desc *kd = &graph->values[node->inputs[2]].desc;
                    int B = (int)qd->sizes[0];
                    int Tq = (int)qd->sizes[1];
                    int Hq = (int)qd->sizes[2];
                    int Dh = (int)qd->sizes[3];
                    int Tk = (int)kd->sizes[1];
                    int Hkv = (int)kd->sizes[2];
                    int nsplit = sdpa_num_splits(Tk > Tq ? Tk : Tq);
                    int64_t nfloats;
                    if (Dh <= GD_METAL_SDPA_DHT && nsplit > 1) {
                        nfloats = sdpa_bwd_scratch_layout(B, Hq, Hkv, Tq, Tk, Dh, nsplit).total;
                    } else {
                        nfloats = (int64_t)B * Hq * Tq * 3;
                    }
                    exe->node_scratch_bytes[j] = (size_t)nfloats * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            }
        }
        if (scratch_max_bytes > 0U) {
            gd_storage_desc sdesc = {{GD_DEVICE_METAL, self->device_index}, GD_MEM_UNIFIED,
                                     scratch_max_bytes, sizeof(float)};
            status = gd_storage_create(graph->ctx, &sdesc, &exe->scratch_arena);
            if (status != GD_OK) {
                goto fail;
            }
            exe->scratch_arena_bytes = scratch_max_bytes;
        }
        metal_plan_fusions(self, graph, exe);
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
    p->same_shape = (a_desc->ndim == out_desc->ndim && b_desc->ndim == out_desc->ndim) ? 1 : 0;
    for (i = 0; i < a_desc->ndim; ++i) {
        if (a_desc->sizes[i] != out_desc->sizes[i]) {
            p->same_shape = 0;
        }
    }
    for (i = 0; i < b_desc->ndim; ++i) {
        p->b_sizes[i] = (int)b_desc->sizes[i];
        if (b_desc->sizes[i] != out_desc->sizes[i]) {
            p->same_shape = 0;
        }
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
    NSUInteger gx = (cols + GD_METAL_GEMM_BN - 1) / GD_METAL_GEMM_BN;
    NSUInteger gy = (rows + GD_METAL_GEMM_BM - 1) / GD_METAL_GEMM_BM;
    NSUInteger tx = GD_METAL_GEMM_BN / GD_METAL_GEMM_TN;
    NSUInteger ty = GD_METAL_GEMM_BM / GD_METAL_GEMM_TM;

    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, batch)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
}

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

/* F1 fused SwiGLU: encode gd_silu_mul for a MUL head that absorbed a SILU.
 * gate = silu input, up = head input 1, hh = head output, act = silu output
 * (kept materialized for the unfused backward). */
static void encode_fused_silu_mul(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  _gd_executable *exe,
                                  const _gd_node *head,
                                  const _gd_node *silu)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[head->outputs[0]].desc;
    int64_t numel = desc_numel(out_desc);
    gd_metal_unary_params params;

    params.numel = (int)numel;
    params.scale = 0.0F;
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, silu->inputs[0]) offset:0 atIndex:0];  /* gate */
    [enc setBuffer:value_buffer(exe, head->inputs[1]) offset:0 atIndex:1];  /* up */
    [enc setBuffer:value_buffer(exe, head->outputs[0]) offset:0 atIndex:2]; /* hh */
    [enc setBuffer:value_buffer(exe, silu->outputs[0]) offset:0 atIndex:3]; /* act */
    [enc setBytes:&params length:sizeof(params) atIndex:4];
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

static gd_status encode_mps_gemm(id<MTLCommandBuffer> cmd,
                                 id<MTLComputeCommandEncoder> *enc,
                                 GDMPSGemmPlan *plan)
{
    if (cmd == nil || enc == NULL || *enc == nil || plan == nil || plan.kernel == nil ||
        plan.left == nil || plan.right == nil || plan.result == nil) {
        return _gd_error(GD_ERR_BACKEND, "invalid MPS GEMM encode");
    }
    [*enc endEncoding];
    *enc = nil;
    [plan.kernel encodeToCommandBuffer:cmd
                            leftMatrix:plan.left
                           rightMatrix:plan.right
                          resultMatrix:plan.result];
    *enc = [cmd computeCommandEncoder];
    if (*enc == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to resume compute encoder after MPS GEMM");
    }
    return GD_OK;
}

static MPSMatrix *mps_matrix(id<MTLBuffer> buffer,
                             NSUInteger offset,
                             NSUInteger rows,
                             NSUInteger cols,
                             NSUInteger row_bytes)
{
    MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                    columns:cols
                                                                   rowBytes:row_bytes
                                                                   dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:d];
}

static gd_status encode_mps_mm(id<MTLCommandBuffer> cmd,
                               id<MTLComputeCommandEncoder> *enc,
                               id<MTLDevice> device,
                               MPSMatrix *left,
                               MPSMatrix *right,
                               MPSMatrix *result,
                               BOOL trans_left,
                               BOOL trans_right,
                               NSUInteger rows,
                               NSUInteger cols,
                               NSUInteger inner,
                               double beta)
{
    if (cmd == nil || enc == NULL || *enc == nil || device == nil || left == nil ||
        right == nil || result == nil || rows == 0U || cols == 0U || inner == 0U) {
        return _gd_error(GD_ERR_BACKEND, "invalid dynamic MPS GEMM encode");
    }
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                    transposeLeft:trans_left
                                                                   transposeRight:trans_right
                                                                       resultRows:rows
                                                                    resultColumns:cols
                                                                  interiorColumns:inner
                                                                            alpha:1.0
                                                                             beta:beta];
    if (mm == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to create dynamic MPS GEMM");
    }
    [*enc endEncoding];
    *enc = nil;
    [mm encodeToCommandBuffer:cmd leftMatrix:left rightMatrix:right resultMatrix:result];
    *enc = [cmd computeCommandEncoder];
    if (*enc == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to resume compute encoder after dynamic MPS GEMM");
    }
    return GD_OK;
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
                                      id<MTLComputePipelineState> reduce_pso,
                                      id<MTLBuffer> scratch,
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
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal cross_entropy scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        if (p.positions > 0) {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.positions, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        }
    }

    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    {
        NSUInteger tg = reduce_pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
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
    /* One threadgroup per position; threads cooperate over the class dim (see
     * gd_cross_entropy_bwd). Power-of-two threadgroup width for the reductions. */
    if (p.outer * p.inner > 0) {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(p.outer * p.inner), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return GD_OK;
}

static gd_status encode_lm_cross_entropy(_gd_backend *self,
                                         id<MTLCommandBuffer> cmd,
                                         id<MTLComputeCommandEncoder> *enc,
                                         id<MTLBuffer> scratch,
                                         _gd_executable *exe,
                                         const _gd_node *node)
{
    GDMetalState *st = state_of(self);
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[2]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(desc_numel(x_desc) / D) : 0;
    int chunk_max = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    lmce_scratch_layout L = lmce_scratch_layout_for(rows, chunk_max);
    id<MTLBuffer> x_b = value_buffer(exe, node->inputs[0]);
    id<MTLBuffer> w_b = value_buffer(exe, node->inputs[1]);
    id<MTLBuffer> t_b = value_buffer(exe, node->inputs[2]);
    id<MTLBuffer> out_b = value_buffer(exe, node->outputs[0]);
    id<MTLComputePipelineState> fwd_pso = pipeline_named(st, "gd_lmce_fwd_chunk");
    id<MTLComputePipelineState> loss_pso = pipeline_named(st, "gd_lmce_loss_rows");
    id<MTLComputePipelineState> reduce_pso = pipeline_named(st, "gd_cross_entropy_reduce");

    if (!st.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs GD_METAL_MPS=1");
    }
    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs I32 targets");
    }
    if (scratch == nil || fwd_pso == nil || loss_pso == nil || reduce_pso == nil ||
        rows <= 0 || D <= 0 || V <= 0) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy plan missing");
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = mps_matrix(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *wm = mps_matrix(w_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *lm = mps_matrix(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float));
        gd_status status = encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, c0 == 0 ? 1 : 0};
        [*enc setComputePipelineState:fwd_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:scratch offset:L.m_off atIndex:2];
        [*enc setBuffer:scratch offset:L.l_off atIndex:3];
        [*enc setBuffer:scratch offset:L.target_logit_off atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        dispatch_1d(*enc, fwd_pso, (NSUInteger)rows);
    }

    {
        gd_metal_lmce_params p = {rows, D, V, 0, chunk_max, 0};
        gd_metal_ce_params rp = {1, 1, V, rows};
        [*enc setComputePipelineState:loss_pso];
        [*enc setBuffer:scratch offset:L.m_off atIndex:0];
        [*enc setBuffer:scratch offset:L.l_off atIndex:1];
        [*enc setBuffer:scratch offset:L.target_logit_off atIndex:2];
        [*enc setBuffer:scratch offset:L.losses_off atIndex:3];
        [*enc setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(*enc, loss_pso, (NSUInteger)rows);

        [*enc setComputePipelineState:reduce_pso];
        [*enc setBuffer:scratch offset:L.losses_off atIndex:0];
        [*enc setBuffer:out_b offset:0 atIndex:1];
        [*enc setBytes:&rp length:sizeof(rp) atIndex:2];
        [*enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return GD_OK;
}

static gd_status encode_lm_cross_entropy_bwd(_gd_backend *self,
                                             id<MTLCommandBuffer> cmd,
                                             id<MTLComputeCommandEncoder> *enc,
                                             id<MTLBuffer> scratch,
                                             _gd_executable *exe,
                                             const _gd_node *node)
{
    GDMetalState *st = state_of(self);
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[2]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(desc_numel(x_desc) / D) : 0;
    int chunk_max = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    lmce_scratch_layout L = lmce_scratch_layout_for(rows, chunk_max);
    id<MTLBuffer> x_b = value_buffer(exe, node->inputs[0]);
    id<MTLBuffer> w_b = value_buffer(exe, node->inputs[1]);
    id<MTLBuffer> t_b = value_buffer(exe, node->inputs[2]);
    id<MTLBuffer> go_b = value_buffer(exe, node->inputs[3]);
    id<MTLBuffer> dx_b = value_buffer(exe, node->outputs[0]);
    id<MTLBuffer> dw_b = value_buffer(exe, node->outputs[1]);
    id<MTLComputePipelineState> fwd_pso = pipeline_named(st, "gd_lmce_fwd_chunk");
    id<MTLComputePipelineState> dlogits_pso = pipeline_named(st, "gd_lmce_dlogits_chunk");

    if (!st.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs GD_METAL_MPS=1");
    }
    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs I32 targets");
    }
    if (scratch == nil || fwd_pso == nil || dlogits_pso == nil || rows <= 0 ||
        D <= 0 || V <= 0) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy_bwd plan missing");
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = mps_matrix(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *wm = mps_matrix(w_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *lm = mps_matrix(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float));
        gd_status status = encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, c0 == 0 ? 1 : 0};
        [*enc setComputePipelineState:fwd_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:scratch offset:L.m_off atIndex:2];
        [*enc setBuffer:scratch offset:L.l_off atIndex:3];
        [*enc setBuffer:scratch offset:L.target_logit_off atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        dispatch_1d(*enc, fwd_pso, (NSUInteger)rows);
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = mps_matrix(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *wm = mps_matrix(w_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *lm = mps_matrix(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float));
        gd_status status = encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, 0};
        [*enc setComputePipelineState:dlogits_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:go_b offset:0 atIndex:2];
        [*enc setBuffer:scratch offset:L.m_off atIndex:3];
        [*enc setBuffer:scratch offset:L.l_off atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        dispatch_1d(*enc, dlogits_pso, (NSUInteger)rows * (NSUInteger)csz);

        MPSMatrix *dxm = mps_matrix(dx_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                    (NSUInteger)D * sizeof(float));
        MPSMatrix *dwm = mps_matrix(dw_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                    (NSUInteger)csz, (NSUInteger)D,
                                    (NSUInteger)D * sizeof(float));
        status = encode_mps_mm(cmd, enc, st.device, lm, wm, dxm,
                               false, false, (NSUInteger)rows, (NSUInteger)D,
                               (NSUInteger)csz, c0 == 0 ? 0.0 : 1.0);
        if (status != GD_OK) {
            return status;
        }
        status = encode_mps_mm(cmd, enc, st.device, lm, xm, dwm,
                               true, false, (NSUInteger)csz, (NSUInteger)D,
                               (NSUInteger)rows, 0.0);
        if (status != GD_OK) {
            return status;
        }
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
                                      id<MTLComputePipelineState> scatter_pso,
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
    if (scatter_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal embedding_bwd scatter pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    {
        int total = p.vocab * p.dim;
        if (total > 0) {
            dispatch_1d(enc, pso, (NSUInteger)total);
        }
    }
    [enc setComputePipelineState:scatter_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        int total = p.n * p.dim;
        if (total > 0) {
            dispatch_1d(enc, scatter_pso, (NSUInteger)total);
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
    p->n_splits = 1;
    p->split_len = p->Tk;
}

static void encode_sdpa(_gd_backend *self,
                        id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        id<MTLComputePipelineState> splitk_pso,
                        id<MTLComputePipelineState> combine_pso,
                        id<MTLBuffer> scratch,
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
    int n_qb = 0;

    fill_sdpa_params(&p, q, k, bias, node);
    n_qb = (p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;

    /* Split-K (flash-decoding) path for long context: partition the key range
     * across threadgroups to shorten the heaviest query block's critical path,
     * then merge partials. Eligible only when the scratch/pipelines were
     * allocated at compile (Dh <= DHT and >1 split). */
    if (p.Dh <= GD_METAL_SDPA_DHT && scratch != nil &&
        splitk_pso != nil && combine_pso != nil) {
        p.n_splits = sdpa_num_splits(p.Tk);
        p.split_len = sdpa_split_len(p.Tk, p.n_splits);
        /* Pass 1: per-split partial online-softmax into scratch. */
        [enc setComputePipelineState:splitk_pso];
        [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
        [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:3];
        [enc setBuffer:scratch offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        NSUInteger sgroups = (NSUInteger)(p.B * p.Hq * n_qb * p.n_splits);
        if (sgroups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(sgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
        }
        /* Pass 2: combine the splits into the output. */
        [enc setComputePipelineState:combine_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        int grid = p.B * p.Hq * p.Tq;
        if (grid > 0) {
            dispatch_1d(enc, combine_pso, (NSUInteger)grid);
        }
        return;
    }

    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:3];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];

    if (p.Dh <= GD_METAL_SDPA_DHT) {
        /* Tiled: one threadgroup per (b, hq, query-block) of GD_METAL_SDPA_BQ. */
        NSUInteger groups = (NSUInteger)(p.B * p.Hq * n_qb);
        [enc setComputePipelineState:pso];
        if (groups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
        }
    } else {
        /* Fallback to the per-(b,hq,i) reference kernel for large head_dim. */
        id<MTLComputePipelineState> ref = pipeline_named(state_of(self), "gd_sdpa");
        int grid = p.B * p.Hq * p.Tq;
        [enc setComputePipelineState:ref];
        if (grid > 0) {
            dispatch_1d(enc, ref, (NSUInteger)grid);
        }
    }
}

/* SDPA backward: dq kernel over query rows, then dk/dv kernel over kv rows. */
static gd_status encode_sdpa_bwd(_gd_backend *self,
                                 id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> dq_pso,
                                 id<MTLComputePipelineState> dkv_pso,
                                 id<MTLComputePipelineState> stats_pso,
                                 id<MTLBuffer> stats_buf,
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
    int q_groups = 0;
    int kv_groups = 0;

    if (dkv_pso == nil || stats_pso == nil || stats_buf == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd pipeline/scratch missing");
    }
    fill_sdpa_params(&p, q, k, bias, node);
    /* The tiled backward keeps per-thread q/k/v/dO/acc rows in registers, capped
     * at GD_METAL_SDPA_DHT. Fail loud for larger head_dim (CPU_REF is the
     * correctness fallback) rather than overrun the register arrays. */
    if (p.Dh > GD_METAL_SDPA_DHT) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd supports head_dim <= 64");
    }

    /* Long-context split-K backward: like the forward, each pass is critical-path
     * bound on the heaviest block, so split the scanned dimension across
     * threadgroups into per-split partials, then reduce. stats/dq split the key
     * range (per query block); dkv splits the query range (per key block). */
    {
        int nsplit = sdpa_num_splits(p.Tk > p.Tq ? p.Tk : p.Tq);
        if (nsplit > 1) {
            sdpa_bwd_layout L = sdpa_bwd_scratch_layout(p.B, p.Hq, p.Hkv, p.Tq, p.Tk,
                                                        p.Dh, nsplit);
            NSUInteger stat_b = (NSUInteger)L.stats_off * sizeof(float);
            NSUInteger spart_b = (NSUInteger)L.stats_part_off * sizeof(float);
            NSUInteger dqpart_b = (NSUInteger)L.dq_part_off * sizeof(float);
            NSUInteger dkvpart_b = (NSUInteger)L.dkv_part_off * sizeof(float);
            int n_qb = (p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;
            int n_kb = (p.Tk + GD_METAL_SDPA_DKV_KEYS - 1) / GD_METAL_SDPA_DKV_KEYS;
            NSUInteger qsplit_groups = (NSUInteger)(p.B * p.Hq * n_qb * nsplit);
            NSUInteger ksplit_groups = (NSUInteger)(p.B * p.Hkv * n_kb * nsplit);
            NSUInteger q_rows = (NSUInteger)(p.B * p.Hq * p.Tq);
            NSUInteger kv_rows = (NSUInteger)(p.B * p.Hkv * p.Tk);
            GDMetalState *st = state_of(self);
            id<MTLComputePipelineState> sdq_pso = pipeline_named(st, "gd_sdpa_bwd_stats_dq_split");
            id<MTLComputePipelineState> sdqc_pso = pipeline_named(st, "gd_sdpa_bwd_stats_dq_combine");
            id<MTLComputePipelineState> dks_pso = pipeline_named(st, "gd_sdpa_bwd_dkv_split");
            id<MTLComputePipelineState> dkr_pso = pipeline_named(st, "gd_sdpa_bwd_dkv_reduce");
            id<MTLBuffer> go_b = value_buffer(exe, node->inputs[0]);
            id<MTLBuffer> q_b = value_buffer(exe, node->inputs[1]);
            id<MTLBuffer> k_b = value_buffer(exe, node->inputs[2]);
            id<MTLBuffer> v_b = value_buffer(exe, node->inputs[3]);
            id<MTLBuffer> bias_b = value_buffer(exe, bias_input);
            MTLSize tg = MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1);

            if (sdq_pso == nil || sdqc_pso == nil || dks_pso == nil || dkr_pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd split-K pipeline missing");
            }
            p.n_splits = nsplit;

            /* Fused stats+dq: per-split (m,l,raw,acc,ksum) -> final stats+dq.
             * This removes the old second key scan in the dq split pass. */
            [enc setComputePipelineState:sdq_pso];
            [enc setBuffer:go_b offset:0 atIndex:0];
            [enc setBuffer:q_b offset:0 atIndex:1];
            [enc setBuffer:k_b offset:0 atIndex:2];
            [enc setBuffer:v_b offset:0 atIndex:3];
            [enc setBuffer:bias_b offset:0 atIndex:4];
            [enc setBuffer:stats_buf offset:spart_b atIndex:5];
            [enc setBuffer:stats_buf offset:dqpart_b atIndex:6];
            [enc setBytes:&p length:sizeof(p) atIndex:7];
            if (qsplit_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(qsplit_groups, 1, 1) threadsPerThreadgroup:tg];
            }
            [enc setComputePipelineState:sdqc_pso];
            [enc setBuffer:stats_buf offset:spart_b atIndex:0];
            [enc setBuffer:stats_buf offset:dqpart_b atIndex:1];
            [enc setBuffer:stats_buf offset:stat_b atIndex:2];
            [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            if (q_rows > 0) {
                dispatch_1d(enc, sdqc_pso, q_rows * (NSUInteger)p.Dh);
            }

            /* dk/dv: per-split partial sums -> reduce. */
            [enc setComputePipelineState:dks_pso];
            [enc setBuffer:go_b offset:0 atIndex:0];
            [enc setBuffer:q_b offset:0 atIndex:1];
            [enc setBuffer:k_b offset:0 atIndex:2];
            [enc setBuffer:v_b offset:0 atIndex:3];
            [enc setBuffer:bias_b offset:0 atIndex:4];
            [enc setBuffer:stats_buf offset:dkvpart_b atIndex:5];
            [enc setBytes:&p length:sizeof(p) atIndex:6];
            [enc setBuffer:stats_buf offset:stat_b atIndex:7];
            if (ksplit_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(ksplit_groups, 1, 1) threadsPerThreadgroup:tg];
            }
            [enc setComputePipelineState:dkr_pso];
            [enc setBuffer:stats_buf offset:dkvpart_b atIndex:0];
            [enc setBuffer:value_buffer(exe, node->outputs[1]) offset:0 atIndex:1];
            [enc setBuffer:value_buffer(exe, node->outputs[2]) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            if (kv_rows > 0) {
                dispatch_1d(enc, dkr_pso, kv_rows * (NSUInteger)p.Dh);
            }
            return GD_OK;
        }
    }

    /* Threadgroups: query blocks for stats/dq, key blocks for dk/dv; both use
     * GD_METAL_SDPA_BQ threads per group. */
    q_groups = p.B * p.Hq * ((p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ);
    kv_groups = p.B * p.Hkv * ((p.Tk + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ);

    /* Pass 1: per-query softmax stats (m, l, D) into scratch. */
    [enc setComputePipelineState:stats_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:stats_buf offset:0 atIndex:5];                          /* stats */
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    /* Pass 2: dq (one pass over keys using stats). */
    [enc setComputePipelineState:dq_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:5]; /* dq */
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc setBuffer:stats_buf offset:0 atIndex:7];                          /* stats */
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    /* Pass 3: dk/dv (one pass over queries per kv slot using stats). */
    [enc setComputePipelineState:dkv_pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:value_buffer(exe, node->outputs[1]) offset:0 atIndex:5]; /* dk */
    [enc setBuffer:value_buffer(exe, node->outputs[2]) offset:0 atIndex:6]; /* dv */
    [enc setBytes:&p length:sizeof(p) atIndex:7];
    [enc setBuffer:stats_buf offset:0 atIndex:8];                          /* stats */
    if (kv_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
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

static gd_status encode_rms_norm_wbwd(id<MTLComputeCommandEncoder> enc,
                                       id<MTLComputePipelineState> pso,
                                       id<MTLComputePipelineState> reduce_pso,
                                       id<MTLBuffer> scratch,
                                       _gd_executable *exe,
                                       const _gd_node *node)
{
    /* dweight: inputs x(0), go(1); output dweight [last]. dims from x. */
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)x_desc->sizes[x_desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(desc_numel(x_desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    if (p.rows <= 0 || p.last <= 0) {
        return GD_OK;
    }
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal rms_norm_wbwd scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger row_blocks = ((NSUInteger)p.rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        NSUInteger channel_blocks = ((NSUInteger)p.last + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        [enc dispatchThreadgroups:MTLSizeMake(row_blocks * channel_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_TG, 1, 1)];
    }
    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    dispatch_1d(enc, reduce_pso, (NSUInteger)p.last);
    return GD_OK;
}

static gd_status encode_node(_gd_backend *self,
                             id<MTLComputeCommandEncoder> enc,
                             _gd_executable *exe,
                             const _gd_node *node,
                             id<MTLComputePipelineState> pso,
                             id<MTLComputePipelineState> pso2,
                             id<MTLComputePipelineState> pso3,
                             id<MTLBuffer> scratch)
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
        return encode_cross_entropy(enc, pso, pso2, scratch, exe, node);
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
        return encode_embedding_bwd(enc, pso, pso2, exe, node);
    case _GD_OP_ROPE:
        return encode_rope(enc, pso, exe, node, 1.0F);
    case _GD_OP_ROPE_BWD:
        return encode_rope(enc, pso, exe, node, -1.0F);
    case _GD_OP_SDPA:
        encode_sdpa(self, enc, pso, pso2, pso3, scratch, exe, node);
        return GD_OK;
    case _GD_OP_SDPA_BWD:
        return encode_sdpa_bwd(self, enc, pso, pso2, pso3, scratch, exe, node);
    case _GD_OP_RMS_NORM_BWD:
        encode_rms_norm_bwd(enc, pso, exe, node);
        return GD_OK;
    case _GD_OP_RMS_NORM_WBWD:
        return encode_rms_norm_wbwd(enc, pso, pso2, scratch, exe, node);
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
            id<MTLComputePipelineState> pso;
            id<MTLComputePipelineState> pso2;
            if (exe->node_absorbed[i]) {
                continue; /* encoded by its fusion group head */
            }
            pso = (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
            pso2 = (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
            id<MTLComputePipelineState> pso3 = (__bridge id<MTLComputePipelineState>)exe->node_pso3[i];
            id<MTLBuffer> scratch = nil;
            if (exe->node_scratch_bytes[i] > 0U) {
                if (exe->scratch_arena == NULL ||
                    exe->node_scratch_bytes[i] > exe->scratch_arena_bytes) {
                    return _gd_error(GD_ERR_BACKEND, "metal scratch arena too small");
                }
                scratch = (__bridge id<MTLBuffer>)_gd_storage_handle(exe->scratch_arena);
            }
            uint64_t op_start = _gd_profile_now_ns();
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (exe->node_fused_src[i] >= 0) {
                    encode_fused_silu_mul(enc, pso2, exe, node,
                                          &exe->graph->nodes[exe->node_fused_src[i]]);
                } else if (node->op == _GD_OP_LM_CROSS_ENTROPY) {
                    status = encode_lm_cross_entropy(self, cmd, &enc, scratch, exe, node);
                } else if (node->op == _GD_OP_LM_CROSS_ENTROPY_BWD) {
                    status = encode_lm_cross_entropy_bwd(self, cmd, &enc, scratch, exe, node);
                } else if (exe->node_mps[i] != NULL &&
                           (node->op == _GD_OP_LINEAR || node->op == _GD_OP_MATMUL)) {
                    status = encode_mps_gemm(cmd, &enc,
                                             (__bridge GDMPSGemmPlan *)exe->node_mps[i]);
                } else {
                    status = encode_node(self, enc, exe, node, pso, pso2, pso3, scratch);
                }
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
                id<MTLComputePipelineState> pso;
                if (exe->node_absorbed[i]) {
                    continue; /* encoded by its fusion group head */
                }
                pso = (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
                id<MTLComputePipelineState> pso2 =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
                id<MTLComputePipelineState> pso3 =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso3[i];
                id<MTLBuffer> scratch = nil;
                if (exe->node_scratch_bytes[i] > 0U) {
                    if (exe->scratch_arena == NULL ||
                        exe->node_scratch_bytes[i] > exe->scratch_arena_bytes) {
                        [enc endEncoding];
                        return _gd_error(GD_ERR_BACKEND, "metal scratch arena too small");
                    }
                    scratch = (__bridge id<MTLBuffer>)_gd_storage_handle(exe->scratch_arena);
                }
                if (exe->node_fused_src[i] >= 0) {
                    encode_fused_silu_mul(enc, pso2, exe, &exe->graph->nodes[i],
                                          &exe->graph->nodes[exe->node_fused_src[i]]);
                } else if (exe->graph->nodes[i].op == _GD_OP_LM_CROSS_ENTROPY) {
                    status = encode_lm_cross_entropy(self, cmd, &enc, scratch, exe,
                                                     &exe->graph->nodes[i]);
                } else if (exe->graph->nodes[i].op == _GD_OP_LM_CROSS_ENTROPY_BWD) {
                    status = encode_lm_cross_entropy_bwd(self, cmd, &enc, scratch, exe,
                                                         &exe->graph->nodes[i]);
                } else if (exe->node_mps[i] != NULL &&
                           (exe->graph->nodes[i].op == _GD_OP_LINEAR ||
                            exe->graph->nodes[i].op == _GD_OP_MATMUL)) {
                    status = encode_mps_gemm(cmd, &enc,
                                             (__bridge GDMPSGemmPlan *)exe->node_mps[i]);
                } else {
                    status = encode_node(self, enc, exe, &exe->graph->nodes[i], pso, pso2, pso3, scratch);
                }
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
    GDMetalState *st = state_of(self);
    if (node->op == _GD_OP_LM_CROSS_ENTROPY ||
        node->op == _GD_OP_LM_CROSS_ENTROPY_BWD) {
        /* Metal LMCE v1 is chunked around MPS GEMMs, so keep it behind the
         * same opt-in gate as generic MPS GEMM. */
        return st.useMPS;
    }
    return pipeline_for(st, node->op) != nil;
}

/* ---- Init / shutdown / registration -------------------------------------- */

static bool env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return false;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

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
        st.useMPS = env_flag_enabled("GD_METAL_MPS") ? YES : NO;

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
