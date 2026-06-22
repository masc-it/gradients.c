#include "../../backends/metal/metal_backend_internal.h"
#include "metal_sdpa_varlen_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLComputePipelineState> gd_sdpa_varlen_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_compact_qblocks_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_compact_qblocks_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_compact_qkblocks_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_compact_qkblocks_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_prefix_window_dh64_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_prefix_window_dh64_f16_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_prefix_window_dh64_f16_stats_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_prefix_window_dh64_f16_stats_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_stats_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_stats_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_dkv_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_dkv_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_stats_dq_dh64_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_stats_dq_dh64_f16_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_saved_stats_dq_dh64_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_saved_stats_dq_dh64_f16_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_dkv_dh64_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_dkv_dh64_f16_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_dkv_split_dh64_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_dkv_split_dh64_f16_pso;
}

static id<MTLComputePipelineState> gd_sdpa_varlen_bwd_dkv_reduce_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_varlen_bwd_dkv_reduce_f16_pso;
}

static id<MTLBuffer> gd_sdpa_varlen_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static id<MTLDevice> gd_sdpa_varlen_device(gd_backend *backend)
{
    return (__bridge id<MTLDevice>)backend->device;
}

static int gd_sdpa_varlen_env_int(const char *name, int fallback, int min_value, int max_value)
{
    const char *value = getenv(name);
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    parsed = strtol(value, &end, 10);
    if (end == value || parsed < (long)min_value) {
        return fallback;
    }
    if (parsed > (long)max_value) {
        return max_value;
    }
    return (int)parsed;
}

static uint32_t gd_sdpa_varlen_num_splits_for_extent(uint32_t extent)
{
    const int split_min = gd_sdpa_varlen_env_int("GD_METAL_SDPA_SPLIT_MIN",
                                                 (int)GD_METAL_SDPA_SPLIT_MIN,
                                                 (int)GD_METAL_SDPA_BK,
                                                 1 << 20);
    const int split_max = gd_sdpa_varlen_env_int("GD_METAL_SDPA_SPLIT_MAX",
                                                 (int)GD_METAL_SDPA_SPLIT_MAX,
                                                 1,
                                                 1024);
    const uint32_t split_min_u = (uint32_t)split_min;
    const uint32_t split_max_u = (uint32_t)split_max;
    uint64_t splits64;
    if (extent == 0U) {
        return 1U;
    }
    splits64 = ((uint64_t)extent + (uint64_t)split_min_u - 1ULL) / (uint64_t)split_min_u;
    if (splits64 > (uint64_t)split_max_u) {
        return split_max_u;
    }
    return splits64 < 1ULL ? 1U : (uint32_t)splits64;
}

static uint32_t gd_sdpa_varlen_dkv_split_extent(const gd_backend_sdpa_varlen_args *args)
{
    uint32_t local_extent;
    if (args == NULL || args->max_seqlen == 0U) {
        return 1U;
    }
    if (args->causal == 0U || args->sliding_window == 0U || args->prefix_len != 0U) {
        return args->max_seqlen;
    }

    /* Prefix-free causal local attention only lets a key block receive
     * gradients from the union of queries in [k0, k0 + key_block + window).
     * Split dK/dV over that local extent rather than the full sequence; for
     * GPT-style window=256 this keeps n_splits at 1 even when T >> 512 and
     * avoids allocating a large mostly-empty partial dK/dV buffer.  Prefix-LM
     * keeps the full-sequence extent because prefix keys can be seen by every
     * query. */
    if (args->sliding_window > UINT32_MAX - (GD_METAL_SDPA_DKV_WIDE_KEYS - 1U)) {
        local_extent = UINT32_MAX;
    } else {
        local_extent = args->sliding_window + GD_METAL_SDPA_DKV_WIDE_KEYS - 1U;
    }
    return local_extent < args->max_seqlen ? local_extent : args->max_seqlen;
}

static uint32_t gd_sdpa_varlen_num_dkv_splits(const gd_backend_sdpa_varlen_args *args)
{
    return gd_sdpa_varlen_num_splits_for_extent(gd_sdpa_varlen_dkv_split_extent(args));
}

static bool gd_sdpa_varlen_env_enabled(const char *name, bool fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 && strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

static bool gd_sdpa_varlen_fast_enabled(void)
{
    return gd_sdpa_varlen_env_enabled("GD_METAL_SDPA_VARLEN_FAST", true);
}

static bool gd_sdpa_varlen_compact_enabled(void)
{
    return gd_sdpa_varlen_env_enabled("GD_METAL_SDPA_VARLEN_COMPACT", true);
}

static bool gd_sdpa_varlen_size_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool gd_sdpa_varlen_use_dh64_prefix_window(const gd_backend_tensor_view *q,
                                                   const gd_backend_sdpa_varlen_args *args)
{
    /* The lane8 DH=64 causal-window kernels handle both prefix-LM masks
     * (prefix_len > 0) and plain causal local attention (prefix_len == 0).
     * GPT training uses the latter, so do not require a positive prefix. */
    return gd_sdpa_varlen_fast_enabled() && q != NULL && args != NULL &&
           q->dtype == (uint32_t)GD_DTYPE_F16 && q->shape[2] == 64 &&
           args->causal != 0U && args->sliding_window > 0U;
}

static bool gd_sdpa_varlen_byte_range_valid(const gd_backend_buffer *buffer,
                                            size_t offset,
                                            size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_sdpa_varlen_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        *out_size = 2U;
        return true;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static bool gd_sdpa_varlen_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_sdpa_varlen_dtype_size(view->dtype, &elem_size) ||
        view->count > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = view->count * elem_size;
    return gd_sdpa_varlen_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_sdpa_varlen_contiguous_valid(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride) {
            return false;
        }
        if (view->shape[dim] != 0 && stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_sdpa_varlen_shapes_valid(const gd_backend_tensor_view *q,
                                        const gd_backend_tensor_view *k,
                                        const gd_backend_tensor_view *v,
                                        const gd_backend_tensor_view *cu,
                                        const gd_backend_tensor_view *out)
{
    if (!gd_sdpa_varlen_view_range_valid(q) || !gd_sdpa_varlen_view_range_valid(k) ||
        !gd_sdpa_varlen_view_range_valid(v) || !gd_sdpa_varlen_view_range_valid(cu) ||
        !gd_sdpa_varlen_view_range_valid(out) || !gd_sdpa_varlen_contiguous_valid(q) ||
        !gd_sdpa_varlen_contiguous_valid(k) || !gd_sdpa_varlen_contiguous_valid(v) ||
        !gd_sdpa_varlen_contiguous_valid(cu) || !gd_sdpa_varlen_contiguous_valid(out)) {
        return false;
    }
    if ((q->dtype != (uint32_t)GD_DTYPE_F16 && q->dtype != (uint32_t)GD_DTYPE_F32) ||
        k->dtype != q->dtype || v->dtype != q->dtype || out->dtype != q->dtype ||
        cu->dtype != (uint32_t)GD_DTYPE_I32 || q->rank != 3U || k->rank != 3U ||
        v->rank != 3U || out->rank != 3U || cu->rank != 1U || cu->shape[0] < 2 ||
        q->shape[0] <= 0 || q->shape[1] <= 0 || q->shape[2] <= 0 || k->shape[1] <= 0 ||
        k->shape[0] != q->shape[0] || v->shape[0] != q->shape[0] ||
        k->shape[2] != q->shape[2] || v->shape[1] != k->shape[1] ||
        v->shape[2] != q->shape[2] || out->shape[0] != q->shape[0] ||
        out->shape[1] != q->shape[1] || out->shape[2] != q->shape[2] ||
        (q->shape[1] % k->shape[1]) != 0 || q->shape[2] > (int64_t)GD_METAL_SDPA_DHT ||
        q->shape[0] > (int64_t)UINT32_MAX || q->shape[1] > (int64_t)UINT32_MAX ||
        k->shape[1] > (int64_t)UINT32_MAX || cu->shape[0] - 1 > (int64_t)UINT32_MAX) {
        return false;
    }
    return true;
}

static bool gd_sdpa_varlen_stats_valid(const gd_backend_tensor_view *q,
                                       const gd_backend_tensor_view *stats)
{
    return gd_sdpa_varlen_view_range_valid(stats) && gd_sdpa_varlen_contiguous_valid(stats) &&
           stats->dtype == (uint32_t)GD_DTYPE_F32 && stats->rank == 3U &&
           stats->shape[0] == q->shape[0] && stats->shape[1] == q->shape[1] &&
           stats->shape[2] == 3;
}

static bool gd_sdpa_varlen_bwd_shapes_valid(const gd_backend_tensor_view *grad_out,
                                            const gd_backend_tensor_view *q,
                                            const gd_backend_tensor_view *k,
                                            const gd_backend_tensor_view *v,
                                            const gd_backend_tensor_view *cu,
                                            const gd_backend_tensor_view *grad_q,
                                            const gd_backend_tensor_view *grad_k,
                                            const gd_backend_tensor_view *grad_v,
                                            const gd_backend_tensor_view *stats)
{
    return gd_sdpa_varlen_shapes_valid(q, k, v, cu, grad_q) &&
           gd_sdpa_varlen_view_range_valid(grad_out) &&
           gd_sdpa_varlen_contiguous_valid(grad_out) &&
           gd_sdpa_varlen_view_range_valid(grad_k) &&
           gd_sdpa_varlen_contiguous_valid(grad_k) &&
           gd_sdpa_varlen_view_range_valid(grad_v) &&
           gd_sdpa_varlen_contiguous_valid(grad_v) &&
           grad_out->dtype == q->dtype && grad_out->rank == 3U &&
           grad_out->shape[0] == q->shape[0] && grad_out->shape[1] == q->shape[1] &&
           grad_out->shape[2] == q->shape[2] && grad_k->dtype == k->dtype &&
           grad_v->dtype == v->dtype && grad_k->rank == 3U && grad_v->rank == 3U &&
           grad_k->shape[0] == k->shape[0] && grad_k->shape[1] == k->shape[1] &&
           grad_k->shape[2] == k->shape[2] && grad_v->shape[0] == v->shape[0] &&
           grad_v->shape[1] == v->shape[1] && grad_v->shape[2] == v->shape[2] &&
           gd_sdpa_varlen_stats_valid(q, stats);
}

static bool gd_sdpa_varlen_args_valid(const gd_backend_sdpa_varlen_args *args,
                                      const gd_backend_tensor_view *q,
                                      const gd_backend_tensor_view *cu)
{
    (void)q;
    (void)cu;
    return args != NULL && args->scale == args->scale && args->scale > 0.0f &&
           args->max_seqlen > 0U && args->prefix_len <= args->max_seqlen &&
           (args->prefix_len == 0U || args->causal != 0U);
}

static void gd_sdpa_varlen_fill_metal_args(gd_metal_sdpa_varlen_args *p,
                                           const gd_backend_tensor_view *q,
                                           const gd_backend_tensor_view *k,
                                           const gd_backend_tensor_view *cu,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_q,
                                           const gd_backend_tensor_view *grad_k,
                                           const gd_backend_tensor_view *grad_v,
                                           const gd_backend_tensor_view *stats,
                                           const gd_backend_sdpa_varlen_args *args)
{
    memset(p, 0, sizeof(*p));
    p->q_offset = (uint64_t)q->offset;
    p->k_offset = (uint64_t)k->offset;
    p->v_offset = 0U;
    p->cu_offset = (uint64_t)cu->offset;
    p->out_offset = out != NULL ? (uint64_t)out->offset : 0U;
    p->grad_out_offset = grad_out != NULL ? (uint64_t)grad_out->offset : 0U;
    p->grad_q_offset = grad_q != NULL ? (uint64_t)grad_q->offset : 0U;
    p->grad_k_offset = grad_k != NULL ? (uint64_t)grad_k->offset : 0U;
    p->grad_v_offset = grad_v != NULL ? (uint64_t)grad_v->offset : 0U;
    p->stats_offset = stats != NULL ? (uint64_t)stats->offset : 0U;
    p->total_tokens = (uint64_t)q->shape[0];
    p->batch = (uint32_t)(cu->shape[0] - 1);
    p->hq = (uint32_t)q->shape[1];
    p->hkv = (uint32_t)k->shape[1];
    p->dh = (uint32_t)q->shape[2];
    p->max_seqlen = args->max_seqlen;
    p->n_qb_max = (args->max_seqlen + GD_METAL_SDPA_BQ - 1U) / GD_METAL_SDPA_BQ;
    p->causal = args->causal != 0U ? 1U : 0U;
    p->window = args->sliding_window;
    p->prefix_len = args->prefix_len;
    p->dtype = q->dtype;
    p->scale = args->scale;
    p->n_splits = 1U;
    p->compact_blocks = 0U;
    p->reserved0 = 0U;
}

static uint32_t gd_sdpa_varlen_kblock_count(uint32_t max_seqlen)
{
    return (max_seqlen + GD_METAL_SDPA_DKV_WIDE_KEYS - 1U) /
           GD_METAL_SDPA_DKV_WIDE_KEYS;
}

static bool gd_sdpa_varlen_worklist_nbytes(uint32_t batch, uint32_t blocks_per_segment, size_t *out_bytes)
{
    size_t entries;
    if (!gd_sdpa_varlen_size_mul((size_t)batch, (size_t)blocks_per_segment, &entries) ||
        !gd_sdpa_varlen_size_mul(entries, 2U * sizeof(uint32_t), out_bytes)) {
        return false;
    }
    return *out_bytes > 0U;
}

static id<MTLBuffer> gd_sdpa_varlen_new_private_buffer(gd_backend *backend, size_t nbytes)
{
    id<MTLDevice> device = gd_sdpa_varlen_device(backend);
    if (device == nil || nbytes == 0U) {
        return nil;
    }
    return [device newBufferWithLength:nbytes options:MTLResourceStorageModePrivate];
}

static gd_status gd_sdpa_varlen_make_compact_q_buffers(gd_backend *backend,
                                                       const gd_metal_sdpa_varlen_args *p,
                                                       id<MTLBuffer> *q_blocks,
                                                       id<MTLBuffer> *counts,
                                                       id<MTLBuffer> *dispatch_args)
{
    size_t q_bytes;
    if (backend == NULL || p == NULL || q_blocks == NULL || counts == NULL || dispatch_args == NULL ||
        !gd_sdpa_varlen_worklist_nbytes(p->batch, p->n_qb_max, &q_bytes)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *q_blocks = gd_sdpa_varlen_new_private_buffer(backend, q_bytes);
    *counts = gd_sdpa_varlen_new_private_buffer(backend, GD_METAL_SDPA_COMPACT_COUNT_BYTES);
    *dispatch_args = gd_sdpa_varlen_new_private_buffer(backend, GD_METAL_SDPA_COMPACT_DISPATCH_BYTES);
    return (*q_blocks != nil && *counts != nil && *dispatch_args != nil) ? GD_OK : GD_ERR_OUT_OF_MEMORY;
}

static gd_status gd_sdpa_varlen_make_compact_qk_buffers(gd_backend *backend,
                                                        const gd_metal_sdpa_varlen_args *p,
                                                        id<MTLBuffer> *q_blocks,
                                                        id<MTLBuffer> *k_blocks,
                                                        id<MTLBuffer> *counts,
                                                        id<MTLBuffer> *dispatch_args)
{
    size_t q_bytes;
    size_t k_bytes;
    if (backend == NULL || p == NULL || q_blocks == NULL || k_blocks == NULL || counts == NULL ||
        dispatch_args == NULL || !gd_sdpa_varlen_worklist_nbytes(p->batch, p->n_qb_max, &q_bytes) ||
        !gd_sdpa_varlen_worklist_nbytes(p->batch, gd_sdpa_varlen_kblock_count(p->max_seqlen), &k_bytes)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *q_blocks = gd_sdpa_varlen_new_private_buffer(backend, q_bytes);
    *k_blocks = gd_sdpa_varlen_new_private_buffer(backend, k_bytes);
    *counts = gd_sdpa_varlen_new_private_buffer(backend, GD_METAL_SDPA_COMPACT_COUNT_BYTES);
    *dispatch_args = gd_sdpa_varlen_new_private_buffer(backend, GD_METAL_SDPA_COMPACT_DISPATCH_BYTES);
    return (*q_blocks != nil && *k_blocks != nil && *counts != nil && *dispatch_args != nil) ?
               GD_OK :
               GD_ERR_OUT_OF_MEMORY;
}

static gd_status gd_sdpa_varlen_encode_compact_q_setup(id<MTLComputeCommandEncoder> encoder,
                                                       gd_backend *backend,
                                                       const gd_backend_tensor_view *cu_seqlens,
                                                       id<MTLBuffer> q_blocks,
                                                       id<MTLBuffer> counts,
                                                       id<MTLBuffer> dispatch_args,
                                                       const gd_metal_sdpa_varlen_args *p)
{
    id<MTLComputePipelineState> setup_pso;
    if (encoder == nil || backend == NULL || cu_seqlens == NULL || q_blocks == nil || counts == nil ||
        dispatch_args == nil || p == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    setup_pso = gd_sdpa_varlen_compact_qblocks_pso(backend);
    if (setup_pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    [encoder setComputePipelineState:setup_pso];
    [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:0U];
    [encoder setBuffer:q_blocks offset:0U atIndex:1U];
    [encoder setBuffer:counts offset:0U atIndex:2U];
    [encoder setBuffer:dispatch_args offset:0U atIndex:3U];
    [encoder setBytes:p length:sizeof(*p) atIndex:4U];
    [encoder dispatchThreadgroups:MTLSizeMake(1U, 1U, 1U)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_COMPACT_SETUP_THREADS, 1U, 1U)];
    return GD_OK;
}

static gd_status gd_sdpa_varlen_encode_compact_qk_setup(id<MTLComputeCommandEncoder> encoder,
                                                        gd_backend *backend,
                                                        const gd_backend_tensor_view *cu_seqlens,
                                                        id<MTLBuffer> q_blocks,
                                                        id<MTLBuffer> k_blocks,
                                                        id<MTLBuffer> counts,
                                                        id<MTLBuffer> dispatch_args,
                                                        const gd_metal_sdpa_varlen_args *p)
{
    id<MTLComputePipelineState> setup_pso;
    if (encoder == nil || backend == NULL || cu_seqlens == NULL || q_blocks == nil || k_blocks == nil ||
        counts == nil || dispatch_args == nil || p == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    setup_pso = gd_sdpa_varlen_compact_qkblocks_pso(backend);
    if (setup_pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    [encoder setComputePipelineState:setup_pso];
    [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:0U];
    [encoder setBuffer:q_blocks offset:0U atIndex:1U];
    [encoder setBuffer:k_blocks offset:0U atIndex:2U];
    [encoder setBuffer:counts offset:0U atIndex:3U];
    [encoder setBuffer:dispatch_args offset:0U atIndex:4U];
    [encoder setBytes:p length:sizeof(*p) atIndex:5U];
    [encoder dispatchThreadgroups:MTLSizeMake(1U, 1U, 1U)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_COMPACT_SETUP_THREADS, 1U, 1U)];
    return GD_OK;
}

gd_status gd_backend_sdpa_varlen(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k,
                                 const gd_backend_tensor_view *v,
                                 const gd_backend_tensor_view *cu_seqlens,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_tensor_view *stats,
                                 const gd_backend_sdpa_varlen_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> q_blocks = nil;
    id<MTLBuffer> compact_counts = nil;
    id<MTLBuffer> compact_dispatch = nil;
    gd_metal_sdpa_varlen_args p;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_sdpa_varlen_shapes_valid(q, k, v, cu_seqlens, out) ||
        (stats != NULL && !gd_sdpa_varlen_stats_valid(q, stats)) ||
        !gd_sdpa_varlen_args_valid(args, q, cu_seqlens)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (stats != NULL && !gd_sdpa_varlen_use_dh64_prefix_window(q, args)) {
        return GD_ERR_UNSUPPORTED;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_sdpa_varlen_fill_metal_args(&p, q, k, cu_seqlens, out, NULL, NULL, NULL, NULL, stats, args);
    p.v_offset = (uint64_t)v->offset;
    if (gd_sdpa_varlen_use_dh64_prefix_window(q, args)) {
        bool compact = gd_sdpa_varlen_compact_enabled();
        p.n_qb_max = (args->max_seqlen + GD_METAL_SDPA_CAUSAL_QROWS - 1U) /
                     GD_METAL_SDPA_CAUSAL_QROWS;
        p.compact_blocks = compact ? 1U : 0U;
        if (compact) {
            st = gd_sdpa_varlen_make_compact_q_buffers(backend, &p, &q_blocks, &compact_counts, &compact_dispatch);
            if (st != GD_OK) {
                [encoder endEncoding];
                return st;
            }
            st = gd_sdpa_varlen_encode_compact_q_setup(encoder,
                                                       backend,
                                                       cu_seqlens,
                                                       q_blocks,
                                                       compact_counts,
                                                       compact_dispatch,
                                                       &p);
            if (st != GD_OK) {
                [encoder endEncoding];
                return st;
            }
        }
        [encoder setComputePipelineState:stats != NULL ? gd_sdpa_varlen_prefix_window_dh64_f16_stats_pso(backend) :
                                          gd_sdpa_varlen_prefix_window_dh64_f16_pso(backend)];
        [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:q->offset atIndex:0U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:k->offset atIndex:1U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:v->offset atIndex:2U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:3U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(out->buffer) offset:out->offset atIndex:4U];
        [encoder setBytes:&p length:sizeof(p) atIndex:5U];
        if (stats != NULL) {
            [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:stats->offset atIndex:6U];
            [encoder setBuffer:compact ? q_blocks : gd_sdpa_varlen_buffer(cu_seqlens->buffer)
                         offset:compact ? 0U : cu_seqlens->offset
                        atIndex:7U];
        } else {
            [encoder setBuffer:compact ? q_blocks : gd_sdpa_varlen_buffer(cu_seqlens->buffer)
                         offset:compact ? 0U : cu_seqlens->offset
                        atIndex:6U];
        }
        threads = MTLSizeMake((NSUInteger)GD_METAL_SDPA_CAUSAL_THREADS, 1U, 1U);
        if (compact) {
            [encoder dispatchThreadgroupsWithIndirectBuffer:compact_dispatch
                                       indirectBufferOffset:GD_METAL_SDPA_COMPACT_Q_DISPATCH_OFFSET
                                      threadsPerThreadgroup:threads];
        } else {
            groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq * (NSUInteger)p.n_qb_max, 1U, 1U);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        }
    } else {
        [encoder setComputePipelineState:gd_sdpa_varlen_pso(backend)];
        [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(out->buffer) offset:0U atIndex:4U];
        [encoder setBytes:&p length:sizeof(p) atIndex:5U];
        groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq * (NSUInteger)p.n_qb_max, 1U, 1U);
        threads = MTLSizeMake((NSUInteger)GD_METAL_SDPA_BQ, 1U, 1U);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_sdpa_varlen_backward(gd_backend *backend,
                                          const gd_backend_tensor_view *grad_out,
                                          const gd_backend_tensor_view *q,
                                          const gd_backend_tensor_view *k,
                                          const gd_backend_tensor_view *v,
                                          const gd_backend_tensor_view *cu_seqlens,
                                          const gd_backend_tensor_view *grad_q,
                                          const gd_backend_tensor_view *grad_k,
                                          const gd_backend_tensor_view *grad_v,
                                          const gd_backend_tensor_view *stats,
                                          const gd_backend_sdpa_varlen_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> q_blocks = nil;
    id<MTLBuffer> k_blocks = nil;
    id<MTLBuffer> compact_counts = nil;
    id<MTLBuffer> compact_dispatch = nil;
    gd_metal_sdpa_varlen_args p;
    MTLSize q_groups;
    MTLSize kv_groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_sdpa_varlen_bwd_shapes_valid(grad_out,
                                                            q,
                                                            k,
                                                            v,
                                                            cu_seqlens,
                                                            grad_q,
                                                            grad_k,
                                                            grad_v,
                                                            stats) ||
        !gd_sdpa_varlen_args_valid(args, q, cu_seqlens)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_sdpa_varlen_fill_metal_args(&p,
                                   q,
                                   k,
                                   cu_seqlens,
                                   NULL,
                                   grad_out,
                                   grad_q,
                                   grad_k,
                                   grad_v,
                                   stats,
                                   args);
    p.v_offset = (uint64_t)v->offset;
    if (args->use_forward_stats != 0U && !gd_sdpa_varlen_use_dh64_prefix_window(q, args)) {
        [encoder endEncoding];
        return GD_ERR_UNSUPPORTED;
    }
    if (gd_sdpa_varlen_use_dh64_prefix_window(q, args)) {
        const uint32_t n_qb = (args->max_seqlen + GD_METAL_SDPA_CAUSAL_QROWS - 1U) /
                               GD_METAL_SDPA_CAUSAL_QROWS;
        const uint32_t n_kb = gd_sdpa_varlen_kblock_count(args->max_seqlen);
        const uint32_t n_splits = gd_sdpa_varlen_num_dkv_splits(args);
        const bool compact = gd_sdpa_varlen_compact_enabled();
        id<MTLBuffer> part_buffer = nil;
        p.n_qb_max = n_qb;
        p.n_splits = n_splits;
        p.compact_blocks = compact ? 1U : 0U;
        if (compact) {
            st = gd_sdpa_varlen_make_compact_qk_buffers(backend,
                                                        &p,
                                                        &q_blocks,
                                                        &k_blocks,
                                                        &compact_counts,
                                                        &compact_dispatch);
            if (st != GD_OK) {
                [encoder endEncoding];
                return st;
            }
            st = gd_sdpa_varlen_encode_compact_qk_setup(encoder,
                                                        backend,
                                                        cu_seqlens,
                                                        q_blocks,
                                                        k_blocks,
                                                        compact_counts,
                                                        compact_dispatch,
                                                        &p);
            if (st != GD_OK) {
                [encoder endEncoding];
                return st;
            }
        } else {
            q_groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq * (NSUInteger)n_qb, 1U, 1U);
            kv_groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hkv * (NSUInteger)n_kb, 1U, 1U);
        }
        if (n_splits > 1U) {
            size_t part_elems;
            size_t part_bytes;
            if (p.total_tokens > (uint64_t)(SIZE_MAX / (size_t)p.hkv) ||
                (size_t)p.total_tokens * (size_t)p.hkv > SIZE_MAX / (size_t)n_splits ||
                (size_t)p.total_tokens * (size_t)p.hkv * (size_t)n_splits > SIZE_MAX / 2U ||
                (size_t)p.total_tokens * (size_t)p.hkv * (size_t)n_splits * 2U >
                    SIZE_MAX / (size_t)GD_METAL_SDPA_DHT) {
                [encoder endEncoding];
                return GD_ERR_OUT_OF_MEMORY;
            }
            part_elems = (size_t)p.total_tokens * (size_t)p.hkv * (size_t)n_splits *
                         2U * (size_t)GD_METAL_SDPA_DHT;
            if (part_elems > SIZE_MAX / sizeof(float)) {
                [encoder endEncoding];
                return GD_ERR_OUT_OF_MEMORY;
            }
            part_bytes = part_elems * sizeof(float);
            part_buffer = [gd_sdpa_varlen_device(backend) newBufferWithLength:part_bytes
                                                                       options:MTLResourceStorageModePrivate];
            if (part_buffer == nil) {
                [encoder endEncoding];
                return GD_ERR_OUT_OF_MEMORY;
            }
        }

        [encoder setComputePipelineState:args->use_forward_stats != 0U ?
                                          gd_sdpa_varlen_bwd_saved_stats_dq_dh64_f16_pso(backend) :
                                          gd_sdpa_varlen_bwd_stats_dq_dh64_f16_pso(backend)];
        [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:grad_out->offset atIndex:0U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:q->offset atIndex:1U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:k->offset atIndex:2U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:v->offset atIndex:3U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:4U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(grad_q->buffer) offset:grad_q->offset atIndex:5U];
        [encoder setBytes:&p length:sizeof(p) atIndex:6U];
        [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:stats->offset atIndex:7U];
        [encoder setBuffer:compact ? q_blocks : gd_sdpa_varlen_buffer(cu_seqlens->buffer)
                     offset:compact ? 0U : cu_seqlens->offset
                    atIndex:8U];
        if (compact) {
            [encoder dispatchThreadgroupsWithIndirectBuffer:compact_dispatch
                                       indirectBufferOffset:GD_METAL_SDPA_COMPACT_Q_DISPATCH_OFFSET
                                      threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_CAUSAL_THREADS,
                                                                         1U,
                                                                         1U)];
        } else {
            [encoder dispatchThreadgroups:q_groups
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_CAUSAL_THREADS, 1U, 1U)];
        }

        if (n_splits > 1U) {
            const NSUInteger reduce_count = (NSUInteger)p.total_tokens * (NSUInteger)p.hkv *
                                            (NSUInteger)GD_METAL_SDPA_DHT;
            [encoder setComputePipelineState:gd_sdpa_varlen_bwd_dkv_split_dh64_f16_pso(backend)];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:grad_out->offset atIndex:0U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:q->offset atIndex:1U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:k->offset atIndex:2U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:v->offset atIndex:3U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:4U];
            [encoder setBuffer:part_buffer offset:0U atIndex:5U];
            [encoder setBytes:&p length:sizeof(p) atIndex:6U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:stats->offset atIndex:7U];
            [encoder setBuffer:compact ? k_blocks : gd_sdpa_varlen_buffer(cu_seqlens->buffer)
                         offset:compact ? 0U : cu_seqlens->offset
                        atIndex:8U];
            if (compact) {
                [encoder dispatchThreadgroupsWithIndirectBuffer:compact_dispatch
                                           indirectBufferOffset:GD_METAL_SDPA_COMPACT_K_DISPATCH_OFFSET
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_DKV_WIDE_THREADS,
                                                                             1U,
                                                                             1U)];
            } else {
                [encoder dispatchThreadgroups:MTLSizeMake(kv_groups.width * (NSUInteger)n_splits, 1U, 1U)
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_DKV_WIDE_THREADS, 1U, 1U)];
            }

            [encoder setComputePipelineState:gd_sdpa_varlen_bwd_dkv_reduce_f16_pso(backend)];
            [encoder setBuffer:part_buffer offset:0U atIndex:0U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_k->buffer) offset:grad_k->offset atIndex:1U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_v->buffer) offset:grad_v->offset atIndex:2U];
            [encoder setBytes:&p length:sizeof(p) atIndex:3U];
            [encoder dispatchThreads:MTLSizeMake(reduce_count, 1U, 1U)
                 threadsPerThreadgroup:MTLSizeMake(256U, 1U, 1U)];
        } else {
            [encoder setComputePipelineState:gd_sdpa_varlen_bwd_dkv_dh64_f16_pso(backend)];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:grad_out->offset atIndex:0U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:q->offset atIndex:1U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:k->offset atIndex:2U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:v->offset atIndex:3U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:cu_seqlens->offset atIndex:4U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_k->buffer) offset:grad_k->offset atIndex:5U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(grad_v->buffer) offset:grad_v->offset atIndex:6U];
            [encoder setBytes:&p length:sizeof(p) atIndex:7U];
            [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:stats->offset atIndex:8U];
            [encoder setBuffer:compact ? k_blocks : gd_sdpa_varlen_buffer(cu_seqlens->buffer)
                         offset:compact ? 0U : cu_seqlens->offset
                        atIndex:9U];
            if (compact) {
                [encoder dispatchThreadgroupsWithIndirectBuffer:compact_dispatch
                                           indirectBufferOffset:GD_METAL_SDPA_COMPACT_K_DISPATCH_OFFSET
                                          threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_DKV_WIDE_THREADS,
                                                                             1U,
                                                                             1U)];
            } else {
                [encoder dispatchThreadgroups:kv_groups
                        threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_SDPA_DKV_WIDE_THREADS, 1U, 1U)];
            }
        }
        [encoder endEncoding];
        return gd_metal_finish_immediate(command_buffer, immediate);
    }
    q_groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq * (NSUInteger)p.n_qb_max, 1U, 1U);
    kv_groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hkv * (NSUInteger)p.n_qb_max, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)GD_METAL_SDPA_BQ, 1U, 1U);

    [encoder setComputePipelineState:gd_sdpa_varlen_bwd_stats_pso(backend)];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:0U atIndex:5U];
    [encoder setBytes:&p length:sizeof(p) atIndex:6U];
    [encoder dispatchThreadgroups:q_groups threadsPerThreadgroup:threads];

    [encoder setComputePipelineState:gd_sdpa_varlen_bwd_pso(backend)];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_q->buffer) offset:0U atIndex:5U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:0U atIndex:6U];
    [encoder setBytes:&p length:sizeof(p) atIndex:7U];
    [encoder dispatchThreadgroups:q_groups threadsPerThreadgroup:threads];

    [encoder setComputePipelineState:gd_sdpa_varlen_bwd_dkv_pso(backend)];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(q->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(k->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(v->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(cu_seqlens->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_k->buffer) offset:0U atIndex:5U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(grad_v->buffer) offset:0U atIndex:6U];
    [encoder setBuffer:gd_sdpa_varlen_buffer(stats->buffer) offset:0U atIndex:7U];
    [encoder setBytes:&p length:sizeof(p) atIndex:8U];
    [encoder dispatchThreadgroups:kv_groups threadsPerThreadgroup:threads];

    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
