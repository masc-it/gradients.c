#include "../../backends/metal/metal_backend_internal.h"
#include "metal_minimax_m3_sparse_attention_types.h"

#include <gradients/tensor.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_minimax_m3_sparse_attention_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->minimax_m3_sparse_attention_pso;
}

static id<MTLComputePipelineState> gd_minimax_m3_sparse_attention_q4_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->minimax_m3_sparse_attention_q4_pso;
}

static id<MTLComputePipelineState> gd_minimax_m3_sparse_attention_bwd_dq_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->minimax_m3_sparse_attention_bwd_dq_pso;
}

static id<MTLComputePipelineState> gd_minimax_m3_sparse_attention_bwd_dkv_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->minimax_m3_sparse_attention_bwd_dkv_pso;
}

static id<MTLBuffer> gd_minimax_m3_sparse_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_minimax_m3_sparse_byte_range_valid(const gd_backend_buffer *buffer,
                                                   size_t offset,
                                                   size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_minimax_m3_sparse_dtype_size(uint32_t dtype, size_t *out_size)
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

static bool gd_minimax_m3_sparse_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_minimax_m3_sparse_dtype_size(view->dtype, &elem_size) ||
        view->count > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = view->count * elem_size;
    return gd_minimax_m3_sparse_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_minimax_m3_sparse_contiguous_valid(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        const uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride) {
            return false;
        }
        if (stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_minimax_m3_sparse_qkv_shapes_valid(const gd_backend_tensor_view *q,
                                                   const gd_backend_tensor_view *k,
                                                   const gd_backend_tensor_view *v,
                                                   const gd_backend_tensor_view *cu,
                                                   const gd_backend_tensor_view *topk,
                                                   const gd_backend_tensor_view *out,
                                                   const gd_backend_minimax_m3_sparse_args *args)
{
    if (!gd_minimax_m3_sparse_view_range_valid(q) || !gd_minimax_m3_sparse_view_range_valid(k) ||
        !gd_minimax_m3_sparse_view_range_valid(v) || !gd_minimax_m3_sparse_view_range_valid(cu) ||
        !gd_minimax_m3_sparse_view_range_valid(topk) || !gd_minimax_m3_sparse_view_range_valid(out) ||
        !gd_minimax_m3_sparse_contiguous_valid(q) || !gd_minimax_m3_sparse_contiguous_valid(k) ||
        !gd_minimax_m3_sparse_contiguous_valid(v) || !gd_minimax_m3_sparse_contiguous_valid(cu) ||
        !gd_minimax_m3_sparse_contiguous_valid(topk) || !gd_minimax_m3_sparse_contiguous_valid(out)) {
        return false;
    }
    if (q->dtype != (uint32_t)GD_DTYPE_F16 || k->dtype != q->dtype || v->dtype != q->dtype ||
        out->dtype != q->dtype || cu->dtype != (uint32_t)GD_DTYPE_I32 ||
        topk->dtype != (uint32_t)GD_DTYPE_I32 || q->rank != 3U || k->rank != 3U ||
        v->rank != 3U || out->rank != 3U || cu->rank != 1U || topk->rank != 3U ||
        cu->shape[0] < 2 || q->shape[0] <= 0 || q->shape[1] <= 0 || q->shape[2] <= 0 ||
        k->shape[0] != q->shape[0] || v->shape[0] != q->shape[0] ||
        k->shape[1] <= 0 || v->shape[1] != k->shape[1] || k->shape[2] != q->shape[2] ||
        v->shape[2] != q->shape[2] || out->shape[0] != q->shape[0] ||
        out->shape[1] != q->shape[1] || out->shape[2] != q->shape[2] ||
        (q->shape[1] % k->shape[1]) != 0 || q->shape[2] > (int64_t)GD_METAL_MINIMAX_M3_MAX_HEAD_DIM ||
        topk->shape[0] != k->shape[1] || topk->shape[1] != q->shape[0] ||
        args == NULL || topk->shape[2] != (int64_t)args->topk || args->topk == 0U ||
        args->topk > GD_METAL_MINIMAX_M3_MAX_TOPK || args->block_size == 0U ||
        args->block_size > GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE || args->max_seqlen == 0U ||
        args->scale != args->scale || args->scale <= 0.0f ||
        q->shape[0] > (int64_t)UINT32_MAX || q->shape[1] > (int64_t)UINT32_MAX ||
        k->shape[1] > (int64_t)UINT32_MAX || q->shape[2] > (int64_t)UINT32_MAX ||
        cu->shape[0] - 1 > (int64_t)UINT32_MAX) {
        return false;
    }
    return true;
}

static bool gd_minimax_m3_sparse_stats_valid(const gd_backend_tensor_view *q,
                                             const gd_backend_tensor_view *stats)
{
    return gd_minimax_m3_sparse_view_range_valid(stats) &&
           gd_minimax_m3_sparse_contiguous_valid(stats) &&
           stats->dtype == (uint32_t)GD_DTYPE_F32 && stats->rank == 3U &&
           stats->shape[0] == q->shape[0] && stats->shape[1] == q->shape[1] &&
           stats->shape[2] == 3;
}

static bool gd_minimax_m3_sparse_bwd_shapes_valid(const gd_backend_tensor_view *grad_out,
                                                   const gd_backend_tensor_view *q,
                                                   const gd_backend_tensor_view *k,
                                                   const gd_backend_tensor_view *v,
                                                   const gd_backend_tensor_view *cu,
                                                   const gd_backend_tensor_view *topk,
                                                   const gd_backend_tensor_view *grad_q,
                                                   const gd_backend_tensor_view *grad_k,
                                                   const gd_backend_tensor_view *grad_v,
                                                   const gd_backend_tensor_view *stats,
                                                   const gd_backend_minimax_m3_sparse_args *args)
{
    return gd_minimax_m3_sparse_qkv_shapes_valid(q, k, v, cu, topk, grad_q, args) &&
           gd_minimax_m3_sparse_view_range_valid(grad_out) &&
           gd_minimax_m3_sparse_contiguous_valid(grad_out) &&
           gd_minimax_m3_sparse_view_range_valid(grad_k) &&
           gd_minimax_m3_sparse_contiguous_valid(grad_k) &&
           gd_minimax_m3_sparse_view_range_valid(grad_v) &&
           gd_minimax_m3_sparse_contiguous_valid(grad_v) && grad_out->dtype == q->dtype &&
           grad_out->rank == 3U && grad_out->shape[0] == q->shape[0] &&
           grad_out->shape[1] == q->shape[1] && grad_out->shape[2] == q->shape[2] &&
           grad_k->dtype == k->dtype && grad_v->dtype == v->dtype && grad_k->rank == 3U &&
           grad_v->rank == 3U && grad_k->shape[0] == k->shape[0] &&
           grad_k->shape[1] == k->shape[1] && grad_k->shape[2] == k->shape[2] &&
           grad_v->shape[0] == v->shape[0] && grad_v->shape[1] == v->shape[1] &&
           grad_v->shape[2] == v->shape[2] && gd_minimax_m3_sparse_stats_valid(q, stats);
}

static void gd_minimax_m3_sparse_fill_args(gd_metal_minimax_m3_sparse_args *p,
                                           const gd_backend_tensor_view *q,
                                           const gd_backend_tensor_view *k,
                                           const gd_backend_tensor_view *v,
                                           const gd_backend_tensor_view *cu,
                                           const gd_backend_tensor_view *topk,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_q,
                                           const gd_backend_tensor_view *grad_k,
                                           const gd_backend_tensor_view *grad_v,
                                           const gd_backend_tensor_view *stats,
                                           const gd_backend_minimax_m3_sparse_args *args)
{
    memset(p, 0, sizeof(*p));
    p->q_offset = (uint64_t)q->offset;
    p->k_offset = (uint64_t)k->offset;
    p->v_offset = v != NULL ? (uint64_t)v->offset : 0U;
    p->cu_offset = (uint64_t)cu->offset;
    p->topk_offset = (uint64_t)topk->offset;
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
    p->block_size = args->block_size;
    p->topk = args->topk;
    p->init_blocks = args->init_blocks;
    p->local_blocks = args->local_blocks;
    p->scale = args->scale;
}

gd_status gd_backend_minimax_m3_sparse_attention(gd_backend *backend,
                                                 const gd_backend_tensor_view *q,
                                                 const gd_backend_tensor_view *k,
                                                 const gd_backend_tensor_view *v,
                                                 const gd_backend_tensor_view *cu_seqlens,
                                                 const gd_backend_tensor_view *topk_idx,
                                                 const gd_backend_tensor_view *out,
                                                 const gd_backend_minimax_m3_sparse_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_minimax_m3_sparse_args p;
    bool immediate;
    gd_status st;
    if (backend == NULL ||
        !gd_minimax_m3_sparse_qkv_shapes_valid(q, k, v, cu_seqlens, topk_idx, out, args)) {
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
    gd_minimax_m3_sparse_fill_args(&p, q, k, v, cu_seqlens, topk_idx, out, NULL, NULL, NULL, NULL, NULL, args);
    /* The default path is a 64-query row-tile kernel. Keep the old q4 PSO compiled
     * only as an emergency fallback for invalidly-large top-k experiments. */
    if (p.topk > GD_METAL_MINIMAX_M3_MAX_TOPK) {
        [encoder setComputePipelineState:gd_minimax_m3_sparse_attention_q4_pso(backend)];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(q->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(k->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(v->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(cu_seqlens->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(topk_idx->buffer) offset:0U atIndex:4U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(out->buffer) offset:0U atIndex:5U];
        [encoder setBytes:&p length:sizeof(p) atIndex:6U];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((p.max_seqlen + GD_METAL_MINIMAX_M3_QTILE - 1U) /
                                                               GD_METAL_MINIMAX_M3_QTILE),
                                                  (NSUInteger)p.hq,
                                                  (NSUInteger)p.batch)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_MINIMAX_M3_QTILE_THREADS, 1U, 1U)];
    } else {
        [encoder setComputePipelineState:gd_minimax_m3_sparse_attention_pso(backend)];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(q->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(k->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(v->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(cu_seqlens->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(topk_idx->buffer) offset:0U atIndex:4U];
        [encoder setBuffer:gd_minimax_m3_sparse_buffer(out->buffer) offset:0U atIndex:5U];
        [encoder setBytes:&p length:sizeof(p) atIndex:6U];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((p.max_seqlen + GD_METAL_MINIMAX_M3_ATTENTION_THREADS - 1U) /
                                                               GD_METAL_MINIMAX_M3_ATTENTION_THREADS),
                                                  (NSUInteger)p.hq,
                                                  (NSUInteger)p.batch)
                threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_MINIMAX_M3_ATTENTION_THREADS, 1U, 1U)];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_minimax_m3_sparse_attention_backward(
    gd_backend *backend,
    const gd_backend_tensor_view *grad_out,
    const gd_backend_tensor_view *q,
    const gd_backend_tensor_view *k,
    const gd_backend_tensor_view *v,
    const gd_backend_tensor_view *cu_seqlens,
    const gd_backend_tensor_view *topk_idx,
    const gd_backend_tensor_view *grad_q,
    const gd_backend_tensor_view *grad_k,
    const gd_backend_tensor_view *grad_v,
    const gd_backend_tensor_view *stats,
    const gd_backend_minimax_m3_sparse_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_minimax_m3_sparse_args p;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_minimax_m3_sparse_bwd_shapes_valid(grad_out,
                                                                  q,
                                                                  k,
                                                                  v,
                                                                  cu_seqlens,
                                                                  topk_idx,
                                                                  grad_q,
                                                                  grad_k,
                                                                  grad_v,
                                                                  stats,
                                                                  args)) {
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
    gd_minimax_m3_sparse_fill_args(&p,
                                   q,
                                   k,
                                   v,
                                   cu_seqlens,
                                   topk_idx,
                                   NULL,
                                   grad_out,
                                   grad_q,
                                   grad_k,
                                   grad_v,
                                   stats,
                                   args);
    [encoder setComputePipelineState:gd_minimax_m3_sparse_attention_bwd_dq_pso(backend)];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(q->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(k->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(v->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(cu_seqlens->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(topk_idx->buffer) offset:0U atIndex:5U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(grad_q->buffer) offset:0U atIndex:6U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(stats->buffer) offset:0U atIndex:7U];
    [encoder setBytes:&p length:sizeof(p) atIndex:8U];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((p.max_seqlen + GD_METAL_MINIMAX_M3_ATTENTION_THREADS - 1U) /
                                                           GD_METAL_MINIMAX_M3_ATTENTION_THREADS),
                                              (NSUInteger)p.hq,
                                              (NSUInteger)p.batch)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_MINIMAX_M3_ATTENTION_THREADS, 1U, 1U)];

    [encoder setComputePipelineState:gd_minimax_m3_sparse_attention_bwd_dkv_pso(backend)];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(q->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(k->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(v->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(cu_seqlens->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(topk_idx->buffer) offset:0U atIndex:5U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(grad_k->buffer) offset:0U atIndex:6U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(grad_v->buffer) offset:0U atIndex:7U];
    [encoder setBuffer:gd_minimax_m3_sparse_buffer(stats->buffer) offset:0U atIndex:8U];
    [encoder setBytes:&p length:sizeof(p) atIndex:9U];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)((p.max_seqlen + GD_METAL_MINIMAX_M3_ATTENTION_THREADS - 1U) /
                                                           GD_METAL_MINIMAX_M3_ATTENTION_THREADS),
                                              (NSUInteger)p.hkv,
                                              (NSUInteger)p.batch)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)GD_METAL_MINIMAX_M3_ATTENTION_THREADS, 1U, 1U)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
