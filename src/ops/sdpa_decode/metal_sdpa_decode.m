#include "../../backends/metal/metal_backend_internal.h"
#include "metal_sdpa_decode_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_sdpa_decode_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->sdpa_decode_pso;
}

static id<MTLBuffer> gd_sdpa_decode_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_sdpa_decode_byte_range_valid(const gd_backend_buffer *buffer,
                                            size_t offset,
                                            size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_sdpa_decode_dtype_size(uint32_t dtype, size_t *out_size)
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

static bool gd_sdpa_decode_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_sdpa_decode_dtype_size(view->dtype, &elem_size) || view->count > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = view->count * elem_size;
    return gd_sdpa_decode_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_sdpa_decode_contiguous_valid(const gd_backend_tensor_view *view)
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

static bool gd_sdpa_decode_shapes_common_valid(const gd_backend_tensor_view *q,
                                               const gd_backend_tensor_view *k,
                                               const gd_backend_tensor_view *v,
                                               const gd_backend_tensor_view *out)
{
    uint32_t i;
    if (!gd_sdpa_decode_view_range_valid(q) || !gd_sdpa_decode_view_range_valid(k) ||
        !gd_sdpa_decode_view_range_valid(v) || !gd_sdpa_decode_view_range_valid(out) ||
        !gd_sdpa_decode_contiguous_valid(q) || !gd_sdpa_decode_contiguous_valid(k) ||
        !gd_sdpa_decode_contiguous_valid(v) || !gd_sdpa_decode_contiguous_valid(out)) {
        return false;
    }
    if ((q->dtype != (uint32_t)GD_DTYPE_F16 && q->dtype != (uint32_t)GD_DTYPE_F32) ||
        k->dtype != q->dtype || v->dtype != q->dtype || out->dtype != q->dtype ||
        q->rank != 4U || k->rank != 4U || v->rank != 4U || out->rank != 4U) {
        return false;
    }
    for (i = 0U; i < 4U; ++i) {
        if (k->shape[i] != v->shape[i]) {
            return false;
        }
    }
    return q->shape[0] == k->shape[0] && q->shape[1] > 0 && k->shape[1] > 0 &&
           q->shape[2] > 0 && k->shape[2] > 0 && q->shape[3] > 0 &&
           q->shape[3] == k->shape[3] && (q->shape[2] % k->shape[2]) == 0 &&
           out->shape[0] == q->shape[0] && out->shape[1] == q->shape[1] &&
           out->shape[2] == q->shape[2] && out->shape[3] == q->shape[3] &&
           q->shape[0] <= (int64_t)UINT32_MAX && q->shape[1] <= (int64_t)UINT32_MAX &&
           k->shape[1] <= (int64_t)UINT32_MAX && q->shape[2] <= (int64_t)UINT32_MAX &&
           k->shape[2] <= (int64_t)UINT32_MAX && q->shape[3] <= (int64_t)GD_METAL_SDPA_DECODE_DHT;
}

static bool gd_sdpa_decode_shapes_valid(const gd_backend_tensor_view *q,
                                        const gd_backend_tensor_view *k,
                                        const gd_backend_tensor_view *v,
                                        const gd_backend_tensor_view *pos,
                                        const gd_backend_tensor_view *out)
{
    return gd_sdpa_decode_shapes_common_valid(q, k, v, out) &&
           gd_sdpa_decode_view_range_valid(pos) && gd_sdpa_decode_contiguous_valid(pos) &&
           pos->dtype == (uint32_t)GD_DTYPE_I32 && pos->rank == 0U && pos->count == 1U;
}

static bool gd_sdpa_decode_shapes_at_valid(const gd_backend_tensor_view *q,
                                           const gd_backend_tensor_view *k,
                                           const gd_backend_tensor_view *v,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_sdpa_decode_args *args)
{
    return gd_sdpa_decode_shapes_common_valid(q, k, v, out) && args != NULL &&
           args->cache_pos >= 0 && (int64_t)args->cache_pos <= k->shape[1] &&
           q->shape[1] <= k->shape[1] - (int64_t)args->cache_pos;
}

static bool gd_sdpa_decode_args_valid(const gd_backend_sdpa_decode_args *args,
                                      const gd_backend_tensor_view *k)
{
    return args != NULL && args->scale == args->scale && args->scale > 0.0f &&
           args->prefix_len <= (uint32_t)k->shape[1];
}

static void gd_sdpa_decode_fill_metal_args(gd_metal_sdpa_decode_args *p,
                                           const gd_backend_tensor_view *q,
                                           const gd_backend_tensor_view *k,
                                           const gd_backend_tensor_view *pos,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_sdpa_decode_args *args)
{
    memset(p, 0, sizeof(*p));
    p->q_offset = (uint64_t)q->offset;
    p->k_offset = (uint64_t)k->offset;
    p->pos_offset = pos != NULL ? (uint64_t)pos->offset : 0U;
    p->out_offset = (uint64_t)out->offset;
    p->batch = (uint32_t)q->shape[0];
    p->tq = (uint32_t)q->shape[1];
    p->tmax = (uint32_t)k->shape[1];
    p->hq = (uint32_t)q->shape[2];
    p->hkv = (uint32_t)k->shape[2];
    p->dh = (uint32_t)q->shape[3];
    p->window = args->sliding_window;
    p->prefix_len = args->prefix_len;
    p->dtype = q->dtype;
    p->scale = args->scale;
    p->use_pos_buffer = pos != NULL ? 1U : 0U;
    p->cache_pos = args->cache_pos >= 0 ? (uint32_t)args->cache_pos : 0U;
}

gd_status gd_backend_sdpa_decode(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k_cache,
                                 const gd_backend_tensor_view *v_cache,
                                 const gd_backend_tensor_view *cache_pos,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_sdpa_decode_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_sdpa_decode_args p;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_sdpa_decode_shapes_valid(q, k_cache, v_cache, cache_pos, out) ||
        !gd_sdpa_decode_args_valid(args, k_cache)) {
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
    gd_sdpa_decode_fill_metal_args(&p, q, k_cache, cache_pos, out, args);
    p.v_offset = (uint64_t)v_cache->offset;
    [encoder setComputePipelineState:gd_sdpa_decode_pso(backend)];
    [encoder setBuffer:gd_sdpa_decode_buffer(q->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sdpa_decode_buffer(k_cache->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sdpa_decode_buffer(v_cache->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_sdpa_decode_buffer(cache_pos->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_sdpa_decode_buffer(out->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&p length:sizeof(p) atIndex:5U];
    groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq *
                             ((NSUInteger)(p.tq + GD_METAL_SDPA_DECODE_BQ - 1U) /
                              (NSUInteger)GD_METAL_SDPA_DECODE_BQ),
                         1U,
                         1U);
    threads = MTLSizeMake((NSUInteger)GD_METAL_SDPA_DECODE_BQ, 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_sdpa_decode_at(gd_backend *backend,
                                    const gd_backend_tensor_view *q,
                                    const gd_backend_tensor_view *k_cache,
                                    const gd_backend_tensor_view *v_cache,
                                    const gd_backend_tensor_view *out,
                                    const gd_backend_sdpa_decode_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_sdpa_decode_args p;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_sdpa_decode_shapes_at_valid(q, k_cache, v_cache, out, args) ||
        !gd_sdpa_decode_args_valid(args, k_cache)) {
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
    gd_sdpa_decode_fill_metal_args(&p, q, k_cache, NULL, out, args);
    p.v_offset = (uint64_t)v_cache->offset;
    [encoder setComputePipelineState:gd_sdpa_decode_pso(backend)];
    [encoder setBuffer:gd_sdpa_decode_buffer(q->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sdpa_decode_buffer(k_cache->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sdpa_decode_buffer(v_cache->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_sdpa_decode_buffer(q->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_sdpa_decode_buffer(out->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&p length:sizeof(p) atIndex:5U];
    groups = MTLSizeMake((NSUInteger)p.batch * (NSUInteger)p.hq *
                             ((NSUInteger)(p.tq + GD_METAL_SDPA_DECODE_BQ - 1U) /
                              (NSUInteger)GD_METAL_SDPA_DECODE_BQ),
                         1U,
                         1U);
    threads = MTLSizeMake((NSUInteger)GD_METAL_SDPA_DECODE_BQ, 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
