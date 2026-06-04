#include "metal_binary_common.h"

static id<MTLComputePipelineState> gd_binary_reduce_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_reduce_pso;
}

static id<MTLComputePipelineState> gd_binary_reduce_suffix_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_reduce_suffix_pso;
}

static bool gd_binary_reduce_shape_compatible(const gd_backend_tensor_view *src,
                                              const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    if (src == NULL || dst == NULL || src->rank > GD_MAX_DIMS || dst->rank > src->rank) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t src_dim = src->shape[dim];
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        int64_t dst_stride = dim < prefix ? 0 : dst->strides[dim - prefix];
        if (src_dim <= 0 || dst_dim <= 0 || src->strides[dim] < 0 || dst_stride < 0 ||
            (dst_dim != src_dim && dst_dim != 1)) {
            return false;
        }
    }
    return true;
}

static bool gd_binary_reduce_is_suffix(const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    bool suffix_started = false;
    if (src == NULL || dst == NULL || dst->rank > src->rank || dst->count == 0U ||
        src->count % dst->count != 0U) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        if (!suffix_started && dst_dim == 1 && src->shape[dim] != 1) {
            continue;
        }
        suffix_started = true;
        if (dst_dim != src->shape[dim]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_binary_reduce_validate_views(const gd_backend_tensor_view *src,
                                                 const gd_backend_tensor_view *dst)
{
    if (!gd_metal_binary_view_range_valid(src) || !gd_metal_binary_view_range_valid(dst) ||
        src->dtype != dst->dtype || src->count <= dst->count || src->count > UINT32_MAX ||
        dst->count > UINT32_MAX || !gd_binary_reduce_shape_compatible(src, dst)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_binary_reduce_fill_args(gd_metal_binary_reduce_args *args,
                                       const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst,
                                       float scale)
{
    uint32_t dim;
    uint32_t prefix = src->rank - dst->rank;
    memset(args, 0, sizeof(*args));
    args->src_offset = (uint64_t)src->offset;
    args->dst_offset = (uint64_t)dst->offset;
    args->src_count = (uint64_t)src->count;
    args->dst_count = (uint64_t)dst->count;
    args->dtype = src->dtype;
    args->rank = src->rank;
    args->scale = scale;
    for (dim = 0U; dim < src->rank; ++dim) {
        args->src_shape[dim] = (uint64_t)src->shape[dim];
        args->src_strides[dim] = (uint64_t)src->strides[dim];
        if (dim < prefix) {
            args->dst_shape[dim] = 1U;
            args->dst_strides[dim] = 0U;
        } else {
            uint32_t dst_dim = dim - prefix;
            args->dst_shape[dim] = (uint64_t)dst->shape[dst_dim];
            args->dst_strides[dim] = (uint64_t)dst->strides[dst_dim];
        }
    }
}

gd_status gd_backend_reduce_broadcast(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_binary_reduce_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool suffix;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_binary_reduce_validate_views(src, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_binary_reduce_fill_args(&args, src, dst, scale);
    suffix = gd_binary_reduce_is_suffix(src, dst);
    [encoder setComputePipelineState:suffix ? gd_binary_reduce_suffix_pso(backend) : gd_binary_reduce_pso(backend)];
    [encoder setBuffer:gd_metal_binary_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_binary_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    if (suffix) {
        grid = MTLSizeMake((NSUInteger)((dst->count + GD_METAL_BINARY_REDUCE_SUFFIX_SIMDGROUPS - 1U) /
                                        GD_METAL_BINARY_REDUCE_SUFFIX_SIMDGROUPS),
                           1U,
                           1U);
        threads = MTLSizeMake(32U, GD_METAL_BINARY_REDUCE_SUFFIX_SIMDGROUPS, 1U);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    } else {
        grid = MTLSizeMake((NSUInteger)dst->count, 1U, 1U);
        threads = MTLSizeMake((NSUInteger)(dst->count < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ?
                                           dst->count : GD_METAL_BINARY_MAX_THREADS_PER_GROUP),
                              1U,
                              1U);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
