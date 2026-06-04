#include "metal_reduce_common.h"

static id<MTLComputePipelineState> gd_reduce_contiguous_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->reduce_contiguous_pso;
}

static id<MTLComputePipelineState> gd_reduce_axis_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->reduce_axis_pso;
}

static id<MTLComputePipelineState> gd_broadcast_axis_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->broadcast_axis_pso;
}

static id<MTLComputePipelineState> gd_broadcast_to_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->broadcast_to_pso;
}

static gd_status gd_reduce_contiguous_validate(const gd_backend_tensor_view *src,
                                               const gd_backend_tensor_view *dst)
{
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->dtype != dst->dtype || src->count < dst->count || src->count > UINT32_MAX ||
        dst->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_reduce_axis_counts(const gd_backend_tensor_view *full,
                                  uint32_t axis,
                                  size_t *out_outer,
                                  size_t *out_reduce,
                                  size_t *out_inner)
{
    uint32_t dim;
    size_t outer = 1U;
    size_t inner = 1U;
    size_t reduce;
    if (full == NULL || out_outer == NULL || out_reduce == NULL || out_inner == NULL ||
        full->rank == 0U || axis >= full->rank || full->rank > GD_MAX_DIMS ||
        full->shape[axis] <= 0) {
        return false;
    }
    for (dim = 0U; dim < axis; ++dim) {
        if (full->shape[dim] <= 0 || outer > SIZE_MAX / (size_t)full->shape[dim]) {
            return false;
        }
        outer *= (size_t)full->shape[dim];
    }
    for (dim = axis + 1U; dim < full->rank; ++dim) {
        if (full->shape[dim] <= 0 || inner > SIZE_MAX / (size_t)full->shape[dim]) {
            return false;
        }
        inner *= (size_t)full->shape[dim];
    }
    reduce = (size_t)full->shape[axis];
    *out_outer = outer;
    *out_reduce = reduce;
    *out_inner = inner;
    return true;
}

static bool gd_reduce_axis_reduced_shape_compatible(const gd_backend_tensor_view *full,
                                                    const gd_backend_tensor_view *reduced,
                                                    uint32_t axis)
{
    uint32_t dim;
    size_t outer;
    size_t reduce;
    size_t inner;
    if (full == NULL || reduced == NULL || axis >= full->rank || full->rank == 0U ||
        full->rank > GD_MAX_DIMS || reduced->rank > full->rank || full->dtype != reduced->dtype ||
        !gd_reduce_axis_counts(full, axis, &outer, &reduce, &inner) || reduce == 0U) {
        return false;
    }
    (void)outer;
    (void)inner;
    if (full->count / reduce != reduced->count || full->count % reduce != 0U) {
        return false;
    }
    if (reduced->rank == full->rank) {
        for (dim = 0U; dim < full->rank; ++dim) {
            int64_t want = dim == axis ? 1 : full->shape[dim];
            if (reduced->shape[dim] != want) {
                return false;
            }
        }
        return true;
    }
    if (reduced->rank + 1U == full->rank) {
        for (dim = 0U; dim < full->rank; ++dim) {
            uint32_t reduced_dim;
            if (dim == axis) {
                continue;
            }
            reduced_dim = dim < axis ? dim : dim - 1U;
            if (reduced->shape[reduced_dim] != full->shape[dim]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

static gd_status gd_reduce_axis_validate(const gd_backend_tensor_view *src,
                                         const gd_backend_tensor_view *dst,
                                         uint32_t axis)
{
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->count > UINT32_MAX || dst->count > UINT32_MAX ||
        !gd_reduce_axis_reduced_shape_compatible(src, dst, axis)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_broadcast_axis_validate(const gd_backend_tensor_view *src,
                                            const gd_backend_tensor_view *dst,
                                            uint32_t axis)
{
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->count > UINT32_MAX || dst->count > UINT32_MAX ||
        !gd_reduce_axis_reduced_shape_compatible(dst, src, axis)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_reduce_axis_fill_args(gd_metal_reduce_axis_args *args,
                                     const gd_backend_tensor_view *src,
                                     const gd_backend_tensor_view *dst,
                                     uint32_t axis,
                                     float scale)
{
    size_t outer;
    size_t reduce;
    size_t inner;
    memset(args, 0, sizeof(*args));
    (void)gd_reduce_axis_counts(src, axis, &outer, &reduce, &inner);
    args->src_offset = (uint64_t)src->offset;
    args->dst_offset = (uint64_t)dst->offset;
    args->outer_count = (uint64_t)outer;
    args->reduce_count = (uint64_t)reduce;
    args->inner_count = (uint64_t)inner;
    args->dst_count = (uint64_t)dst->count;
    args->dtype = src->dtype;
    args->scale = scale;
}

static void gd_broadcast_axis_fill_args(gd_metal_reduce_axis_args *args,
                                        const gd_backend_tensor_view *src,
                                        const gd_backend_tensor_view *dst,
                                        uint32_t axis,
                                        float scale)
{
    size_t outer;
    size_t reduce;
    size_t inner;
    memset(args, 0, sizeof(*args));
    (void)gd_reduce_axis_counts(dst, axis, &outer, &reduce, &inner);
    args->src_offset = (uint64_t)src->offset;
    args->dst_offset = (uint64_t)dst->offset;
    args->outer_count = (uint64_t)outer;
    args->reduce_count = (uint64_t)reduce;
    args->inner_count = (uint64_t)inner;
    args->dst_count = (uint64_t)dst->count;
    args->dtype = dst->dtype;
    args->scale = scale;
}

static gd_status gd_broadcast_to_validate(const gd_backend_tensor_view *src,
                                          const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint64_t stride;
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->dtype != dst->dtype || src->rank > dst->rank || src->rank > GD_MAX_DIMS ||
        dst->rank > GD_MAX_DIMS || dst->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (dim = 0U; dim < dst->rank; ++dim) {
        if (!gd_metal_broadcast_dim(src, dst, dim, &stride)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

gd_status gd_backend_reduce_contiguous(gd_backend *backend,
                                       const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst,
                                       float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_reduce_contiguous_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_contiguous_validate(src, dst);
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
    memset(&args, 0, sizeof(args));
    args.src_offset = (uint64_t)src->offset;
    args.dst_offset = (uint64_t)dst->offset;
    args.src_count = (uint64_t)src->count;
    args.dst_count = (uint64_t)dst->count;
    args.dtype = src->dtype;
    args.scale = scale;
    [encoder setComputePipelineState:gd_reduce_contiguous_pso(backend)];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)((dst->count + GD_METAL_REDUCE_CONTIGUOUS_SIMDGROUPS - 1U) /
                                    GD_METAL_REDUCE_CONTIGUOUS_SIMDGROUPS),
                       1U,
                       1U);
    threads = MTLSizeMake(32U, GD_METAL_REDUCE_CONTIGUOUS_SIMDGROUPS, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_reduce_axis(gd_backend *backend,
                                 const gd_backend_tensor_view *src,
                                 const gd_backend_tensor_view *dst,
                                 uint32_t axis,
                                 float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_reduce_axis_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_axis_validate(src, dst, axis);
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
    gd_reduce_axis_fill_args(&args, src, dst, axis, scale);
    [encoder setComputePipelineState:gd_reduce_axis_pso(backend)];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)((dst->count + GD_METAL_REDUCE_AXIS_SIMDGROUPS - 1U) /
                                    GD_METAL_REDUCE_AXIS_SIMDGROUPS),
                       1U,
                       1U);
    threads = MTLSizeMake(32U, GD_METAL_REDUCE_AXIS_SIMDGROUPS, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_broadcast_axis(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    uint32_t axis,
                                    float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_reduce_axis_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_broadcast_axis_validate(src, dst, axis);
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
    gd_broadcast_axis_fill_args(&args, src, dst, axis, scale);
    [encoder setComputePipelineState:gd_broadcast_axis_pso(backend)];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    {
        NSUInteger tx = args.inner_count < GD_METAL_REDUCE_MAX_THREADS_PER_GROUP ?
                            (NSUInteger)args.inner_count : GD_METAL_REDUCE_MAX_THREADS_PER_GROUP;
        NSUInteger ty_cap;
        NSUInteger ty;
        if (tx == 0U) {
            tx = 1U;
        }
        ty_cap = GD_METAL_REDUCE_MAX_THREADS_PER_GROUP / tx;
        if (ty_cap == 0U) {
            ty_cap = 1U;
        }
        ty = args.reduce_count < (uint64_t)ty_cap ? (NSUInteger)args.reduce_count : ty_cap;
        if (ty == 0U) {
            ty = 1U;
        }
        grid = MTLSizeMake((NSUInteger)args.inner_count,
                           (NSUInteger)args.reduce_count,
                           (NSUInteger)args.outer_count);
        threads = MTLSizeMake(tx, ty, 1U);
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_broadcast_to(gd_backend *backend,
                                  const gd_backend_tensor_view *src,
                                  const gd_backend_tensor_view *dst,
                                  float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_broadcast_to_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    uint32_t dim;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_broadcast_to_validate(src, dst);
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
    memset(&args, 0, sizeof(args));
    args.src_offset = (uint64_t)src->offset;
    args.dst_offset = (uint64_t)dst->offset;
    args.dst_count = (uint64_t)dst->count;
    args.dtype = dst->dtype;
    args.rank = dst->rank;
    args.scale = scale;
    for (dim = 0U; dim < dst->rank; ++dim) {
        uint64_t stride = 0U;
        args.dst_shape[dim] = (uint64_t)dst->shape[dim];
        (void)gd_metal_broadcast_dim(src, dst, dim, &stride);
        args.src_strides[dim] = stride;
    }
    [encoder setComputePipelineState:gd_broadcast_to_pso(backend)];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)dst->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(dst->count < GD_METAL_REDUCE_MAX_THREADS_PER_GROUP ?
                                       dst->count : GD_METAL_REDUCE_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
