#include "metal_reduce_common.h"

static id<MTLComputePipelineState> gd_reduce_contiguous_pso_for(gd_backend *backend,
                                                                uint32_t src_dtype,
                                                                uint32_t dst_dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F16 && dst_dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->reduce_contiguous_f16_to_f16_pso;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F16 && dst_dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->reduce_contiguous_f16_to_f32_pso;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F32 && dst_dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->reduce_contiguous_f32_to_f32_pso;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F32 && dst_dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->reduce_contiguous_f32_to_f16_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_reduce_axis_pso_for(gd_backend *backend,
                                                          uint32_t dtype,
                                                          bool last_axis)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return last_axis ? (__bridge id<MTLComputePipelineState>)backend->reduce_axis_last_f16_pso :
                           (__bridge id<MTLComputePipelineState>)backend->reduce_axis_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return last_axis ? (__bridge id<MTLComputePipelineState>)backend->reduce_axis_last_f32_pso :
                           (__bridge id<MTLComputePipelineState>)backend->reduce_axis_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_broadcast_axis_pso_for(gd_backend *backend,
                                                               uint32_t dtype,
                                                               bool last_axis)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return last_axis ? (__bridge id<MTLComputePipelineState>)backend->broadcast_axis_last_f16_pso :
                           (__bridge id<MTLComputePipelineState>)backend->broadcast_axis_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return last_axis ? (__bridge id<MTLComputePipelineState>)backend->broadcast_axis_last_f32_pso :
                           (__bridge id<MTLComputePipelineState>)backend->broadcast_axis_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_broadcast_to_pso_for(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->broadcast_to_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->broadcast_to_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_broadcast_scalar_pso_for(gd_backend *backend,
                                                                 uint32_t src_dtype,
                                                                 uint32_t dst_dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F16 && dst_dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->broadcast_scalar_f16_pso;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F32 && dst_dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->broadcast_scalar_f32_pso;
    }
    if (src_dtype == (uint32_t)GD_DTYPE_F32 && dst_dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->broadcast_scalar_f32_to_f16_pso;
    }
    return nil;
}

static bool gd_reduce_dtype_supported(uint32_t dtype)
{
    return dtype == (uint32_t)GD_DTYPE_F16 || dtype == (uint32_t)GD_DTYPE_F32;
}

static size_t gd_reduce_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static NSUInteger gd_reduce_threads_for_count(size_t count)
{
    if (count == 0U) {
        return 1U;
    }
    return count < GD_METAL_REDUCE_MAX_THREADS_PER_GROUP ? (NSUInteger)count :
                                                           GD_METAL_REDUCE_MAX_THREADS_PER_GROUP;
}

static uint32_t gd_reduce_simdgroups_for_count(uint64_t count)
{
    if (count <= 128U) {
        return 1U;
    }
    if (count <= 512U) {
        return 2U;
    }
    if (count <= 2048U) {
        return 4U;
    }
    return GD_METAL_REDUCE_MAX_SIMDGROUPS;
}

static uint32_t gd_reduce_contiguous_simdgroups_for_stage(uint64_t src_count, uint64_t dst_count)
{
    uint64_t chunk;
    if (dst_count == 0U) {
        return 1U;
    }
    chunk = (src_count + dst_count - 1U) / dst_count;
    if (dst_count > 1U) {
        if (chunk <= 8192U) {
            return 1U;
        }
        if (chunk <= 32768U) {
            return 2U;
        }
        if (chunk <= 131072U) {
            return 4U;
        }
    }
    return gd_reduce_simdgroups_for_count(chunk);
}


static gd_status gd_reduce_contiguous_validate(const gd_backend_tensor_view *src,
                                               const gd_backend_tensor_view *dst)
{
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->count < dst->count || src->count > UINT32_MAX || dst->count > UINT32_MAX ||
        !gd_reduce_dtype_supported(src->dtype) || !gd_reduce_dtype_supported(dst->dtype)) {
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
        src->count > UINT32_MAX || dst->count > UINT32_MAX || !gd_reduce_dtype_supported(src->dtype) ||
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
        src->count > UINT32_MAX || dst->count > UINT32_MAX || !gd_reduce_dtype_supported(src->dtype) ||
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
    args->simdgroups_per_output = gd_reduce_simdgroups_for_count((uint64_t)reduce);
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
    args->simdgroups_per_output = 1U;
    args->scale = scale;
}

static gd_status gd_broadcast_to_validate(const gd_backend_tensor_view *src,
                                          const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint64_t stride;
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->dtype != dst->dtype || !gd_reduce_dtype_supported(src->dtype) || src->rank > dst->rank ||
        src->rank > GD_MAX_DIMS || dst->rank > GD_MAX_DIMS || dst->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (dim = 0U; dim < dst->rank; ++dim) {
        if (!gd_metal_broadcast_dim(src, dst, dim, &stride)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_broadcast_scalar_validate(const gd_backend_tensor_view *src,
                                              const gd_backend_tensor_view *dst)
{
    size_t src_elem_size;
    size_t dst_elem_size;
    bool dtype_pair_ok;
    if (!gd_metal_reduce_view_range_valid(src) || !gd_metal_reduce_view_range_valid(dst) ||
        src->count != 1U || dst->count > UINT32_MAX || !gd_reduce_dtype_supported(src->dtype) ||
        !gd_reduce_dtype_supported(dst->dtype)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dtype_pair_ok = src->dtype == dst->dtype ||
                    (src->dtype == (uint32_t)GD_DTYPE_F32 && dst->dtype == (uint32_t)GD_DTYPE_F16);
    if (!dtype_pair_ok) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    src_elem_size = gd_reduce_dtype_size(src->dtype);
    dst_elem_size = gd_reduce_dtype_size(dst->dtype);
    if (src_elem_size == 0U || dst_elem_size == 0U || src->offset % src_elem_size != 0U ||
        dst->offset % dst_elem_size != 0U) {
        return GD_ERR_INVALID_ARGUMENT;
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
    id<MTLComputePipelineState> pso;
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
    pso = gd_reduce_contiguous_pso_for(backend, src->dtype, dst->dtype);
    if (pso == nil) {
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
    memset(&args, 0, sizeof(args));
    args.src_offset = (uint64_t)src->offset;
    args.dst_offset = (uint64_t)dst->offset;
    args.src_count = (uint64_t)src->count;
    args.dst_count = (uint64_t)dst->count;
    args.dtype = src->dtype;
    args.simdgroups_per_output = gd_reduce_contiguous_simdgroups_for_stage(args.src_count, args.dst_count);
    args.scale = scale;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)dst->count, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups_per_output, 1U);
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
    id<MTLComputePipelineState> pso;
    gd_metal_reduce_axis_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool last_axis;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_axis_validate(src, dst, axis);
    if (st != GD_OK) {
        return st;
    }
    gd_reduce_axis_fill_args(&args, src, dst, axis, scale);
    last_axis = args.inner_count == 1U;
    pso = gd_reduce_axis_pso_for(backend, src->dtype, last_axis);
    if (pso == nil) {
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
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)dst->count, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups_per_output, 1U);
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
    id<MTLComputePipelineState> pso;
    gd_metal_reduce_axis_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    bool last_axis;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_broadcast_axis_validate(src, dst, axis);
    if (st != GD_OK) {
        return st;
    }
    gd_broadcast_axis_fill_args(&args, src, dst, axis, scale);
    last_axis = args.inner_count == 1U;
    if (last_axis) {
        size_t elem_size = gd_reduce_dtype_size(dst->dtype);
        last_axis = elem_size != 0U &&
                    args.reduce_count % GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH == 0U &&
                    dst->offset % (elem_size * GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) == 0U;
    }
    pso = gd_broadcast_axis_pso_for(backend, src->dtype, last_axis);
    if (pso == nil) {
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
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    if (last_axis) {
        size_t row_vectors = ((size_t)args.reduce_count + GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH - 1U) /
                             GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH;
        grid = MTLSizeMake((NSUInteger)row_vectors, (NSUInteger)args.outer_count, 1U);
        threads = MTLSizeMake(gd_reduce_threads_for_count(row_vectors), 1U, 1U);
    } else {
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
    id<MTLComputePipelineState> pso;
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
    pso = gd_broadcast_to_pso_for(backend, src->dtype);
    if (pso == nil) {
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
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)dst->count, 1U, 1U);
    threads = MTLSizeMake(gd_reduce_threads_for_count(dst->count), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_broadcast_scalar(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_broadcast_scalar_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool vectorized;
    size_t elem_size;
    size_t grid_count;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_broadcast_scalar_validate(src, dst);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_broadcast_scalar_pso_for(backend, src->dtype, dst->dtype);
    if (pso == nil) {
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
    elem_size = gd_reduce_dtype_size(dst->dtype);
    vectorized = elem_size != 0U &&
                 dst->offset % (elem_size * GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) == 0U;
    grid_count = vectorized ? (dst->count / GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) +
                                  (dst->count % GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) :
                              dst->count;
    memset(&args, 0, sizeof(args));
    args.src_offset = (uint64_t)src->offset;
    args.dst_offset = (uint64_t)dst->offset;
    args.dst_count = (uint64_t)dst->count;
    args.scale = scale;
    args.vector_width = vectorized ? GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH : 1U;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_reduce_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_reduce_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)grid_count, 1U, 1U);
    threads = MTLSizeMake(gd_reduce_threads_for_count(grid_count), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
