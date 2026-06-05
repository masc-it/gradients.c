#include "../_shared/binary/metal_binary_common.h"
#include "metal_mul_types.h"

#define GD_MUL_F16_VECTOR_WIDTH ((NSUInteger)GD_METAL_MUL_F16_VECTOR_WIDTH)

static id<MTLComputePipelineState> gd_mul_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_pso[GD_OP_MUL];
}

static id<MTLComputePipelineState> gd_mul_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_bcast_pso[GD_OP_MUL];
}

static id<MTLComputePipelineState> gd_mul_row_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_row_bcast_pso[GD_OP_MUL];
}

static id<MTLComputePipelineState> gd_mul_backward_direct_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->mul_backward_direct_pso;
}

static id<MTLComputePipelineState> gd_mul_reduce_suffix_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->mul_reduce_suffix_pso;
}

static id<MTLComputePipelineState> gd_mul_reduce_suffix_small_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->mul_reduce_suffix_small_pso;
}

static NSUInteger gd_mul_div_up_nsu(size_t value, NSUInteger divisor)
{
    return (NSUInteger)((value + (size_t)divisor - 1U) / (size_t)divisor);
}

static bool gd_mul_view_is_f16(const gd_backend_tensor_view *view)
{
    return view != NULL && view->dtype == (uint32_t)GD_DTYPE_F16;
}

static bool gd_mul_shapes_equal(const gd_backend_tensor_view *x, const gd_backend_tensor_view *y)
{
    uint32_t dim;
    if (x == NULL || y == NULL || x->rank != y->rank) {
        return false;
    }
    for (dim = 0U; dim < x->rank; ++dim) {
        if (x->shape[dim] != y->shape[dim]) {
            return false;
        }
    }
    return true;
}

static bool gd_mul_reduce_suffix_compatible(const gd_backend_tensor_view *src,
                                            const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    bool suffix_started = false;
    if (src == NULL || dst == NULL || dst->rank > src->rank || dst->count == 0U ||
        src->count <= dst->count || src->count % dst->count != 0U) {
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

static gd_status gd_mul_validate_direct_f16(const gd_backend_tensor_view *x,
                                            const gd_backend_tensor_view *y,
                                            const gd_backend_tensor_view *out)
{
    gd_status st = gd_metal_binary_validate_direct_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    return gd_mul_view_is_f16(x) && gd_mul_view_is_f16(y) && gd_mul_view_is_f16(out) ? GD_OK : GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_mul(gd_backend *backend,
                         const gd_backend_tensor_view *x,
                         const gd_backend_tensor_view *y,
                         const gd_backend_tensor_view *out)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_binary_args args;
    gd_metal_binary_bcast_args bcast_args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool direct;
    bool row_broadcast;
    gd_status st;
    if (backend == NULL || gd_mul_pso(backend) == nil || gd_mul_bcast_pso(backend) == nil ||
        gd_mul_row_bcast_pso(backend) == nil) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_binary_validate_broadcast_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_mul_view_is_f16(x) || !gd_mul_view_is_f16(y) || !gd_mul_view_is_f16(out)) {
        return GD_ERR_UNSUPPORTED;
    }
    direct = x->count == out->count && y->count == out->count;
    row_broadcast = !direct && out->rank == 2U;
    if (direct) {
        st = gd_mul_validate_direct_f16(x, y, out);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    if (direct) {
        memset(&args, 0, sizeof(args));
        args.x_offset = (uint64_t)x->offset;
        args.y_offset = (uint64_t)y->offset;
        args.out_offset = (uint64_t)out->offset;
        args.count = (uint64_t)out->count;
        args.dtype = out->dtype;
        [encoder setComputePipelineState:gd_mul_pso(backend)];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&args length:sizeof(args) atIndex:3U];
        grid = MTLSizeMake(gd_mul_div_up_nsu(out->count, GD_MUL_F16_VECTOR_WIDTH), 1U, 1U);
        threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                   GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                              1U,
                              1U);
    } else {
        gd_metal_binary_fill_bcast_args(&bcast_args, x, y, out);
        [encoder setComputePipelineState:row_broadcast ? gd_mul_row_bcast_pso(backend) : gd_mul_bcast_pso(backend)];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&bcast_args length:sizeof(bcast_args) atIndex:3U];
        if (row_broadcast) {
            grid = MTLSizeMake(gd_mul_div_up_nsu((size_t)out->shape[1], GD_MUL_F16_VECTOR_WIDTH),
                               (NSUInteger)out->shape[0],
                               1U);
            threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                       GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                                  1U,
                                  1U);
        } else {
            grid = MTLSizeMake(gd_mul_div_up_nsu(out->count, GD_MUL_F16_VECTOR_WIDTH), 1U, 1U);
            threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                       GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                                  1U,
                                  1U);
        }
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_mul_backward_direct(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_x,
                                         const gd_backend_tensor_view *grad_y)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_mul_backward_direct_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || gd_mul_backward_direct_pso(backend) == nil || grad_x == NULL || grad_y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (gd_mul_validate_direct_f16(x, y, grad_out) != GD_OK ||
        gd_mul_validate_direct_f16(x, grad_out, grad_x) != GD_OK ||
        gd_mul_validate_direct_f16(y, grad_out, grad_y) != GD_OK ||
        !gd_mul_shapes_equal(x, grad_out) || !gd_mul_shapes_equal(x, grad_x) ||
        !gd_mul_shapes_equal(y, grad_y)) {
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
    memset(&args, 0, sizeof(args));
    args.x_offset = (uint64_t)x->offset;
    args.y_offset = (uint64_t)y->offset;
    args.grad_offset = (uint64_t)grad_out->offset;
    args.grad_x_offset = (uint64_t)grad_x->offset;
    args.grad_y_offset = (uint64_t)grad_y->offset;
    args.count = (uint64_t)grad_out->count;
    [encoder setComputePipelineState:gd_mul_backward_direct_pso(backend)];
    [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_binary_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_metal_binary_buffer(grad_x->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_metal_binary_buffer(grad_y->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake(gd_mul_div_up_nsu(grad_out->count, GD_MUL_F16_VECTOR_WIDTH), 1U, 1U);
    threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                               GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_mul_reduce_suffix(gd_backend *backend,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *other,
                                       const gd_backend_tensor_view *dst)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_mul_reduce_suffix_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || gd_mul_reduce_suffix_pso(backend) == nil ||
        gd_mul_reduce_suffix_small_pso(backend) == nil) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_metal_binary_view_range_valid(grad_out) || !gd_metal_binary_view_range_valid(other) ||
        !gd_metal_binary_view_range_valid(dst) || !gd_mul_view_is_f16(grad_out) ||
        !gd_mul_view_is_f16(other) || !gd_mul_view_is_f16(dst) || grad_out->count != other->count ||
        grad_out->count > UINT32_MAX || dst->count > UINT32_MAX || !gd_mul_shapes_equal(grad_out, other) ||
        !gd_mul_reduce_suffix_compatible(grad_out, dst)) {
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
    memset(&args, 0, sizeof(args));
    args.grad_offset = (uint64_t)grad_out->offset;
    args.other_offset = (uint64_t)other->offset;
    args.dst_offset = (uint64_t)dst->offset;
    args.src_count = (uint64_t)grad_out->count;
    args.dst_count = (uint64_t)dst->count;
    if (grad_out->count / dst->count <= 32U) {
        [encoder setComputePipelineState:gd_mul_reduce_suffix_small_pso(backend)];
    } else {
        [encoder setComputePipelineState:gd_mul_reduce_suffix_pso(backend)];
    }
    [encoder setBuffer:gd_metal_binary_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_binary_buffer(other->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_binary_buffer(dst->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    if (grad_out->count / dst->count <= 32U) {
        grid = MTLSizeMake(gd_mul_div_up_nsu(dst->count, GD_MUL_F16_VECTOR_WIDTH), 1U, 1U);
        threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                   GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                              1U,
                              1U);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        grid = MTLSizeMake((NSUInteger)((dst->count + GD_METAL_MUL_REDUCE_SUFFIX_SIMDGROUPS - 1U) /
                                        GD_METAL_MUL_REDUCE_SUFFIX_SIMDGROUPS),
                           1U,
                           1U);
        threads = MTLSizeMake(32U, GD_METAL_MUL_REDUCE_SUFFIX_SIMDGROUPS, 1U);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
