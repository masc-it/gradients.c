#ifndef GD_OPS_SHARED_BINARY_METAL_COMMON_H
#define GD_OPS_SHARED_BINARY_METAL_COMMON_H

#include "../../../backends/metal/metal_backend_internal.h"
#include "metal_binary_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

#define GD_METAL_BINARY_MAX_THREADS_PER_GROUP 256U

static inline id<MTLBuffer> gd_metal_binary_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static inline bool gd_metal_binary_byte_range_valid(const gd_backend_buffer *buffer,
                                                    size_t offset,
                                                    size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static inline bool gd_metal_binary_count_bytes(size_t count,
                                               uint32_t dtype,
                                               size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        elem_size = 2U;
    } else if (dtype == (uint32_t)GD_DTYPE_F32) {
        elem_size = 4U;
    } else {
        return false;
    }
    if (count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static inline bool gd_metal_binary_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_metal_binary_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_metal_binary_byte_range_valid(view->buffer, view->offset, nbytes);
}

static inline gd_status gd_metal_binary_validate_direct_views(const gd_backend_tensor_view *x,
                                                              const gd_backend_tensor_view *y,
                                                              const gd_backend_tensor_view *out)
{
    if (!gd_metal_binary_view_range_valid(x) || !gd_metal_binary_view_range_valid(y) ||
        !gd_metal_binary_view_range_valid(out) || x->count != y->count || x->count != out->count ||
        x->dtype != y->dtype || x->dtype != out->dtype || x->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static inline bool gd_metal_binary_broadcast_dims(const gd_backend_tensor_view *arg,
                                                  const gd_backend_tensor_view *out,
                                                  uint32_t dim,
                                                  uint64_t *out_stride)
{
    uint32_t prefix;
    int64_t arg_dim;
    int64_t out_dim;
    int64_t stride;
    if (arg == NULL || out == NULL || out_stride == NULL || dim >= out->rank ||
        out->rank > GD_MAX_DIMS || arg->rank > out->rank) {
        return false;
    }
    prefix = out->rank - arg->rank;
    out_dim = out->shape[dim];
    if (out_dim <= 0) {
        return false;
    }
    if (dim < prefix) {
        *out_stride = 0U;
        return true;
    }
    arg_dim = arg->shape[dim - prefix];
    stride = arg->strides[dim - prefix];
    if (arg_dim == out_dim && stride >= 0) {
        *out_stride = (uint64_t)stride;
        return true;
    }
    if (arg_dim == 1) {
        *out_stride = 0U;
        return true;
    }
    return false;
}

static inline gd_status gd_metal_binary_validate_broadcast_views(const gd_backend_tensor_view *x,
                                                                 const gd_backend_tensor_view *y,
                                                                 const gd_backend_tensor_view *out)
{
    uint32_t dim;
    uint64_t stride;
    if (!gd_metal_binary_view_range_valid(x) || !gd_metal_binary_view_range_valid(y) ||
        !gd_metal_binary_view_range_valid(out) || x->dtype != y->dtype || x->dtype != out->dtype ||
        out->rank > GD_MAX_DIMS || x->rank > out->rank || y->rank > out->rank ||
        out->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (dim = 0U; dim < out->rank; ++dim) {
        uint32_t x_prefix = out->rank - x->rank;
        uint32_t y_prefix = out->rank - y->rank;
        int64_t x_dim = dim < x_prefix ? 1 : x->shape[dim - x_prefix];
        int64_t y_dim = dim < y_prefix ? 1 : y->shape[dim - y_prefix];
        int64_t expected = x_dim > y_dim ? x_dim : y_dim;
        if (out->shape[dim] != expected ||
            !gd_metal_binary_broadcast_dims(x, out, dim, &stride) ||
            !gd_metal_binary_broadcast_dims(y, out, dim, &stride)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static inline void gd_metal_binary_fill_bcast_args(gd_metal_binary_bcast_args *args,
                                                   const gd_backend_tensor_view *x,
                                                   const gd_backend_tensor_view *y,
                                                   const gd_backend_tensor_view *out)
{
    uint32_t dim;
    memset(args, 0, sizeof(*args));
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->out_offset = (uint64_t)out->offset;
    args->count = (uint64_t)out->count;
    args->dtype = out->dtype;
    args->rank = out->rank;
    for (dim = 0U; dim < out->rank; ++dim) {
        uint64_t stride = 0U;
        args->out_shape[dim] = (uint64_t)out->shape[dim];
        (void)gd_metal_binary_broadcast_dims(x, out, dim, &stride);
        args->x_strides[dim] = stride;
        (void)gd_metal_binary_broadcast_dims(y, out, dim, &stride);
        args->y_strides[dim] = stride;
    }
}

static inline gd_status gd_metal_binary_dispatch(gd_backend *backend,
                                                 const gd_backend_tensor_view *x,
                                                 const gd_backend_tensor_view *y,
                                                 const gd_backend_tensor_view *out,
                                                 id<MTLComputePipelineState> direct_pso,
                                                 id<MTLComputePipelineState> bcast_pso,
                                                 id<MTLComputePipelineState> row_bcast_pso)
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
    if (backend == NULL || direct_pso == nil || bcast_pso == nil || row_bcast_pso == nil) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_binary_validate_broadcast_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    direct = x->count == out->count && y->count == out->count;
    row_broadcast = !direct && out->rank == 2U;
    if (direct) {
        st = gd_metal_binary_validate_direct_views(x, y, out);
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
        [encoder setComputePipelineState:direct_pso];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    } else {
        gd_metal_binary_fill_bcast_args(&bcast_args, x, y, out);
        [encoder setComputePipelineState:row_broadcast ? row_bcast_pso : bcast_pso];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&bcast_args length:sizeof(bcast_args) atIndex:3U];
    }
    if (row_broadcast) {
        grid = MTLSizeMake((NSUInteger)out->shape[1], (NSUInteger)out->shape[0], 1U);
        threads = MTLSizeMake((NSUInteger)(out->shape[1] < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ?
                                           out->shape[1] : GD_METAL_BINARY_MAX_THREADS_PER_GROUP),
                              1U,
                              1U);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        grid = MTLSizeMake((NSUInteger)out->count, 1U, 1U);
        threads = MTLSizeMake((NSUInteger)(out->count < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ?
                                           out->count : GD_METAL_BINARY_MAX_THREADS_PER_GROUP),
                              1U,
                              1U);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

#endif /* GD_OPS_SHARED_BINARY_METAL_COMMON_H */
