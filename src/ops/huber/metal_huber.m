#include "../../backends/metal/metal_backend_internal.h"
#include "metal_huber_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_huber_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_forward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_forward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_huber_backward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_backward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_backward_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_huber_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static size_t gd_huber_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool gd_huber_byte_range_valid(const gd_backend_buffer *buffer,
                                      size_t offset,
                                      size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_huber_count_bytes(size_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    elem_size = gd_huber_dtype_size(dtype);
    if (elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_huber_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    size_t elem_size;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_huber_count_bytes(view->count, view->dtype, &nbytes)) {
        return false;
    }
    elem_size = gd_huber_dtype_size(view->dtype);
    return elem_size != 0U && view->offset % elem_size == 0U &&
           gd_huber_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_huber_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    int32_t d;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (d = (int32_t)view->rank - 1; d >= 0; --d) {
        if (view->shape[d] <= 0 || view->strides[d] != expected ||
            expected > INT64_MAX / view->shape[d]) {
            return false;
        }
        expected *= view->shape[d];
    }
    return view->count == (size_t)expected;
}

static bool gd_huber_same_shape(const gd_backend_tensor_view *x,
                                const gd_backend_tensor_view *y)
{
    uint32_t d;
    if (x == NULL || y == NULL || x->rank != y->rank || x->count != y->count) {
        return false;
    }
    for (d = 0U; d < x->rank; ++d) {
        if (x->shape[d] != y->shape[d]) {
            return false;
        }
    }
    return true;
}

static uint32_t gd_huber_simdgroups_for_chunk(uint64_t count)
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
    return GD_METAL_HUBER_MAX_SIMDGROUPS;
}

static gd_status gd_huber_forward_validate(const gd_backend_tensor_view *x,
                                           const gd_backend_tensor_view *y,
                                           const gd_backend_tensor_view *out,
                                           uint64_t chunk_size,
                                           float scale,
                                           float delta)
{
    uint64_t expected_chunks;
    if (!(scale == scale) || !(delta > 0.0f) || chunk_size == 0U || chunk_size > UINT32_MAX ||
        !gd_huber_view_range_valid(x) || !gd_huber_view_range_valid(y) ||
        !gd_huber_view_range_valid(out) || x->dtype != y->dtype ||
        (x->dtype != (uint32_t)GD_DTYPE_F16 && x->dtype != (uint32_t)GD_DTYPE_F32) ||
        out->dtype != (uint32_t)GD_DTYPE_F32 || x->count > UINT32_MAX ||
        !gd_huber_same_shape(x, y) || !gd_huber_contiguous_view(x) ||
        !gd_huber_contiguous_view(y) || !gd_huber_contiguous_view(out)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    expected_chunks = ((uint64_t)x->count + chunk_size - 1U) / chunk_size;
    if (expected_chunks == 0U || expected_chunks > UINT32_MAX ||
        out->count != (size_t)expected_chunks) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!(out->rank == 0U || out->rank == 1U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status gd_backend_huber_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x,
                                   const gd_backend_tensor_view *y,
                                   const gd_backend_tensor_view *out,
                                   uint64_t chunk_size,
                                   float scale,
                                   float delta)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_huber_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    uint64_t logical_chunk;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_huber_forward_validate(x, y, out, chunk_size, scale, delta);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_huber_forward_pso(backend, x->dtype);
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
    logical_chunk = ((uint64_t)x->count + (uint64_t)out->count - 1U) / (uint64_t)out->count;
    memset(&args, 0, sizeof(args));
    args.x_offset = (uint64_t)x->offset;
    args.y_offset = (uint64_t)y->offset;
    args.out_offset = (uint64_t)out->offset;
    args.count = (uint64_t)x->count;
    args.chunk_size = chunk_size;
    args.scale = scale;
    args.delta = delta;
    args.simdgroups = gd_huber_simdgroups_for_chunk(logical_chunk);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_huber_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_huber_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_huber_buffer(out->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)out->count, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static bool gd_huber_grad_like_input(const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *grad)
{
    return grad != NULL && gd_huber_view_range_valid(grad) && grad->dtype == x->dtype &&
           gd_huber_same_shape(x, grad) && gd_huber_contiguous_view(grad);
}

static gd_status gd_huber_backward_validate(const gd_backend_tensor_view *x,
                                            const gd_backend_tensor_view *y,
                                            const gd_backend_tensor_view *grad_out,
                                            const gd_backend_tensor_view *grad_x,
                                            const gd_backend_tensor_view *grad_y,
                                            float scale,
                                            float delta)
{
    if (!(scale == scale) || !(delta > 0.0f) || !gd_huber_view_range_valid(x) ||
        !gd_huber_view_range_valid(y) || !gd_huber_view_range_valid(grad_out) ||
        x->dtype != y->dtype ||
        (x->dtype != (uint32_t)GD_DTYPE_F16 && x->dtype != (uint32_t)GD_DTYPE_F32) ||
        grad_out->dtype != (uint32_t)GD_DTYPE_F32 || grad_out->rank != 0U ||
        grad_out->count != 1U || x->count > UINT32_MAX || !gd_huber_same_shape(x, y) ||
        !gd_huber_contiguous_view(x) || !gd_huber_contiguous_view(y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x == NULL && grad_y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x != NULL && !gd_huber_grad_like_input(x, grad_x)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_y != NULL && !gd_huber_grad_like_input(x, grad_y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status gd_backend_huber_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x,
                                    const gd_backend_tensor_view *y,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x,
                                    const gd_backend_tensor_view *grad_y,
                                    float scale,
                                    float delta)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_huber_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_huber_backward_validate(x, y, grad_out, grad_x, grad_y, scale, delta);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_huber_backward_pso(backend, x->dtype);
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
    args.x_offset = (uint64_t)x->offset;
    args.y_offset = (uint64_t)y->offset;
    args.grad_out_offset = (uint64_t)grad_out->offset;
    args.out_offset = grad_x != NULL ? (uint64_t)grad_x->offset : 0U;
    args.dy_offset = grad_y != NULL ? (uint64_t)grad_y->offset : 0U;
    args.count = (uint64_t)x->count;
    args.scale = scale;
    args.delta = delta;
    args.write_x = grad_x != NULL ? 1U : 0U;
    args.write_y = grad_y != NULL ? 1U : 0U;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_huber_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_huber_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_huber_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_huber_buffer(grad_x != NULL ? grad_x->buffer : x->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_huber_buffer(grad_y != NULL ? grad_y->buffer : y->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake((NSUInteger)x->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(x->count < 256U ? x->count : 256U), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
