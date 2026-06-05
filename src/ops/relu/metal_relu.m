#include "../../backends/metal/metal_backend_internal.h"
#include "metal_relu_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

#define GD_METAL_RELU_MAX_THREADS_PER_GROUP 256U

static id<MTLComputePipelineState> gd_relu_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->unary_pso[GD_OP_RELU];
}

static id<MTLComputePipelineState> gd_relu_backward_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->unary_backward_pso[GD_OP_RELU];
}

static id<MTLBuffer> gd_relu_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_relu_byte_range_valid(const gd_backend_buffer *buffer,
                                     size_t offset,
                                     size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_relu_count_bytes(size_t count, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || count > SIZE_MAX / 2U) {
        return false;
    }
    *out_nbytes = count * 2U;
    return true;
}

static gd_status gd_relu_validate_views(const gd_backend_tensor_view *x,
                                        const gd_backend_tensor_view *y,
                                        size_t *out_nbytes)
{
    size_t nbytes;
    if (x == NULL || y == NULL || out_nbytes == NULL || x->buffer == NULL || y->buffer == NULL ||
        x->count == 0U || x->count != y->count || x->dtype != (uint32_t)GD_DTYPE_F16 ||
        y->dtype != (uint32_t)GD_DTYPE_F16 || x->count > UINT32_MAX ||
        !gd_relu_count_bytes(x->count, &nbytes) ||
        !gd_relu_byte_range_valid(x->buffer, x->offset, nbytes) ||
        !gd_relu_byte_range_valid(y->buffer, y->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_nbytes = nbytes;
    return GD_OK;
}

gd_status gd_backend_relu(gd_backend *backend,
                          const gd_backend_tensor_view *x,
                          const gd_backend_tensor_view *y)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_relu_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t nbytes;
    size_t thread_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_relu_validate_views(x, y, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)nbytes;
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
    args.count = (uint64_t)x->count;
    thread_count = (x->count + GD_METAL_RELU_ELEMENTS_PER_THREAD - 1U) /
                   GD_METAL_RELU_ELEMENTS_PER_THREAD;
    [encoder setComputePipelineState:gd_relu_pso(backend)];
    [encoder setBuffer:gd_relu_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_relu_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(thread_count < GD_METAL_RELU_MAX_THREADS_PER_GROUP ?
                                       thread_count : GD_METAL_RELU_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_relu_backward(gd_backend *backend,
                                   const gd_backend_tensor_view *x,
                                   const gd_backend_tensor_view *grad_out,
                                   const gd_backend_tensor_view *grad_x)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_relu_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t nbytes;
    size_t thread_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_relu_validate_views(x, grad_out, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL || grad_x->buffer == NULL || grad_x->count != x->count ||
        grad_x->dtype != x->dtype || !gd_relu_byte_range_valid(grad_x->buffer, grad_x->offset, nbytes)) {
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
    args.grad_offset = (uint64_t)grad_out->offset;
    args.y_offset = (uint64_t)grad_x->offset;
    args.count = (uint64_t)x->count;
    thread_count = (x->count + GD_METAL_RELU_ELEMENTS_PER_THREAD - 1U) /
                   GD_METAL_RELU_ELEMENTS_PER_THREAD;
    [encoder setComputePipelineState:gd_relu_backward_pso(backend)];
    [encoder setBuffer:gd_relu_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_relu_buffer(grad_out->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_relu_buffer(grad_x->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(thread_count < GD_METAL_RELU_MAX_THREADS_PER_GROUP ?
                                       thread_count : GD_METAL_RELU_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
