#include "../../backends/metal/metal_backend_internal.h"
#include "metal_sigmoid_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

#define GD_METAL_SIGMOID_MAX_THREADS_PER_GROUP 256U

static bool gd_sigmoid_dtype_supported(uint32_t dtype)
{
    return dtype == (uint32_t)GD_DTYPE_F16 || dtype == (uint32_t)GD_DTYPE_F32;
}

static id<MTLComputePipelineState> gd_sigmoid_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->unary_pso[GD_OP_SIGMOID];
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->sigmoid_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_sigmoid_backward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->unary_backward_pso[GD_OP_SIGMOID];
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->sigmoid_backward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_sigmoid_backward_saved_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->sigmoid_backward_saved_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->sigmoid_backward_saved_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_sigmoid_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_sigmoid_byte_range_valid(const gd_backend_buffer *buffer,
                                        size_t offset,
                                        size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_sigmoid_count_bytes(size_t count,
                                   uint32_t dtype,
                                   size_t *out_elem_size,
                                   size_t *out_nbytes)
{
    size_t elem_size;
    if (out_elem_size == NULL || out_nbytes == NULL || count == 0U ||
        !gd_sigmoid_dtype_supported(dtype)) {
        return false;
    }
    elem_size = gd_dtype_size((gd_dtype)dtype);
    if (elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_elem_size = elem_size;
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_sigmoid_same_shape(const gd_backend_tensor_view *a,
                                  const gd_backend_tensor_view *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->rank != b->rank || a->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_sigmoid_validate_pair(const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *y,
                                          size_t *out_elem_size,
                                          size_t *out_nbytes)
{
    size_t elem_size;
    size_t nbytes;
    if (x == NULL || y == NULL || out_elem_size == NULL || out_nbytes == NULL ||
        x->buffer == NULL || y->buffer == NULL || x->count == 0U || x->count != y->count ||
        x->dtype != y->dtype || !gd_sigmoid_same_shape(x, y) ||
        !gd_sigmoid_count_bytes(x->count, x->dtype, &elem_size, &nbytes) ||
        !gd_sigmoid_byte_range_valid(x->buffer, x->offset, nbytes) ||
        !gd_sigmoid_byte_range_valid(y->buffer, y->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_elem_size = elem_size;
    *out_nbytes = nbytes;
    return GD_OK;
}

static gd_status gd_sigmoid_validate_backward_views(const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *grad_out,
                                                    const gd_backend_tensor_view *grad_x,
                                                    size_t *out_elem_size,
                                                    size_t *out_nbytes)
{
    gd_status st;
    size_t elem_size;
    size_t nbytes;
    st = gd_sigmoid_validate_pair(x, grad_out, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL || grad_x->buffer == NULL || grad_x->count != x->count ||
        grad_x->dtype != x->dtype || !gd_sigmoid_same_shape(x, grad_x) ||
        !gd_sigmoid_byte_range_valid(grad_x->buffer, grad_x->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_elem_size = elem_size;
    *out_nbytes = nbytes;
    return GD_OK;
}

static void gd_sigmoid_fill_forward_args(const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         gd_metal_sigmoid_args *args)
{
    memset(args, 0, sizeof(*args));
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->count = (uint64_t)x->count;
    args->dtype = x->dtype;
}

static void gd_sigmoid_fill_backward_args(const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *grad_out,
                                          const gd_backend_tensor_view *grad_x,
                                          gd_metal_sigmoid_args *args)
{
    memset(args, 0, sizeof(*args));
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)grad_x->offset;
    args->grad_offset = (uint64_t)grad_out->offset;
    args->count = (uint64_t)x->count;
    args->dtype = x->dtype;
}

static MTLSize gd_sigmoid_grid(size_t count)
{
    const size_t thread_count = ((count - 1U) / GD_METAL_SIGMOID_ELEMENTS_PER_THREAD) + 1U;
    return MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
}

static MTLSize gd_sigmoid_threads(size_t count)
{
    size_t thread_count = ((count - 1U) / GD_METAL_SIGMOID_ELEMENTS_PER_THREAD) + 1U;
    if (thread_count > GD_METAL_SIGMOID_MAX_THREADS_PER_GROUP) {
        thread_count = GD_METAL_SIGMOID_MAX_THREADS_PER_GROUP;
    }
    return MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
}

gd_status gd_backend_sigmoid(gd_backend *backend,
                             const gd_backend_tensor_view *x,
                             const gd_backend_tensor_view *y)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_sigmoid_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate_pair(x, y, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    pso = gd_sigmoid_forward_pso(backend, x->dtype);
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
    gd_sigmoid_fill_forward_args(x, y, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_sigmoid_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sigmoid_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    [encoder dispatchThreads:gd_sigmoid_grid(x->count) threadsPerThreadgroup:gd_sigmoid_threads(x->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_sigmoid_backward(gd_backend *backend,
                                      const gd_backend_tensor_view *x,
                                      const gd_backend_tensor_view *grad_out,
                                      const gd_backend_tensor_view *grad_x)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_sigmoid_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate_backward_views(x, grad_out, grad_x, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    pso = gd_sigmoid_backward_pso(backend, x->dtype);
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
    gd_sigmoid_fill_backward_args(x, grad_out, grad_x, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_sigmoid_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sigmoid_buffer(grad_out->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sigmoid_buffer(grad_x->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    [encoder dispatchThreads:gd_sigmoid_grid(x->count) threadsPerThreadgroup:gd_sigmoid_threads(x->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_sigmoid_backward_from_output(gd_backend *backend,
                                                  const gd_backend_tensor_view *sigmoid_out,
                                                  const gd_backend_tensor_view *grad_out,
                                                  const gd_backend_tensor_view *grad_x)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_sigmoid_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate_backward_views(sigmoid_out, grad_out, grad_x, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    pso = gd_sigmoid_backward_saved_pso(backend, sigmoid_out->dtype);
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
    gd_sigmoid_fill_backward_args(sigmoid_out, grad_out, grad_x, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_sigmoid_buffer(sigmoid_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_sigmoid_buffer(grad_out->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_sigmoid_buffer(grad_x->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    [encoder dispatchThreads:gd_sigmoid_grid(sigmoid_out->count)
       threadsPerThreadgroup:gd_sigmoid_threads(sigmoid_out->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
