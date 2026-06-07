#include "../../backends/metal/metal_backend_internal.h"
#include "metal_powlu_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_powlu_forward_pso(gd_backend *backend)
{
    return backend != NULL ? (__bridge id<MTLComputePipelineState>)backend->powlu_forward_f16_pso : nil;
}

static id<MTLComputePipelineState> gd_powlu_backward_pso(gd_backend *backend)
{
    return backend != NULL ? (__bridge id<MTLComputePipelineState>)backend->powlu_backward_f16_pso : nil;
}

static id<MTLBuffer> gd_powlu_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_powlu_byte_range_valid(const gd_backend_buffer *buffer,
                                      size_t offset,
                                      size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_powlu_count_bytes(size_t count, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || count > SIZE_MAX / sizeof(uint16_t)) {
        return false;
    }
    *out_nbytes = count * sizeof(uint16_t);
    return true;
}

static bool gd_powlu_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    int32_t axis;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (axis = (int32_t)view->rank - 1; axis >= 0; --axis) {
        if (view->shape[axis] <= 0 || view->strides[axis] != expected ||
            expected > INT64_MAX / view->shape[axis]) {
            return false;
        }
        expected *= view->shape[axis];
    }
    return view->count == (size_t)expected;
}

static bool gd_powlu_same_shape(const gd_backend_tensor_view *x,
                                const gd_backend_tensor_view *y)
{
    uint32_t axis;
    if (x == NULL || y == NULL || x->rank != y->rank || x->count != y->count) {
        return false;
    }
    for (axis = 0U; axis < x->rank; ++axis) {
        if (x->shape[axis] != y->shape[axis]) {
            return false;
        }
    }
    return true;
}

static bool gd_powlu_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->dtype != (uint32_t)GD_DTYPE_F16 ||
        view->count == 0U || view->count > UINT32_MAX || view->offset % sizeof(uint16_t) != 0U ||
        !gd_powlu_count_bytes(view->count, &nbytes)) {
        return false;
    }
    return gd_powlu_byte_range_valid(view->buffer, view->offset, nbytes);
}

static gd_status gd_powlu_forward_validate(const gd_backend_tensor_view *x1,
                                           const gd_backend_tensor_view *x2,
                                           const gd_backend_tensor_view *out,
                                           float m)
{
    if (!(m == m) || m <= 0.0f || m >= 10.0f ||
        !gd_powlu_view_range_valid(x1) || !gd_powlu_view_range_valid(x2) ||
        !gd_powlu_view_range_valid(out) || !gd_powlu_same_shape(x1, x2) ||
        !gd_powlu_same_shape(x1, out) || !gd_powlu_contiguous_view(x1) ||
        !gd_powlu_contiguous_view(x2) || !gd_powlu_contiguous_view(out)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status gd_backend_powlu_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x1,
                                   const gd_backend_tensor_view *x2,
                                   const gd_backend_tensor_view *out,
                                   float m)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_powlu_fwd_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t thread_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_powlu_forward_validate(x1, x2, out, m);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_powlu_forward_pso(backend);
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
    args.x1_offset = (uint64_t)x1->offset;
    args.x2_offset = (uint64_t)x2->offset;
    args.out_offset = (uint64_t)out->offset;
    args.count = (uint64_t)x1->count;
    args.m = m;
    thread_count = (x1->count + GD_METAL_POWLU_ELEMENTS_PER_THREAD - 1U) /
                   GD_METAL_POWLU_ELEMENTS_PER_THREAD;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_powlu_buffer(x1->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_powlu_buffer(x2->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_powlu_buffer(out->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(thread_count < GD_METAL_POWLU_MAX_THREADS_PER_GROUP ?
                                       thread_count : GD_METAL_POWLU_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static bool gd_powlu_grad_like_input(const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *grad)
{
    return grad != NULL && gd_powlu_view_range_valid(grad) && gd_powlu_same_shape(x, grad) &&
           gd_powlu_contiguous_view(grad);
}

static gd_status gd_powlu_backward_validate(const gd_backend_tensor_view *x1,
                                            const gd_backend_tensor_view *x2,
                                            const gd_backend_tensor_view *grad_out,
                                            const gd_backend_tensor_view *grad_x1,
                                            const gd_backend_tensor_view *grad_x2,
                                            float m)
{
    if (!(m == m) || m <= 0.0f || m >= 10.0f || !gd_powlu_view_range_valid(x1) ||
        !gd_powlu_view_range_valid(x2) || !gd_powlu_view_range_valid(grad_out) ||
        !gd_powlu_same_shape(x1, x2) || !gd_powlu_same_shape(x1, grad_out) ||
        !gd_powlu_contiguous_view(x1) || !gd_powlu_contiguous_view(x2) ||
        !gd_powlu_contiguous_view(grad_out)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x1 == NULL && grad_x2 == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x1 != NULL && !gd_powlu_grad_like_input(x1, grad_x1)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x2 != NULL && !gd_powlu_grad_like_input(x1, grad_x2)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status gd_backend_powlu_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x1,
                                    const gd_backend_tensor_view *x2,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x1,
                                    const gd_backend_tensor_view *grad_x2,
                                    float m)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_powlu_bwd_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t thread_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_powlu_backward_validate(x1, x2, grad_out, grad_x1, grad_x2, m);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_powlu_backward_pso(backend);
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
    args.x1_offset = (uint64_t)x1->offset;
    args.x2_offset = (uint64_t)x2->offset;
    args.grad_offset = (uint64_t)grad_out->offset;
    args.dx1_offset = grad_x1 != NULL ? (uint64_t)grad_x1->offset : 0U;
    args.dx2_offset = grad_x2 != NULL ? (uint64_t)grad_x2->offset : 0U;
    args.count = (uint64_t)x1->count;
    args.m = m;
    args.write_x1 = grad_x1 != NULL ? 1U : 0U;
    args.write_x2 = grad_x2 != NULL ? 1U : 0U;
    thread_count = (x1->count + GD_METAL_POWLU_ELEMENTS_PER_THREAD - 1U) /
                   GD_METAL_POWLU_ELEMENTS_PER_THREAD;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_powlu_buffer(x1->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_powlu_buffer(x2->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_powlu_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_powlu_buffer(grad_x1 != NULL ? grad_x1->buffer : x1->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_powlu_buffer(grad_x2 != NULL ? grad_x2->buffer : x2->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(thread_count < GD_METAL_POWLU_MAX_THREADS_PER_GROUP ?
                                       thread_count : GD_METAL_POWLU_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
