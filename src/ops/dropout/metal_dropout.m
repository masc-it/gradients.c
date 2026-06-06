#include "../../backends/metal/metal_backend_internal.h"
#include "metal_dropout_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

#define GD_METAL_DROPOUT_MAX_THREADS_PER_GROUP 256U

static bool gd_dropout_dtype_supported(uint32_t dtype)
{
    return dtype == (uint32_t)GD_DTYPE_F16 || dtype == (uint32_t)GD_DTYPE_F32;
}

static size_t gd_dropout_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static id<MTLComputePipelineState> gd_dropout_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_forward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_forward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_dropout_backward_recompute_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_backward_recompute_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_backward_recompute_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_dropout_backward_mask_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_backward_mask_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->dropout_backward_mask_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_dropout_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_dropout_byte_range_valid(const gd_backend_buffer *buffer,
                                        size_t offset,
                                        size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_dropout_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_dropout_same_shape(const gd_backend_tensor_view *a,
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

static bool gd_dropout_probability_valid(float p)
{
    return p >= 0.0f && p < 1.0f;
}

static gd_status gd_dropout_validate_pair(const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *y,
                                          size_t *out_elem_size,
                                          size_t *out_nbytes)
{
    size_t elem_size;
    size_t nbytes;
    if (x == NULL || y == NULL || out_elem_size == NULL || out_nbytes == NULL ||
        x->buffer == NULL || y->buffer == NULL || x->count == 0U || x->count != y->count ||
        x->count > UINT32_MAX || x->dtype != y->dtype || !gd_dropout_dtype_supported(x->dtype) ||
        !gd_dropout_same_shape(x, y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_dropout_dtype_size(x->dtype);
    if (!gd_dropout_count_bytes(x->count, elem_size, &nbytes) ||
        !gd_dropout_byte_range_valid(x->buffer, x->offset, nbytes) ||
        !gd_dropout_byte_range_valid(y->buffer, y->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_elem_size = elem_size;
    *out_nbytes = nbytes;
    return GD_OK;
}

static gd_status gd_dropout_validate_mask(const gd_backend_tensor_view *ref,
                                          const gd_backend_tensor_view *mask)
{
    if (ref == NULL || mask == NULL || mask->buffer == NULL || mask->count != ref->count ||
        mask->dtype != (uint32_t)GD_DTYPE_U8 || !gd_dropout_same_shape(ref, mask) ||
        !gd_dropout_byte_range_valid(mask->buffer, mask->offset, ref->count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_dropout_fill_args(const gd_backend_tensor_view *src,
                                 const gd_backend_tensor_view *dst,
                                 const gd_backend_tensor_view *mask,
                                 float p,
                                 float scale,
                                 uint64_t seed,
                                 gd_metal_dropout_args *args)
{
    memset(args, 0, sizeof(*args));
    args->x_offset = (uint64_t)src->offset;
    args->y_offset = (uint64_t)dst->offset;
    args->mask_offset = mask != NULL ? (uint64_t)mask->offset : 0U;
    args->count = (uint64_t)src->count;
    args->seed = seed;
    args->p = p;
    args->scale = scale;
    args->dtype = src->dtype;
}

static MTLSize gd_dropout_grid(size_t count)
{
    const size_t thread_count = ((count - 1U) / GD_METAL_DROPOUT_ELEMENTS_PER_THREAD) + 1U;
    return MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
}

static MTLSize gd_dropout_threads(size_t count)
{
    size_t thread_count = ((count - 1U) / GD_METAL_DROPOUT_ELEMENTS_PER_THREAD) + 1U;
    if (thread_count > GD_METAL_DROPOUT_MAX_THREADS_PER_GROUP) {
        thread_count = GD_METAL_DROPOUT_MAX_THREADS_PER_GROUP;
    }
    return MTLSizeMake((NSUInteger)thread_count, 1U, 1U);
}

gd_status gd_backend_dropout_forward(gd_backend *backend,
                                     const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *y,
                                     const gd_backend_tensor_view *mask,
                                     float p,
                                     uint64_t seed)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_dropout_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    float scale;
    gd_status st;
    if (backend == NULL || !gd_dropout_probability_valid(p) || p == 0.0f) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_dropout_validate_pair(x, y, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    st = gd_dropout_validate_mask(x, mask);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_dropout_forward_pso(backend, x->dtype);
    if (pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    scale = 1.0f / (1.0f - p);
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_dropout_fill_args(x, y, mask, p, scale, seed, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_dropout_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_dropout_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_dropout_buffer(mask->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    [encoder dispatchThreads:gd_dropout_grid(x->count) threadsPerThreadgroup:gd_dropout_threads(x->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_dropout_backward(gd_backend *backend,
                                      const gd_backend_tensor_view *grad_out,
                                      const gd_backend_tensor_view *grad_x,
                                      float p,
                                      uint64_t seed)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_dropout_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    float scale;
    gd_status st;
    if (backend == NULL || !gd_dropout_probability_valid(p) || p == 0.0f) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_dropout_validate_pair(grad_out, grad_x, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    pso = gd_dropout_backward_recompute_pso(backend, grad_out->dtype);
    if (pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    scale = 1.0f / (1.0f - p);
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_dropout_fill_args(grad_out, grad_x, NULL, p, scale, seed, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_dropout_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_dropout_buffer(grad_x->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    [encoder dispatchThreads:gd_dropout_grid(grad_out->count)
       threadsPerThreadgroup:gd_dropout_threads(grad_out->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_dropout_backward_mask(gd_backend *backend,
                                           const gd_backend_tensor_view *mask,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_x,
                                           float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_dropout_args args;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (backend == NULL || !(scale > 0.0f) || scale != scale) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_dropout_validate_pair(grad_out, grad_x, &elem_size, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)elem_size;
    (void)nbytes;
    st = gd_dropout_validate_mask(grad_out, mask);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_dropout_backward_mask_pso(backend, grad_out->dtype);
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
    gd_dropout_fill_args(grad_out, grad_x, mask, 0.0f, scale, 0U, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_dropout_buffer(mask->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_dropout_buffer(grad_out->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_dropout_buffer(grad_x->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    [encoder dispatchThreads:gd_dropout_grid(grad_out->count)
       threadsPerThreadgroup:gd_dropout_threads(grad_out->count)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
