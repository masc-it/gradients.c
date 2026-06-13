#ifndef GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_COMMON_H
#define GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_COMMON_H

#import <Metal/Metal.h>

#include <gradients/tensor.h>

#include "../../../backends/metal/metal_backend_internal.h"

#include <stdint.h>
#include <string.h>

#define GD_METAL_PAIRWISE_LOSS_MAX_ARGS_SIZE 128U

typedef id<MTLComputePipelineState> (*gd_metal_pairwise_loss_pso_fn)(gd_backend *backend,
                                                                     uint32_t dtype);
typedef gd_status (*gd_metal_pairwise_loss_attr_validate_fn)(const void *attrs);

typedef void (*gd_metal_pairwise_loss_forward_args_fn)(
    void *args,
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *out,
    uint64_t chunk_size,
    float scale,
    uint32_t simdgroups,
    const void *attrs);

typedef void (*gd_metal_pairwise_loss_backward_args_fn)(
    void *args,
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *grad_out,
    const gd_backend_tensor_view *grad_x,
    const gd_backend_tensor_view *grad_y,
    float scale,
    const void *attrs);

typedef struct gd_metal_pairwise_loss_config {
    size_t args_size;
    uint32_t max_simdgroups;
    gd_metal_pairwise_loss_pso_fn forward_pso;
    gd_metal_pairwise_loss_pso_fn backward_pso;
    gd_metal_pairwise_loss_attr_validate_fn validate_attrs;
    gd_metal_pairwise_loss_forward_args_fn init_forward_args;
    gd_metal_pairwise_loss_backward_args_fn init_backward_args;
} gd_metal_pairwise_loss_config;

typedef union gd_metal_pairwise_loss_args_storage {
    unsigned char bytes[GD_METAL_PAIRWISE_LOSS_MAX_ARGS_SIZE];
    uint64_t align_u64;
    long double align_long_double;
} gd_metal_pairwise_loss_args_storage;

static inline id<MTLBuffer> gd_metal_pairwise_loss_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static inline size_t gd_metal_pairwise_loss_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static inline bool gd_metal_pairwise_loss_byte_range_valid(const gd_backend_buffer *buffer,
                                                           size_t offset,
                                                           size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static inline bool gd_metal_pairwise_loss_count_bytes(size_t count,
                                                      uint32_t dtype,
                                                      size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    elem_size = gd_metal_pairwise_loss_dtype_size(dtype);
    if (elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static inline bool gd_metal_pairwise_loss_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    size_t elem_size;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_metal_pairwise_loss_count_bytes(view->count, view->dtype, &nbytes)) {
        return false;
    }
    elem_size = gd_metal_pairwise_loss_dtype_size(view->dtype);
    return elem_size != 0U && view->offset % elem_size == 0U &&
           gd_metal_pairwise_loss_byte_range_valid(view->buffer, view->offset, nbytes);
}

static inline bool gd_metal_pairwise_loss_contiguous_view(const gd_backend_tensor_view *view)
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

static inline bool gd_metal_pairwise_loss_same_shape(const gd_backend_tensor_view *x,
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

static inline uint32_t gd_metal_pairwise_loss_simdgroups_for_chunk(uint64_t count,
                                                                   uint32_t max_simdgroups)
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
    return max_simdgroups;
}

static inline bool gd_metal_pairwise_loss_config_valid(const gd_metal_pairwise_loss_config *config)
{
    return config != NULL && config->args_size > 0U &&
           config->args_size <= GD_METAL_PAIRWISE_LOSS_MAX_ARGS_SIZE &&
           config->max_simdgroups > 0U && config->forward_pso != NULL &&
           config->backward_pso != NULL && config->init_forward_args != NULL &&
           config->init_backward_args != NULL;
}

static inline gd_status gd_metal_pairwise_loss_forward_validate(
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *out,
    uint64_t chunk_size,
    float scale)
{
    uint64_t expected_chunks;
    if (!(scale == scale) || chunk_size == 0U || chunk_size > UINT32_MAX ||
        !gd_metal_pairwise_loss_view_range_valid(x) ||
        !gd_metal_pairwise_loss_view_range_valid(y) ||
        !gd_metal_pairwise_loss_view_range_valid(out) || x->dtype != y->dtype ||
        (x->dtype != (uint32_t)GD_DTYPE_F16 && x->dtype != (uint32_t)GD_DTYPE_F32) ||
        out->dtype != (uint32_t)GD_DTYPE_F32 || x->count > UINT32_MAX ||
        !gd_metal_pairwise_loss_same_shape(x, y) ||
        !gd_metal_pairwise_loss_contiguous_view(x) ||
        !gd_metal_pairwise_loss_contiguous_view(y) ||
        !gd_metal_pairwise_loss_contiguous_view(out)) {
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

static inline bool gd_metal_pairwise_loss_grad_like_input(const gd_backend_tensor_view *x,
                                                          const gd_backend_tensor_view *grad)
{
    return grad != NULL && gd_metal_pairwise_loss_view_range_valid(grad) &&
           grad->dtype == x->dtype && gd_metal_pairwise_loss_same_shape(x, grad) &&
           gd_metal_pairwise_loss_contiguous_view(grad);
}

static inline gd_status gd_metal_pairwise_loss_backward_validate(
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *grad_out,
    const gd_backend_tensor_view *grad_x,
    const gd_backend_tensor_view *grad_y,
    float scale)
{
    if (!(scale == scale) || !gd_metal_pairwise_loss_view_range_valid(x) ||
        !gd_metal_pairwise_loss_view_range_valid(y) ||
        !gd_metal_pairwise_loss_view_range_valid(grad_out) || x->dtype != y->dtype ||
        (x->dtype != (uint32_t)GD_DTYPE_F16 && x->dtype != (uint32_t)GD_DTYPE_F32) ||
        grad_out->dtype != (uint32_t)GD_DTYPE_F32 || grad_out->rank != 0U ||
        grad_out->count != 1U || x->count > UINT32_MAX ||
        !gd_metal_pairwise_loss_same_shape(x, y) ||
        !gd_metal_pairwise_loss_contiguous_view(x) ||
        !gd_metal_pairwise_loss_contiguous_view(y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x == NULL && grad_y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_x != NULL && !gd_metal_pairwise_loss_grad_like_input(x, grad_x)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_y != NULL && !gd_metal_pairwise_loss_grad_like_input(x, grad_y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static inline gd_status gd_metal_pairwise_loss_forward(gd_backend *backend,
                                                       const gd_backend_tensor_view *x,
                                                       const gd_backend_tensor_view *y,
                                                       const gd_backend_tensor_view *out,
                                                       uint64_t chunk_size,
                                                       float scale,
                                                       const void *attrs,
                                                       const gd_metal_pairwise_loss_config *config)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_pairwise_loss_args_storage args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    uint64_t logical_chunk;
    uint32_t simdgroups;
    gd_status st;
    if (backend == NULL || !gd_metal_pairwise_loss_config_valid(config)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (config->validate_attrs != NULL) {
        st = config->validate_attrs(attrs);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_metal_pairwise_loss_forward_validate(x, y, out, chunk_size, scale);
    if (st != GD_OK) {
        return st;
    }
    pso = config->forward_pso(backend, x->dtype);
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
    simdgroups = gd_metal_pairwise_loss_simdgroups_for_chunk(logical_chunk,
                                                             config->max_simdgroups);
    memset(&args, 0, sizeof(args));
    config->init_forward_args(args.bytes, x, y, out, chunk_size, scale, simdgroups, attrs);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(out->buffer) offset:0U atIndex:2U];
    [encoder setBytes:args.bytes length:config->args_size atIndex:3U];
    grid = MTLSizeMake((NSUInteger)out->count, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static inline gd_status gd_metal_pairwise_loss_backward(gd_backend *backend,
                                                        const gd_backend_tensor_view *x,
                                                        const gd_backend_tensor_view *y,
                                                        const gd_backend_tensor_view *grad_out,
                                                        const gd_backend_tensor_view *grad_x,
                                                        const gd_backend_tensor_view *grad_y,
                                                        float scale,
                                                        const void *attrs,
                                                        const gd_metal_pairwise_loss_config *config)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_pairwise_loss_args_storage args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_metal_pairwise_loss_config_valid(config)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (config->validate_attrs != NULL) {
        st = config->validate_attrs(attrs);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_metal_pairwise_loss_backward_validate(x, y, grad_out, grad_x, grad_y, scale);
    if (st != GD_OK) {
        return st;
    }
    pso = config->backward_pso(backend, x->dtype);
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
    config->init_backward_args(args.bytes, x, y, grad_out, grad_x, grad_y, scale, attrs);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(y->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(grad_x != NULL ? grad_x->buffer : x->buffer)
               offset:0U
              atIndex:3U];
    [encoder setBuffer:gd_metal_pairwise_loss_buffer(grad_y != NULL ? grad_y->buffer : y->buffer)
               offset:0U
              atIndex:4U];
    [encoder setBytes:args.bytes length:config->args_size atIndex:5U];
    grid = MTLSizeMake((NSUInteger)x->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(x->count < 256U ? x->count : 256U), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

#endif /* GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_COMMON_H */
