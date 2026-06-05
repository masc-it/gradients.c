#include "../../backends/metal/metal_backend_internal.h"
#include "metal_cross_entropy_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_cross_entropy_loss_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->cross_entropy_loss_f16_pso;
}

static id<MTLComputePipelineState> gd_cross_entropy_loss_stats_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->cross_entropy_loss_stats_f16_pso;
}

static id<MTLComputePipelineState> gd_cross_entropy_backward_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->cross_entropy_backward_f16_pso;
}

static id<MTLComputePipelineState> gd_cross_entropy_backward_stats_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->cross_entropy_backward_stats_f16_pso;
}

static id<MTLBuffer> gd_cross_entropy_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_cross_entropy_byte_range_valid(const gd_backend_buffer *buffer,
                                              size_t offset,
                                              size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_cross_entropy_count_bytes(size_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        elem_size = 2U;
    } else if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
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

static bool gd_cross_entropy_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_cross_entropy_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_cross_entropy_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_cross_entropy_logits_shape_valid(const gd_backend_tensor_view *logits,
                                                const gd_backend_tensor_view *targets)
{
    size_t rows;
    size_t classes;
    if (logits == NULL || targets == NULL || logits->rank != 2U || targets->rank != 1U ||
        logits->shape[0] <= 0 || logits->shape[1] <= 1 || targets->shape[0] != logits->shape[0] ||
        logits->strides[1] != 1 || logits->strides[0] != logits->shape[1] ||
        targets->strides[0] != 1) {
        return false;
    }
    rows = (size_t)logits->shape[0];
    classes = (size_t)logits->shape[1];
    return rows <= SIZE_MAX / classes && logits->count == rows * classes && targets->count == rows;
}

static bool gd_cross_entropy_row_vector_valid(const gd_backend_tensor_view *logits,
                                              const gd_backend_tensor_view *row_vector)
{
    return logits != NULL && row_vector != NULL && gd_cross_entropy_view_range_valid(row_vector) &&
           row_vector->dtype == (uint32_t)GD_DTYPE_F32 && row_vector->rank == 1U &&
           row_vector->shape[0] == logits->shape[0] && row_vector->strides[0] == 1 &&
           row_vector->count == (size_t)logits->shape[0];
}

static gd_status gd_cross_entropy_loss_validate(const gd_backend_tensor_view *logits,
                                                const gd_backend_tensor_view *targets,
                                                const gd_backend_tensor_view *row_loss)
{
    if (!gd_cross_entropy_view_range_valid(logits) ||
        !gd_cross_entropy_view_range_valid(targets) ||
        !gd_cross_entropy_view_range_valid(row_loss) ||
        logits->dtype != (uint32_t)GD_DTYPE_F16 || targets->dtype != (uint32_t)GD_DTYPE_I32 ||
        row_loss->dtype != (uint32_t)GD_DTYPE_F32 || row_loss->rank != 1U ||
        !gd_cross_entropy_logits_shape_valid(logits, targets) ||
        row_loss->shape[0] != logits->shape[0] || row_loss->strides[0] != 1 ||
        row_loss->count != (size_t)logits->shape[0] || logits->shape[0] > (int64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_loss_stats_validate(const gd_backend_tensor_view *logits,
                                                      const gd_backend_tensor_view *targets,
                                                      const gd_backend_tensor_view *row_loss,
                                                      const gd_backend_tensor_view *row_max,
                                                      const gd_backend_tensor_view *row_inv_sum)
{
    gd_status st = gd_cross_entropy_loss_validate(logits, targets, row_loss);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cross_entropy_row_vector_valid(logits, row_max) ||
        !gd_cross_entropy_row_vector_valid(logits, row_inv_sum)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_backward_validate(const gd_backend_tensor_view *logits,
                                                    const gd_backend_tensor_view *targets,
                                                    const gd_backend_tensor_view *grad_loss,
                                                    const gd_backend_tensor_view *grad_logits)
{
    if (!gd_cross_entropy_view_range_valid(logits) ||
        !gd_cross_entropy_view_range_valid(targets) ||
        !gd_cross_entropy_view_range_valid(grad_loss) ||
        !gd_cross_entropy_view_range_valid(grad_logits) ||
        logits->dtype != (uint32_t)GD_DTYPE_F16 || targets->dtype != (uint32_t)GD_DTYPE_I32 ||
        grad_loss->dtype != (uint32_t)GD_DTYPE_F32 || grad_loss->rank != 0U ||
        grad_loss->count != 1U || grad_logits->dtype != (uint32_t)GD_DTYPE_F16 ||
        grad_logits->rank != 2U || !gd_cross_entropy_logits_shape_valid(logits, targets) ||
        grad_logits->shape[0] != logits->shape[0] || grad_logits->shape[1] != logits->shape[1] ||
        grad_logits->strides[1] != 1 || grad_logits->strides[0] != grad_logits->shape[1] ||
        grad_logits->count != logits->count || logits->shape[0] > (int64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_backward_stats_validate(const gd_backend_tensor_view *logits,
                                                          const gd_backend_tensor_view *targets,
                                                          const gd_backend_tensor_view *row_max,
                                                          const gd_backend_tensor_view *row_inv_sum,
                                                          const gd_backend_tensor_view *grad_loss,
                                                          const gd_backend_tensor_view *grad_logits)
{
    gd_status st = gd_cross_entropy_backward_validate(logits, targets, grad_loss, grad_logits);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cross_entropy_row_vector_valid(logits, row_max) ||
        !gd_cross_entropy_row_vector_valid(logits, row_inv_sum)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static uint32_t gd_cross_entropy_simdgroups(size_t classes)
{
    if (classes <= 64U) {
        return 1U;
    }
    if (classes <= 256U) {
        return 2U;
    }
    if (classes <= 1024U) {
        return 4U;
    }
    return GD_METAL_CROSS_ENTROPY_MAX_SIMDGROUPS;
}

static void gd_cross_entropy_fill_args(gd_metal_cross_entropy_args *args,
                                       const gd_backend_tensor_view *logits,
                                       const gd_backend_tensor_view *targets,
                                       const gd_backend_tensor_view *out,
                                       const gd_backend_tensor_view *grad_loss,
                                       const gd_backend_tensor_view *row_max,
                                       const gd_backend_tensor_view *row_inv_sum,
                                       float scale)
{
    memset(args, 0, sizeof(*args));
    args->logits_offset = (uint64_t)logits->offset;
    args->target_offset = (uint64_t)targets->offset;
    args->out_offset = (uint64_t)out->offset;
    args->grad_out_offset = grad_loss != NULL ? (uint64_t)grad_loss->offset : 0U;
    args->row_max_offset = row_max != NULL ? (uint64_t)row_max->offset : 0U;
    args->row_inv_sum_offset = row_inv_sum != NULL ? (uint64_t)row_inv_sum->offset : 0U;
    args->rows = (uint64_t)logits->shape[0];
    args->classes = (uint64_t)logits->shape[1];
    args->scale = scale;
    args->simdgroups = gd_cross_entropy_simdgroups((size_t)logits->shape[1]);
}

gd_status gd_backend_cross_entropy_loss(gd_backend *backend,
                                        const gd_backend_tensor_view *logits,
                                        const gd_backend_tensor_view *targets,
                                        const gd_backend_tensor_view *row_loss)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_loss_validate(logits, targets, row_loss);
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
    gd_cross_entropy_fill_args(&args, logits, targets, row_loss, NULL, NULL, NULL, 1.0f);
    [encoder setComputePipelineState:gd_cross_entropy_loss_f16_pso(backend)];
    [encoder setBuffer:gd_cross_entropy_buffer(logits->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_cross_entropy_buffer(targets->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_loss->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_cross_entropy_loss_stats(gd_backend *backend,
                                              const gd_backend_tensor_view *logits,
                                              const gd_backend_tensor_view *targets,
                                              const gd_backend_tensor_view *row_loss,
                                              const gd_backend_tensor_view *row_max,
                                              const gd_backend_tensor_view *row_inv_sum)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_loss_stats_validate(logits, targets, row_loss, row_max, row_inv_sum);
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
    gd_cross_entropy_fill_args(&args, logits, targets, row_loss, NULL, row_max, row_inv_sum, 1.0f);
    [encoder setComputePipelineState:gd_cross_entropy_loss_stats_f16_pso(backend)];
    [encoder setBuffer:gd_cross_entropy_buffer(logits->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_cross_entropy_buffer(targets->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_loss->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_max->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_inv_sum->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_cross_entropy_backward(gd_backend *backend,
                                            const gd_backend_tensor_view *logits,
                                            const gd_backend_tensor_view *targets,
                                            const gd_backend_tensor_view *grad_loss,
                                            const gd_backend_tensor_view *grad_logits,
                                            float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_backward_validate(logits, targets, grad_loss, grad_logits);
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
    gd_cross_entropy_fill_args(&args, logits, targets, grad_logits, grad_loss, NULL, NULL, scale);
    [encoder setComputePipelineState:gd_cross_entropy_backward_f16_pso(backend)];
    [encoder setBuffer:gd_cross_entropy_buffer(logits->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_cross_entropy_buffer(targets->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_cross_entropy_buffer(grad_loss->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_cross_entropy_buffer(grad_logits->buffer) offset:0U atIndex:3U];
    [encoder setBytes:&args length:sizeof(args) atIndex:4U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_cross_entropy_backward_stats(gd_backend *backend,
                                                  const gd_backend_tensor_view *logits,
                                                  const gd_backend_tensor_view *targets,
                                                  const gd_backend_tensor_view *row_max,
                                                  const gd_backend_tensor_view *row_inv_sum,
                                                  const gd_backend_tensor_view *grad_loss,
                                                  const gd_backend_tensor_view *grad_logits,
                                                  float scale)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_backward_stats_validate(logits, targets, row_max, row_inv_sum, grad_loss, grad_logits);
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
    gd_cross_entropy_fill_args(&args, logits, targets, grad_logits, grad_loss, row_max, row_inv_sum, scale);
    [encoder setComputePipelineState:gd_cross_entropy_backward_stats_f16_pso(backend)];
    [encoder setBuffer:gd_cross_entropy_buffer(logits->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_cross_entropy_buffer(targets->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_cross_entropy_buffer(grad_loss->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_max->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_cross_entropy_buffer(row_inv_sum->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_cross_entropy_buffer(grad_logits->buffer) offset:0U atIndex:5U];
    [encoder setBytes:&args length:sizeof(args) atIndex:6U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
