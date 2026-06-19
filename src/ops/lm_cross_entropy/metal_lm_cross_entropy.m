#include "../../backends/metal/metal_backend_internal.h"
#include "metal_lm_cross_entropy_types.h"

#include <gradients/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_lm_cross_entropy_online_update_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->lm_cross_entropy_online_update_f16_pso;
}

static id<MTLComputePipelineState> gd_lm_cross_entropy_finalize_f32_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->lm_cross_entropy_finalize_f32_pso;
}

static id<MTLComputePipelineState> gd_lm_cross_entropy_reduce_normalize_f32_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->lm_cross_entropy_reduce_normalize_f32_pso;
}

static id<MTLComputePipelineState> gd_lm_cross_entropy_backward_chunk_f16_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->lm_cross_entropy_backward_chunk_f16_pso;
}

static id<MTLBuffer> gd_lm_cross_entropy_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_lm_cross_entropy_byte_range_valid(const gd_backend_buffer *buffer,
                                                 size_t offset,
                                                 size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_lm_cross_entropy_count_bytes(size_t count, uint32_t dtype, size_t *out_nbytes)
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

static bool gd_lm_cross_entropy_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_lm_cross_entropy_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_lm_cross_entropy_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_lm_cross_entropy_softcap_valid(float logits_softcap)
{
    return logits_softcap == 0.0f ||
           (isfinite(logits_softcap) && logits_softcap > 0.0f && isfinite(1.0f / logits_softcap));
}

static bool gd_lm_cross_entropy_logits_chunk_valid(const gd_backend_tensor_view *logits,
                                                   const gd_backend_tensor_view *targets)
{
    size_t rows;
    size_t classes;
    if (logits == NULL || targets == NULL || logits->rank != 2U || targets->rank != 1U ||
        logits->shape[0] <= 0 || logits->shape[1] <= 0 || targets->shape[0] != logits->shape[0] ||
        logits->strides[1] != 1 || logits->strides[0] != logits->shape[1] ||
        targets->strides[0] != 1) {
        return false;
    }
    rows = (size_t)logits->shape[0];
    classes = (size_t)logits->shape[1];
    return rows <= SIZE_MAX / classes && logits->count == rows * classes && targets->count == rows;
}

static bool gd_lm_cross_entropy_bias_chunk_valid(const gd_backend_tensor_view *bias,
                                                 const gd_backend_tensor_view *logits)
{
    return bias == NULL ||
           (logits != NULL && gd_lm_cross_entropy_view_range_valid(bias) &&
            bias->dtype == (uint32_t)GD_DTYPE_F16 && bias->rank == 1U &&
            bias->shape[0] == logits->shape[1] && bias->strides[0] == 1 &&
            bias->count == (size_t)logits->shape[1]);
}

static bool gd_lm_cross_entropy_row_vector_valid(const gd_backend_tensor_view *targets,
                                                 const gd_backend_tensor_view *row_vector)
{
    return targets != NULL && row_vector != NULL && gd_lm_cross_entropy_view_range_valid(row_vector) &&
           row_vector->dtype == (uint32_t)GD_DTYPE_F32 && row_vector->rank == 1U &&
           row_vector->shape[0] == targets->shape[0] && row_vector->strides[0] == 1 &&
           row_vector->count == (size_t)targets->shape[0];
}

static gd_status gd_lm_cross_entropy_online_update_validate(const gd_backend_tensor_view *logits,
                                                            const gd_backend_tensor_view *bias,
                                                            const gd_backend_tensor_view *targets,
                                                            const gd_backend_tensor_view *row_loss,
                                                            const gd_backend_tensor_view *row_max,
                                                            const gd_backend_tensor_view *row_inv_sum,
                                                            uint64_t class_start,
                                                            uint64_t total_classes)
{
    if (!gd_lm_cross_entropy_view_range_valid(logits) ||
        !gd_lm_cross_entropy_view_range_valid(targets) ||
        logits->dtype != (uint32_t)GD_DTYPE_F16 || targets->dtype != (uint32_t)GD_DTYPE_I32 ||
        !gd_lm_cross_entropy_logits_chunk_valid(logits, targets) ||
        !gd_lm_cross_entropy_bias_chunk_valid(bias, logits) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_loss) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_max) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_inv_sum) ||
        class_start > total_classes || (uint64_t)logits->shape[1] > total_classes - class_start ||
        logits->shape[0] > (int64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_finalize_validate(const gd_backend_tensor_view *targets,
                                                       const gd_backend_tensor_view *row_loss,
                                                       const gd_backend_tensor_view *row_max,
                                                       const gd_backend_tensor_view *row_inv_sum,
                                                       const gd_backend_tensor_view *row_valid)
{
    if (!gd_lm_cross_entropy_view_range_valid(targets) ||
        targets->dtype != (uint32_t)GD_DTYPE_I32 || targets->rank != 1U ||
        targets->shape[0] <= 0 || targets->strides[0] != 1 ||
        targets->count != (size_t)targets->shape[0] ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_loss) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_max) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_inv_sum) ||
        !gd_lm_cross_entropy_row_vector_valid(targets, row_valid) ||
        targets->shape[0] > (int64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_lm_cross_entropy_scalar_f32_valid(const gd_backend_tensor_view *view)
{
    return gd_lm_cross_entropy_view_range_valid(view) && view->dtype == (uint32_t)GD_DTYPE_F32 &&
           view->rank == 0U && view->count == 1U;
}

static gd_status gd_lm_cross_entropy_reduce_normalize_validate(const gd_backend_tensor_view *row_loss,
                                                               const gd_backend_tensor_view *row_valid,
                                                               const gd_backend_tensor_view *loss,
                                                               const gd_backend_tensor_view *inv_valid_count)
{
    if (!gd_lm_cross_entropy_view_range_valid(row_loss) ||
        !gd_lm_cross_entropy_view_range_valid(row_valid) ||
        row_loss->dtype != (uint32_t)GD_DTYPE_F32 ||
        row_valid->dtype != (uint32_t)GD_DTYPE_F32 ||
        row_loss->rank != 1U || row_valid->rank != 1U ||
        row_loss->shape[0] <= 0 || row_valid->shape[0] != row_loss->shape[0] ||
        row_loss->strides[0] != 1 || row_valid->strides[0] != 1 ||
        row_loss->count != (size_t)row_loss->shape[0] ||
        row_valid->count != (size_t)row_valid->shape[0] ||
        row_loss->shape[0] > (int64_t)UINT32_MAX ||
        !gd_lm_cross_entropy_scalar_f32_valid(loss) ||
        !gd_lm_cross_entropy_scalar_f32_valid(inv_valid_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_backward_chunk_validate(const gd_backend_tensor_view *logits,
                                                             const gd_backend_tensor_view *bias,
                                                             const gd_backend_tensor_view *targets,
                                                             const gd_backend_tensor_view *row_max,
                                                             const gd_backend_tensor_view *row_inv_sum,
                                                             const gd_backend_tensor_view *grad_loss,
                                                             const gd_backend_tensor_view *inv_valid_count,
                                                             const gd_backend_tensor_view *grad_logits,
                                                             uint64_t class_start,
                                                             uint64_t total_classes)
{
    gd_status st = gd_lm_cross_entropy_online_update_validate(logits,
                                                             bias,
                                                             targets,
                                                             row_inv_sum,
                                                             row_max,
                                                             row_inv_sum,
                                                             class_start,
                                                             total_classes);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_lm_cross_entropy_scalar_f32_valid(grad_loss) ||
        !gd_lm_cross_entropy_scalar_f32_valid(inv_valid_count) ||
        !gd_lm_cross_entropy_view_range_valid(grad_logits) ||
        grad_logits->dtype != (uint32_t)GD_DTYPE_F16 ||
        grad_logits->rank != 2U || grad_logits->shape[0] != logits->shape[0] ||
        grad_logits->shape[1] != logits->shape[1] || grad_logits->strides[1] != 1 ||
        grad_logits->strides[0] != grad_logits->shape[1] || grad_logits->count != logits->count) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static uint32_t gd_lm_cross_entropy_simdgroups(size_t classes)
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
    return GD_METAL_LM_CROSS_ENTROPY_MAX_SIMDGROUPS;
}

static void gd_lm_cross_entropy_fill_args(gd_metal_lm_cross_entropy_args *args,
                                          const gd_backend_tensor_view *logits,
                                          const gd_backend_tensor_view *bias,
                                          const gd_backend_tensor_view *targets,
                                          const gd_backend_tensor_view *row_loss,
                                          const gd_backend_tensor_view *row_max,
                                          const gd_backend_tensor_view *row_inv_sum,
                                          const gd_backend_tensor_view *grad_loss,
                                          const gd_backend_tensor_view *inv_valid_count,
                                          const gd_backend_tensor_view *out,
                                          uint64_t class_start,
                                          uint64_t total_classes,
                                          float scale,
                                          float logits_softcap)
{
    memset(args, 0, sizeof(*args));
    args->logits_offset = logits != NULL ? (uint64_t)logits->offset : 0U;
    args->bias_offset = bias != NULL ? (uint64_t)bias->offset : 0U;
    args->target_offset = targets != NULL ? (uint64_t)targets->offset : 0U;
    args->row_loss_offset = row_loss != NULL ? (uint64_t)row_loss->offset : 0U;
    args->row_max_offset = row_max != NULL ? (uint64_t)row_max->offset : 0U;
    args->row_inv_sum_offset = row_inv_sum != NULL ? (uint64_t)row_inv_sum->offset : 0U;
    args->grad_out_offset = grad_loss != NULL ? (uint64_t)grad_loss->offset : 0U;
    args->inv_valid_count_offset = inv_valid_count != NULL ? (uint64_t)inv_valid_count->offset : 0U;
    args->out_offset = out != NULL ? (uint64_t)out->offset : 0U;
    if (targets != NULL) {
        args->rows = (uint64_t)targets->shape[0];
    } else if (row_loss != NULL && row_loss->rank == 1U) {
        args->rows = (uint64_t)row_loss->shape[0];
    } else {
        args->rows = 0U;
    }
    args->chunk_classes = logits != NULL ? (uint64_t)logits->shape[1] : 0U;
    args->total_classes = total_classes;
    args->class_start = class_start;
    args->scale = scale;
    args->logits_softcap = logits_softcap;
    args->inv_logits_softcap = logits_softcap > 0.0f ? 1.0f / logits_softcap : 0.0f;
    args->has_bias = bias != NULL ? 1U : 0U;
    args->simdgroups = logits != NULL ? gd_lm_cross_entropy_simdgroups((size_t)logits->shape[1]) :
                                        gd_lm_cross_entropy_simdgroups((size_t)args->rows);
}

gd_status gd_backend_lm_cross_entropy_online_update(gd_backend *backend,
                                                    const gd_backend_tensor_view *logits_chunk,
                                                    const gd_backend_tensor_view *bias_chunk,
                                                    const gd_backend_tensor_view *targets,
                                                    const gd_backend_tensor_view *row_loss,
                                                    const gd_backend_tensor_view *row_max,
                                                    const gd_backend_tensor_view *row_inv_sum,
                                                    uint64_t class_start,
                                                    uint64_t total_classes,
                                                    float logits_softcap)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_lm_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_lm_cross_entropy_softcap_valid(logits_softcap)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_lm_cross_entropy_online_update_validate(logits_chunk,
                                                    bias_chunk,
                                                    targets,
                                                    row_loss,
                                                    row_max,
                                                    row_inv_sum,
                                                    class_start,
                                                    total_classes);
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
    gd_lm_cross_entropy_fill_args(&args,
                                  logits_chunk,
                                  bias_chunk,
                                  targets,
                                  row_loss,
                                  row_max,
                                  row_inv_sum,
                                  NULL,
                                  NULL,
                                  NULL,
                                  class_start,
                                  total_classes,
                                  1.0f,
                                  logits_softcap);
    [encoder setComputePipelineState:gd_lm_cross_entropy_online_update_f16_pso(backend)];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(logits_chunk->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(bias_chunk != NULL ? bias_chunk->buffer : logits_chunk->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(targets->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_loss->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_max->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_inv_sum->buffer) offset:0U atIndex:5U];
    [encoder setBytes:&args length:sizeof(args) atIndex:6U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_lm_cross_entropy_finalize(gd_backend *backend,
                                               const gd_backend_tensor_view *targets,
                                               const gd_backend_tensor_view *row_loss,
                                               const gd_backend_tensor_view *row_max,
                                               const gd_backend_tensor_view *row_inv_sum,
                                               const gd_backend_tensor_view *row_valid,
                                               uint64_t total_classes)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_lm_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || total_classes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_lm_cross_entropy_finalize_validate(targets, row_loss, row_max, row_inv_sum, row_valid);
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
    gd_lm_cross_entropy_fill_args(&args,
                                  NULL,
                                  NULL,
                                  targets,
                                  row_loss,
                                  row_max,
                                  row_inv_sum,
                                  NULL,
                                  NULL,
                                  row_valid,
                                  0U,
                                  total_classes,
                                  1.0f,
                                  0.0f);
    [encoder setComputePipelineState:gd_lm_cross_entropy_finalize_f32_pso(backend)];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(targets->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_loss->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_max->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_inv_sum->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_valid->buffer) offset:0U atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(128U, 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_lm_cross_entropy_reduce_normalize(gd_backend *backend,
                                                       const gd_backend_tensor_view *row_loss,
                                                       const gd_backend_tensor_view *row_valid,
                                                       const gd_backend_tensor_view *loss,
                                                       const gd_backend_tensor_view *inv_valid_count)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_lm_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_lm_cross_entropy_reduce_normalize_validate(row_loss, row_valid, loss, inv_valid_count);
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
    gd_lm_cross_entropy_fill_args(&args,
                                  NULL,
                                  NULL,
                                  NULL,
                                  row_loss,
                                  row_valid,
                                  NULL,
                                  NULL,
                                  inv_valid_count,
                                  loss,
                                  0U,
                                  0U,
                                  1.0f,
                                  0.0f);
    [encoder setComputePipelineState:gd_lm_cross_entropy_reduce_normalize_f32_pso(backend)];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_loss->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_valid->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(loss->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(inv_valid_count->buffer) offset:0U atIndex:3U];
    [encoder setBytes:&args length:sizeof(args) atIndex:4U];
    grid = MTLSizeMake(1U, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_lm_cross_entropy_backward_chunk(gd_backend *backend,
                                                     const gd_backend_tensor_view *logits_chunk,
                                                     const gd_backend_tensor_view *bias_chunk,
                                                     const gd_backend_tensor_view *targets,
                                                     const gd_backend_tensor_view *row_max,
                                                     const gd_backend_tensor_view *row_inv_sum,
                                                     const gd_backend_tensor_view *grad_loss,
                                                     const gd_backend_tensor_view *inv_valid_count,
                                                     const gd_backend_tensor_view *grad_logits_chunk,
                                                     uint64_t class_start,
                                                     uint64_t total_classes,
                                                     float scale,
                                                     float logits_softcap)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_lm_cross_entropy_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || !(scale == scale) || !gd_lm_cross_entropy_softcap_valid(logits_softcap)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_lm_cross_entropy_backward_chunk_validate(logits_chunk,
                                                     bias_chunk,
                                                     targets,
                                                     row_max,
                                                     row_inv_sum,
                                                     grad_loss,
                                                     inv_valid_count,
                                                     grad_logits_chunk,
                                                     class_start,
                                                     total_classes);
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
    gd_lm_cross_entropy_fill_args(&args,
                                  logits_chunk,
                                  bias_chunk,
                                  targets,
                                  NULL,
                                  row_max,
                                  row_inv_sum,
                                  grad_loss,
                                  inv_valid_count,
                                  grad_logits_chunk,
                                  class_start,
                                  total_classes,
                                  scale,
                                  logits_softcap);
    [encoder setComputePipelineState:gd_lm_cross_entropy_backward_chunk_f16_pso(backend)];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(logits_chunk->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(bias_chunk != NULL ? bias_chunk->buffer : logits_chunk->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(targets->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_max->buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(row_inv_sum->buffer) offset:0U atIndex:4U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(grad_loss->buffer) offset:0U atIndex:5U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(inv_valid_count->buffer) offset:0U atIndex:6U];
    [encoder setBuffer:gd_lm_cross_entropy_buffer(grad_logits_chunk->buffer) offset:0U atIndex:7U];
    [encoder setBytes:&args length:sizeof(args) atIndex:8U];
    grid = MTLSizeMake((NSUInteger)args.rows, 1U, 1U);
    threads = MTLSizeMake(32U, (NSUInteger)args.simdgroups, 1U);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
