#include "../../backends/metal/metal_backend_internal.h"
#include "metal_adamw_types.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define GD_METAL_ADAMW_MAX_THREADS_PER_GROUP 256U

static id<MTLComputePipelineState> gd_metal_adamw_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->adamw_pso;
}

static id<MTLComputePipelineState> gd_metal_grad_norm_stage_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->grad_norm_stage_pso;
}

static id<MTLComputePipelineState> gd_metal_grad_clip_finalize_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->grad_clip_finalize_pso;
}

static id<MTLBuffer> gd_metal_adamw_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_metal_adamw_byte_range_valid(const gd_backend_buffer *buffer,
                                            size_t offset,
                                            size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_metal_adamw_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static gd_status gd_metal_adamw_elem_size(uint32_t dtype, size_t *out_elem_size)
{
    if (out_elem_size == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (dtype == 1U) {
        *out_elem_size = 2U;
        return GD_OK;
    }
    if (dtype == 3U) {
        *out_elem_size = 4U;
        return GD_OK;
    }
    return GD_ERR_UNSUPPORTED;
}

static gd_status gd_metal_adamw_validate_desc(const gd_backend_adamw_desc *desc)
{
    size_t param_elem_size;
    size_t grad_elem_size;
    size_t param_nbytes;
    size_t grad_nbytes;
    size_t moment_nbytes;
    gd_status st;
    if (desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_adamw_elem_size(desc->param_dtype, &param_elem_size);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_adamw_elem_size(desc->grad_dtype, &grad_elem_size);
    if (st != GD_OK) {
        return st;
    }
    if (desc->param_buffer == NULL || desc->grad_buffer == NULL || desc->m_buffer == NULL ||
        desc->v_buffer == NULL || desc->count == 0U || desc->count > UINT32_MAX ||
        !(desc->lr >= 0.0f) || !(desc->beta1 >= 0.0f) || !(desc->beta1 < 1.0f) ||
        !(desc->beta2 >= 0.0f) || !(desc->beta2 < 1.0f) || !(desc->eps > 0.0f) ||
        !(desc->weight_decay >= 0.0f) || !(desc->bias_correction1 > 0.0f) ||
        !(desc->bias_correction2 > 0.0f) ||
        !gd_metal_adamw_count_bytes(desc->count, param_elem_size, &param_nbytes) ||
        !gd_metal_adamw_count_bytes(desc->count, grad_elem_size, &grad_nbytes) ||
        !gd_metal_adamw_count_bytes(desc->count, 4U, &moment_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->param_buffer, desc->param_offset, param_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->m_buffer, desc->m_offset, moment_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->v_buffer, desc->v_offset, moment_nbytes) ||
        (desc->has_master != 0U &&
         (desc->master_buffer == NULL ||
          !gd_metal_adamw_byte_range_valid(desc->master_buffer,
                                           desc->master_offset,
                                           moment_nbytes))) ||
        (desc->has_grad_scale != 0U &&
         (desc->grad_scale_buffer == NULL ||
          !gd_metal_adamw_byte_range_valid(desc->grad_scale_buffer,
                                           desc->grad_scale_offset,
                                           sizeof(float))))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_metal_adamw_make_args(const gd_backend_adamw_desc *desc,
                                     gd_metal_adamw_args *args)
{
    const float bias_correction2_sqrt = sqrtf(desc->bias_correction2);
    memset(args, 0, sizeof(*args));
    args->param_offset = (uint64_t)desc->param_offset;
    args->master_offset = (uint64_t)desc->master_offset;
    args->grad_offset = (uint64_t)desc->grad_offset;
    args->m_offset = (uint64_t)desc->m_offset;
    args->v_offset = (uint64_t)desc->v_offset;
    args->grad_scale_offset = (uint64_t)desc->grad_scale_offset;
    args->count = (uint64_t)desc->count;
    args->param_dtype = desc->param_dtype;
    args->grad_dtype = desc->grad_dtype;
    args->has_master = desc->has_master != 0U ? 1U : 0U;
    args->has_grad_scale = desc->has_grad_scale != 0U ? 1U : 0U;
    args->lr = desc->lr;
    args->beta1 = desc->beta1;
    args->beta2 = desc->beta2;
    args->eps = desc->eps;
    args->weight_decay = desc->weight_decay;
    args->bias_correction1 = desc->bias_correction1;
    args->bias_correction2 = desc->bias_correction2;
    args->one_minus_beta1 = 1.0f - desc->beta1;
    args->one_minus_beta2 = 1.0f - desc->beta2;
    args->step_scale = desc->lr * bias_correction2_sqrt / desc->bias_correction1;
    args->decay_scale = desc->lr * desc->weight_decay;
    args->eps_scaled = desc->eps * bias_correction2_sqrt;
}

static void gd_metal_adamw_encode(id<MTLComputeCommandEncoder> encoder,
                                  const gd_backend_adamw_desc *desc)
{
    gd_metal_adamw_args args;
    MTLSize grid;
    MTLSize threads;
    gd_metal_adamw_make_args(desc, &args);
    [encoder setBuffer:gd_metal_adamw_buffer(desc->param_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->grad_buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->m_buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->v_buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->has_master != 0U ? desc->master_buffer : desc->param_buffer)
                offset:0U
               atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->has_grad_scale != 0U ?
                                             desc->grad_scale_buffer : desc->param_buffer)
                offset:0U
               atIndex:6U];
    grid = MTLSizeMake((NSUInteger)desc->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(desc->count < GD_METAL_ADAMW_MAX_THREADS_PER_GROUP ?
                                       desc->count : GD_METAL_ADAMW_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
}

gd_status gd_backend_adamw_batch(gd_backend *backend,
                                  const gd_backend_adamw_desc *descs,
                                  uint32_t desc_count)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    bool immediate;
    uint32_t i;
    gd_status st;
    if (desc_count == 0U) {
        return GD_OK;
    }
    if (backend == NULL || descs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < desc_count; ++i) {
        st = gd_metal_adamw_validate_desc(&descs[i]);
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
    [encoder setComputePipelineState:gd_metal_adamw_pso(backend)];
    for (i = 0U; i < desc_count; ++i) {
        gd_metal_adamw_encode(encoder, &descs[i]);
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc)
{
    if (desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_backend_adamw_batch(backend, desc, 1U);
}

static size_t gd_metal_grad_norm_group_count(size_t count)
{
    const size_t block = (size_t)GD_METAL_GRAD_NORM_BLOCK_ELEMS;
    return (count / block) + ((count % block) != 0U ? (size_t)1U : (size_t)0U);
}

static gd_status gd_metal_grad_norm_validate_desc(const gd_backend_grad_norm_desc *desc)
{
    size_t grad_elem_size;
    size_t grad_nbytes;
    size_t partial_nbytes;
    size_t expected_partials;
    gd_status st;
    if (desc == NULL || desc->grad_buffer == NULL || desc->partial_buffer == NULL ||
        desc->count == 0U || desc->count > UINT32_MAX || desc->partial_count == 0U ||
        desc->partial_count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_adamw_elem_size(desc->grad_dtype, &grad_elem_size);
    if (st != GD_OK) {
        return st;
    }
    expected_partials = gd_metal_grad_norm_group_count(desc->count);
    if (desc->partial_count != expected_partials ||
        !gd_metal_adamw_count_bytes(desc->count, grad_elem_size, &grad_nbytes) ||
        !gd_metal_adamw_count_bytes(desc->partial_count, sizeof(float), &partial_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->partial_buffer,
                                         desc->partial_offset,
                                         partial_nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_metal_grad_norm_encode_stage(id<MTLComputeCommandEncoder> encoder,
                                            const gd_backend_grad_norm_desc *desc)
{
    gd_metal_grad_norm_stage_args args;
    MTLSize groups;
    MTLSize threads;
    memset(&args, 0, sizeof(args));
    args.grad_offset = (uint64_t)desc->grad_offset;
    args.partial_offset = (uint64_t)desc->partial_offset;
    args.count = (uint64_t)desc->count;
    args.grad_dtype = desc->grad_dtype;
    [encoder setBuffer:gd_metal_adamw_buffer(desc->grad_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->partial_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    groups = MTLSizeMake((NSUInteger)desc->partial_count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)GD_METAL_GRAD_NORM_THREADS, 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

gd_status gd_backend_grad_clip_scale(gd_backend *backend,
                                      const gd_backend_grad_norm_desc *descs,
                                      uint32_t desc_count,
                                      gd_backend_buffer *scale_buffer,
                                      size_t scale_offset,
                                      float max_norm,
                                      float eps)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_backend_buffer *partial_buffer;
    gd_metal_grad_clip_finalize_args final_args;
    MTLSize final_groups;
    MTLSize final_threads;
    bool immediate;
    size_t partial_base_offset;
    size_t running_partial_offset;
    size_t total_partials = 0U;
    uint32_t i;
    gd_status st;
    if (desc_count == 0U) {
        return GD_OK;
    }
    if (backend == NULL || descs == NULL || scale_buffer == NULL ||
        !(max_norm > 0.0f) || !isfinite(max_norm) || !(eps > 0.0f) || !isfinite(eps) ||
        !gd_metal_adamw_byte_range_valid(scale_buffer, scale_offset, 2U * sizeof(float))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    partial_buffer = descs[0].partial_buffer;
    partial_base_offset = descs[0].partial_offset;
    running_partial_offset = partial_base_offset;
    for (i = 0U; i < desc_count; ++i) {
        size_t partial_nbytes;
        st = gd_metal_grad_norm_validate_desc(&descs[i]);
        if (st != GD_OK) {
            return st;
        }
        if (descs[i].partial_buffer != partial_buffer ||
            descs[i].partial_offset != running_partial_offset ||
            !gd_metal_adamw_count_bytes(descs[i].partial_count, sizeof(float), &partial_nbytes) ||
            total_partials > SIZE_MAX - descs[i].partial_count ||
            running_partial_offset > SIZE_MAX - partial_nbytes) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        total_partials += descs[i].partial_count;
        running_partial_offset += partial_nbytes;
    }
    if (total_partials == 0U || total_partials > UINT32_MAX) {
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
    [encoder setComputePipelineState:gd_metal_grad_norm_stage_pso(backend)];
    for (i = 0U; i < desc_count; ++i) {
        gd_metal_grad_norm_encode_stage(encoder, &descs[i]);
    }
    memset(&final_args, 0, sizeof(final_args));
    final_args.partial_offset = (uint64_t)partial_base_offset;
    final_args.scale_offset = (uint64_t)scale_offset;
    final_args.partial_count = (uint64_t)total_partials;
    final_args.max_norm = max_norm;
    final_args.eps = eps;
    [encoder setComputePipelineState:gd_metal_grad_clip_finalize_pso(backend)];
    [encoder setBuffer:gd_metal_adamw_buffer(partial_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_adamw_buffer(scale_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&final_args length:sizeof(final_args) atIndex:2U];
    final_groups = MTLSizeMake(1U, 1U, 1U);
    final_threads = MTLSizeMake((NSUInteger)GD_METAL_GRAD_NORM_THREADS, 1U, 1U);
    [encoder dispatchThreadgroups:final_groups threadsPerThreadgroup:final_threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
