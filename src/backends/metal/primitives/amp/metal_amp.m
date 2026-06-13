#include "../../metal_backend_internal.h"
#include "metal_amp_types.h"

#include <stdint.h>
#include <string.h>

#define GD_METAL_AMP_MAX_THREADS_PER_GROUP 256U

static id<MTLComputePipelineState> gd_metal_amp_begin_step_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->amp_begin_step_pso;
}

static id<MTLComputePipelineState> gd_metal_amp_finish_step_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->amp_finish_step_pso;
}

static id<MTLComputePipelineState> gd_metal_amp_fill_scale_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->amp_fill_scale_pso;
}

static id<MTLComputePipelineState> gd_metal_amp_scale_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->amp_scale_pso;
}

static id<MTLComputePipelineState> gd_metal_amp_unscale_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->amp_unscale_pso;
}

static id<MTLBuffer> gd_metal_amp_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_metal_amp_byte_range_valid(const gd_backend_buffer *buffer,
                                          size_t offset,
                                          size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_metal_amp_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static gd_status gd_metal_amp_elem_size(uint32_t dtype, size_t *out_elem_size)
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

static gd_status gd_metal_amp_validate_state_desc(const gd_backend_amp_state_desc *desc)
{
    if (desc == NULL || desc->scale_buffer == NULL || desc->flags_buffer == NULL ||
        !(desc->growth_factor > 1.0f) || !(desc->backoff_factor > 0.0f) ||
        !(desc->backoff_factor < 1.0f) || !(desc->min_scale > 0.0f) ||
        !(desc->max_scale >= desc->min_scale) || desc->growth_interval == 0U ||
        !gd_metal_amp_byte_range_valid(desc->scale_buffer, desc->scale_offset, 2U * sizeof(float)) ||
        !gd_metal_amp_byte_range_valid(desc->flags_buffer, desc->flags_offset, 3U * sizeof(uint32_t))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_metal_amp_make_state_args(const gd_backend_amp_state_desc *desc,
                                         gd_metal_amp_state_args *args)
{
    memset(args, 0, sizeof(*args));
    args->scale_offset = (uint64_t)desc->scale_offset;
    args->flags_offset = (uint64_t)desc->flags_offset;
    args->growth_factor = desc->growth_factor;
    args->backoff_factor = desc->backoff_factor;
    args->min_scale = desc->min_scale;
    args->max_scale = desc->max_scale;
    args->growth_interval = desc->growth_interval;
}

static gd_status gd_metal_amp_encode_state_kernel(gd_backend *backend,
                                                  const gd_backend_amp_state_desc *desc,
                                                  id<MTLComputePipelineState> pso)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_amp_state_args args;
    bool immediate;
    gd_status st;
    st = gd_metal_amp_validate_state_desc(desc);
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
    gd_metal_amp_make_state_args(desc, &args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_metal_amp_buffer(desc->scale_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(desc->flags_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    [encoder dispatchThreads:MTLSizeMake(1U, 1U, 1U) threadsPerThreadgroup:MTLSizeMake(1U, 1U, 1U)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_amp_begin_step(gd_backend *backend,
                                    const gd_backend_amp_state_desc *desc)
{
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_metal_amp_encode_state_kernel(backend, desc, gd_metal_amp_begin_step_pso(backend));
}

gd_status gd_backend_amp_finish_step(gd_backend *backend,
                                     const gd_backend_amp_state_desc *desc)
{
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_metal_amp_encode_state_kernel(backend, desc, gd_metal_amp_finish_step_pso(backend));
}

static gd_status gd_metal_amp_validate_tensor_args(gd_backend_buffer *dst_buffer,
                                                   size_t dst_offset,
                                                   gd_backend_buffer *src_buffer,
                                                   size_t src_offset,
                                                   size_t count,
                                                   uint32_t dtype,
                                                   gd_backend_buffer *scale_buffer,
                                                   size_t scale_offset,
                                                   bool has_src)
{
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (dst_buffer == NULL || scale_buffer == NULL || count == 0U || count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_amp_elem_size(dtype, &elem_size);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_metal_amp_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_amp_byte_range_valid(dst_buffer, dst_offset, nbytes) ||
        !gd_metal_amp_byte_range_valid(scale_buffer, scale_offset, 2U * sizeof(float)) ||
        (has_src && (src_buffer == NULL ||
                     !gd_metal_amp_byte_range_valid(src_buffer, src_offset, nbytes)))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status gd_backend_amp_fill_scale(gd_backend *backend,
                                    gd_backend_buffer *dst_buffer,
                                    size_t dst_offset,
                                    size_t count,
                                    uint32_t dtype,
                                    gd_backend_buffer *scale_buffer,
                                    size_t scale_offset)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_amp_tensor_args args;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_amp_validate_tensor_args(dst_buffer,
                                           dst_offset,
                                           NULL,
                                           0U,
                                           count,
                                           dtype,
                                           scale_buffer,
                                           scale_offset,
                                           false);
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
    memset(&args, 0, sizeof(args));
    args.dst_offset = (uint64_t)dst_offset;
    args.scale_offset = (uint64_t)scale_offset;
    args.count = (uint64_t)count;
    args.dtype = dtype;
    [encoder setComputePipelineState:gd_metal_amp_fill_scale_pso(backend)];
    [encoder setBuffer:gd_metal_amp_buffer(dst_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(scale_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1U, 1U)
       threadsPerThreadgroup:MTLSizeMake((NSUInteger)(count < GD_METAL_AMP_MAX_THREADS_PER_GROUP ?
                                                      count : GD_METAL_AMP_MAX_THREADS_PER_GROUP),
                                         1U,
                                         1U)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_amp_scale(gd_backend *backend,
                               gd_backend_buffer *dst_buffer,
                               size_t dst_offset,
                               gd_backend_buffer *src_buffer,
                               size_t src_offset,
                               size_t count,
                               uint32_t dtype,
                               gd_backend_buffer *scale_buffer,
                               size_t scale_offset)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_amp_tensor_args args;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_amp_validate_tensor_args(dst_buffer,
                                           dst_offset,
                                           src_buffer,
                                           src_offset,
                                           count,
                                           dtype,
                                           scale_buffer,
                                           scale_offset,
                                           true);
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
    memset(&args, 0, sizeof(args));
    args.dst_offset = (uint64_t)dst_offset;
    args.src_offset = (uint64_t)src_offset;
    args.scale_offset = (uint64_t)scale_offset;
    args.count = (uint64_t)count;
    args.dtype = dtype;
    [encoder setComputePipelineState:gd_metal_amp_scale_pso(backend)];
    [encoder setBuffer:gd_metal_amp_buffer(dst_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(src_buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_amp_buffer(scale_buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1U, 1U)
       threadsPerThreadgroup:MTLSizeMake((NSUInteger)(count < GD_METAL_AMP_MAX_THREADS_PER_GROUP ?
                                                      count : GD_METAL_AMP_MAX_THREADS_PER_GROUP),
                                         1U,
                                         1U)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static gd_status gd_metal_amp_validate_unscale_desc(const gd_backend_amp_unscale_desc *desc)
{
    size_t elem_size;
    size_t grad_nbytes;
    gd_status st;
    if (desc == NULL || desc->grad_buffer == NULL || desc->inv_scale_buffer == NULL ||
        desc->found_inf_buffer == NULL || desc->count == 0U || desc->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_amp_elem_size(desc->grad_dtype, &elem_size);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_metal_amp_count_bytes(desc->count, elem_size, &grad_nbytes) ||
        !gd_metal_amp_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
        !gd_metal_amp_byte_range_valid(desc->inv_scale_buffer, desc->inv_scale_offset, sizeof(float)) ||
        !gd_metal_amp_byte_range_valid(desc->found_inf_buffer, desc->found_inf_offset, 4U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_metal_amp_encode_unscale(id<MTLComputeCommandEncoder> encoder,
                                        const gd_backend_amp_unscale_desc *desc)
{
    gd_metal_amp_unscale_args args;
    MTLSize grid;
    MTLSize threads;
    memset(&args, 0, sizeof(args));
    args.grad_offset = (uint64_t)desc->grad_offset;
    args.inv_scale_offset = (uint64_t)desc->inv_scale_offset;
    args.found_inf_offset = (uint64_t)desc->found_inf_offset;
    args.count = (uint64_t)desc->count;
    args.grad_dtype = desc->grad_dtype;
    [encoder setBuffer:gd_metal_amp_buffer(desc->grad_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(desc->inv_scale_buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_amp_buffer(desc->found_inf_buffer) offset:0U atIndex:2U];
    [encoder setBytes:&args length:sizeof(args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)desc->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(desc->count < GD_METAL_AMP_MAX_THREADS_PER_GROUP ?
                                       desc->count : GD_METAL_AMP_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
}

gd_status gd_backend_amp_unscale_batch(gd_backend *backend,
                                        const gd_backend_amp_unscale_desc *descs,
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
        st = gd_metal_amp_validate_unscale_desc(&descs[i]);
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
    [encoder setComputePipelineState:gd_metal_amp_unscale_pso(backend)];
    for (i = 0U; i < desc_count; ++i) {
        gd_metal_amp_encode_unscale(encoder, &descs[i]);
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_amp_unscale(gd_backend *backend, const gd_backend_amp_unscale_desc *desc)
{
    if (desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_backend_amp_unscale_batch(backend, desc, 1U);
}
