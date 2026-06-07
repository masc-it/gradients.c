#include "../../metal_backend_internal.h"
#include "metal_amp_types.h"

#include <stdint.h>
#include <string.h>

#define GD_METAL_AMP_MAX_THREADS_PER_GROUP 256U

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

static gd_status gd_metal_amp_grad_elem_size(uint32_t dtype, size_t *out_elem_size)
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

static gd_status gd_metal_amp_validate_unscale_desc(const gd_backend_amp_unscale_desc *desc)
{
    size_t elem_size;
    size_t grad_nbytes;
    gd_status st;
    if (desc == NULL || desc->grad_buffer == NULL || desc->found_inf_buffer == NULL ||
        desc->count == 0U || desc->count > UINT32_MAX || !(desc->inv_scale > 0.0f)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_amp_grad_elem_size(desc->grad_dtype, &elem_size);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_metal_amp_count_bytes(desc->count, elem_size, &grad_nbytes) ||
        !gd_metal_amp_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
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
    args.found_inf_offset = (uint64_t)desc->found_inf_offset;
    args.count = (uint64_t)desc->count;
    args.grad_dtype = desc->grad_dtype;
    args.inv_scale = desc->inv_scale;
    [encoder setBuffer:gd_metal_amp_buffer(desc->grad_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(desc->found_inf_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
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
