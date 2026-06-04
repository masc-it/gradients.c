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

gd_status gd_backend_amp_unscale(gd_backend *backend, const gd_backend_amp_unscale_desc *desc)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_amp_unscale_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    size_t grad_nbytes;
    gd_status st;
    if (backend == NULL || desc == NULL || desc->grad_buffer == NULL ||
        desc->found_inf_buffer == NULL || desc->count == 0U || desc->count > UINT32_MAX ||
        !(desc->inv_scale > 0.0f)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (desc->grad_dtype == 1U) {
        elem_size = 2U;
    } else if (desc->grad_dtype == 3U) {
        elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (!gd_metal_amp_count_bytes(desc->count, elem_size, &grad_nbytes) ||
        !gd_metal_amp_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
        !gd_metal_amp_byte_range_valid(desc->found_inf_buffer, desc->found_inf_offset, 4U)) {
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
    args.grad_offset = (uint64_t)desc->grad_offset;
    args.found_inf_offset = (uint64_t)desc->found_inf_offset;
    args.count = (uint64_t)desc->count;
    args.grad_dtype = desc->grad_dtype;
    args.inv_scale = desc->inv_scale;
    [encoder setComputePipelineState:gd_metal_amp_unscale_pso(backend)];
    [encoder setBuffer:gd_metal_amp_buffer(desc->grad_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_amp_buffer(desc->found_inf_buffer) offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)desc->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(desc->count < GD_METAL_AMP_MAX_THREADS_PER_GROUP ?
                                       desc->count : GD_METAL_AMP_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
