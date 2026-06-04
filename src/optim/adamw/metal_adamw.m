#include "../../backends/metal/metal_backend_internal.h"
#include "metal_adamw_types.h"

#include <stdint.h>
#include <string.h>

#define GD_METAL_ADAMW_MAX_THREADS_PER_GROUP 256U

static id<MTLComputePipelineState> gd_metal_adamw_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->adamw_pso;
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

gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_adamw_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t param_elem_size;
    size_t grad_elem_size;
    size_t param_nbytes;
    size_t grad_nbytes;
    size_t moment_nbytes;
    gd_status st;
    if (desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (desc->param_dtype == 1U) {
        param_elem_size = 2U;
    } else if (desc->param_dtype == 3U) {
        param_elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (desc->grad_dtype == 1U) {
        grad_elem_size = 2U;
    } else if (desc->grad_dtype == 3U) {
        grad_elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend == NULL || desc->param_buffer == NULL || desc->grad_buffer == NULL ||
        desc->m_buffer == NULL || desc->v_buffer == NULL || desc->count == 0U ||
        desc->count > UINT32_MAX || !(desc->lr >= 0.0f) || !(desc->beta1 >= 0.0f) ||
        !(desc->beta1 < 1.0f) || !(desc->beta2 >= 0.0f) || !(desc->beta2 < 1.0f) ||
        !(desc->eps > 0.0f) || !(desc->weight_decay >= 0.0f) ||
        !(desc->bias_correction1 > 0.0f) || !(desc->bias_correction2 > 0.0f) ||
        !gd_metal_adamw_count_bytes(desc->count, param_elem_size, &param_nbytes) ||
        !gd_metal_adamw_count_bytes(desc->count, grad_elem_size, &grad_nbytes) ||
        !gd_metal_adamw_count_bytes(desc->count, 4U, &moment_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->param_buffer, desc->param_offset, param_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->grad_buffer, desc->grad_offset, grad_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->m_buffer, desc->m_offset, moment_nbytes) ||
        !gd_metal_adamw_byte_range_valid(desc->v_buffer, desc->v_offset, moment_nbytes) ||
        (desc->has_master != 0U &&
         (desc->master_buffer == NULL ||
          !gd_metal_adamw_byte_range_valid(desc->master_buffer, desc->master_offset, moment_nbytes)))) {
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
    args.param_offset = (uint64_t)desc->param_offset;
    args.master_offset = (uint64_t)desc->master_offset;
    args.grad_offset = (uint64_t)desc->grad_offset;
    args.m_offset = (uint64_t)desc->m_offset;
    args.v_offset = (uint64_t)desc->v_offset;
    args.count = (uint64_t)desc->count;
    args.param_dtype = desc->param_dtype;
    args.grad_dtype = desc->grad_dtype;
    args.has_master = desc->has_master != 0U ? 1U : 0U;
    args.lr = desc->lr;
    args.beta1 = desc->beta1;
    args.beta2 = desc->beta2;
    args.eps = desc->eps;
    args.weight_decay = desc->weight_decay;
    args.bias_correction1 = desc->bias_correction1;
    args.bias_correction2 = desc->bias_correction2;
    [encoder setComputePipelineState:gd_metal_adamw_pso(backend)];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->param_buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->grad_buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->m_buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->v_buffer) offset:0U atIndex:3U];
    [encoder setBuffer:gd_metal_adamw_buffer(desc->has_master != 0U ? desc->master_buffer : desc->param_buffer)
                offset:0U
               atIndex:4U];
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    grid = MTLSizeMake((NSUInteger)desc->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(desc->count < GD_METAL_ADAMW_MAX_THREADS_PER_GROUP ?
                                       desc->count : GD_METAL_ADAMW_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
