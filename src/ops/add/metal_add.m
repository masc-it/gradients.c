#include "../_shared/binary/metal_binary_common.h"
#include "metal_add_types.h"

#define GD_ADD_F16_VECTOR_WIDTH ((NSUInteger)GD_METAL_ADD_F16_VECTOR_WIDTH)

static id<MTLComputePipelineState> gd_add_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_pso[GD_OP_ADD];
}

static id<MTLComputePipelineState> gd_add_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_bcast_pso[GD_OP_ADD];
}

static id<MTLComputePipelineState> gd_add_row_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_row_bcast_pso[GD_OP_ADD];
}

static NSUInteger gd_add_div_up_nsu(size_t value, NSUInteger divisor)
{
    return (NSUInteger)((value + (size_t)divisor - 1U) / (size_t)divisor);
}

static bool gd_add_view_is_f16(const gd_backend_tensor_view *view)
{
    return view != NULL && view->dtype == (uint32_t)GD_DTYPE_F16;
}

static gd_status gd_add_validate_direct_f16(const gd_backend_tensor_view *x,
                                            const gd_backend_tensor_view *y,
                                            const gd_backend_tensor_view *out)
{
    gd_status st = gd_metal_binary_validate_direct_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    return gd_add_view_is_f16(x) && gd_add_view_is_f16(y) && gd_add_view_is_f16(out) ? GD_OK :
                                                                                       GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_add(gd_backend *backend,
                         const gd_backend_tensor_view *x,
                         const gd_backend_tensor_view *y,
                         const gd_backend_tensor_view *out)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_binary_args args;
    gd_metal_binary_bcast_args bcast_args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool direct;
    bool row_broadcast;
    gd_status st;
    if (backend == NULL || gd_add_pso(backend) == nil || gd_add_bcast_pso(backend) == nil ||
        gd_add_row_bcast_pso(backend) == nil) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_binary_validate_broadcast_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_add_view_is_f16(x) || !gd_add_view_is_f16(y) || !gd_add_view_is_f16(out)) {
        return GD_ERR_UNSUPPORTED;
    }
    direct = x->count == out->count && y->count == out->count;
    row_broadcast = !direct && out->rank == 2U;
    if (direct) {
        st = gd_add_validate_direct_f16(x, y, out);
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
    if (direct) {
        memset(&args, 0, sizeof(args));
        args.x_offset = (uint64_t)x->offset;
        args.y_offset = (uint64_t)y->offset;
        args.out_offset = (uint64_t)out->offset;
        args.count = (uint64_t)out->count;
        args.dtype = out->dtype;
        [encoder setComputePipelineState:gd_add_pso(backend)];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&args length:sizeof(args) atIndex:3U];
        grid = MTLSizeMake(gd_add_div_up_nsu(out->count, GD_ADD_F16_VECTOR_WIDTH), 1U, 1U);
        threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                   GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                              1U,
                              1U);
    } else {
        gd_metal_binary_fill_bcast_args(&bcast_args, x, y, out);
        [encoder setComputePipelineState:row_broadcast ? gd_add_row_bcast_pso(backend) : gd_add_bcast_pso(backend)];
        [encoder setBuffer:gd_metal_binary_buffer(x->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_metal_binary_buffer(y->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_metal_binary_buffer(out->buffer) offset:0U atIndex:2U];
        [encoder setBytes:&bcast_args length:sizeof(bcast_args) atIndex:3U];
        if (row_broadcast) {
            grid = MTLSizeMake(gd_add_div_up_nsu((size_t)out->shape[1], GD_ADD_F16_VECTOR_WIDTH),
                               (NSUInteger)out->shape[0],
                               1U);
            threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                       GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                                  1U,
                                  1U);
        } else {
            grid = MTLSizeMake(gd_add_div_up_nsu(out->count, GD_ADD_F16_VECTOR_WIDTH), 1U, 1U);
            threads = MTLSizeMake(grid.width < GD_METAL_BINARY_MAX_THREADS_PER_GROUP ? grid.width :
                                                                                       GD_METAL_BINARY_MAX_THREADS_PER_GROUP,
                                  1U,
                                  1U);
        }
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
