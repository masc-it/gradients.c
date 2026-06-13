#include "../../backends/metal/metal_backend_internal.h"
#include "metal_concat_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLBuffer> gd_concat_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static size_t gd_concat_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_U8) {
        return 1U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16 || dtype == (uint32_t)GD_DTYPE_BF16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
        return 4U;
    }
    return 0U;
}

static id<MTLComputePipelineState> gd_concat_to_full_pso(gd_backend *backend, size_t elem_size)
{
    if (backend == NULL) {
        return nil;
    }
    if (elem_size == 1U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_to_full_u8_pso;
    }
    if (elem_size == 2U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_to_full_u16_pso;
    }
    if (elem_size == 4U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_to_full_u32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_concat_from_full_pso(gd_backend *backend, size_t elem_size)
{
    if (backend == NULL) {
        return nil;
    }
    if (elem_size == 1U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_from_full_u8_pso;
    }
    if (elem_size == 2U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_from_full_u16_pso;
    }
    if (elem_size == 4U) {
        return (__bridge id<MTLComputePipelineState>)backend->concat_from_full_u32_pso;
    }
    return nil;
}

static bool gd_concat_byte_range_valid(const gd_backend_buffer *buffer,
                                       size_t offset,
                                       size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_concat_count_bytes(uint64_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    *out_nbytes = (size_t)count * elem_size;
    return true;
}

static bool gd_concat_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    int32_t d;
    if (view == NULL || view->rank == 0U || view->rank > GD_MAX_DIMS) {
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

static bool gd_concat_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    size_t elem_size;
    if (view == NULL || view->buffer == NULL || view->count == 0U) {
        return false;
    }
    elem_size = gd_concat_dtype_size(view->dtype);
    if (!gd_concat_count_bytes((uint64_t)view->count, elem_size, &nbytes)) {
        return false;
    }
    return elem_size != 0U && view->offset % elem_size == 0U &&
           gd_concat_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_concat_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL) {
        return false;
    }
    if (a != 0U && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static gd_status gd_concat_validate_args(const gd_backend_tensor_view *slice,
                                         const gd_backend_tensor_view *full,
                                         const gd_backend_concat_args *args,
                                         bool slice_to_full,
                                         size_t *out_elem_size,
                                         uint64_t *out_full_count)
{
    size_t elem_size;
    uint64_t slice_block;
    uint64_t outer;
    uint64_t full_row;
    uint64_t full_count;
    if (slice == NULL || full == NULL || args == NULL || out_elem_size == NULL ||
        out_full_count == NULL || !gd_concat_view_range_valid(slice) ||
        !gd_concat_view_range_valid(full) || !gd_concat_contiguous_view(slice) ||
        !gd_concat_contiguous_view(full) || slice->dtype != full->dtype || args->count == 0U ||
        args->count > UINT32_MAX || args->inner == 0U || args->slice_axis == 0U ||
        args->full_axis == 0U || args->slice_axis > args->full_axis ||
        args->axis_offset > args->full_axis - args->slice_axis) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_concat_dtype_size(slice->dtype);
    if (elem_size == 0U) {
        return GD_ERR_UNSUPPORTED;
    }
    if (!gd_concat_u64_mul(args->slice_axis, args->inner, &slice_block) ||
        slice_block == 0U || args->count % slice_block != 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    outer = args->count / slice_block;
    if (!gd_concat_u64_mul(args->full_axis, args->inner, &full_row) ||
        !gd_concat_u64_mul(outer, full_row, &full_count) || full_count == 0U ||
        full_count > (uint64_t)SIZE_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (slice_to_full) {
        if (slice->count != (size_t)args->count || full->count != (size_t)full_count) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    } else if (slice->count != (size_t)full_count || full->count != (size_t)args->count) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_elem_size = elem_size;
    *out_full_count = full_count;
    return GD_OK;
}

static uint32_t gd_concat_threads_for_count(uint64_t count)
{
    if (count <= 64U) {
        return 64U;
    }
    if (count <= 128U) {
        return 128U;
    }
    return GD_METAL_CONCAT_THREADS;
}

static gd_status gd_concat_dispatch(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    const gd_backend_concat_args *args,
                                    bool to_full)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_concat_args metal_args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    uint64_t full_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_concat_validate_args(src, dst, args, to_full, &elem_size, &full_count);
    if (st != GD_OK) {
        return st;
    }
    pso = to_full ? gd_concat_to_full_pso(backend, elem_size) :
                    gd_concat_from_full_pso(backend, elem_size);
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
    memset(&metal_args, 0, sizeof(metal_args));
    metal_args.src_offset = (uint64_t)src->offset;
    metal_args.dst_offset = (uint64_t)dst->offset;
    metal_args.count = args->count;
    metal_args.inner = args->inner;
    metal_args.slice_axis = args->slice_axis;
    metal_args.full_axis = args->full_axis;
    metal_args.axis_offset = args->axis_offset;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_concat_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_concat_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)args->count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)gd_concat_threads_for_count(args->count), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    (void)full_count;
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_concat_to_full(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    const gd_backend_concat_args *args)
{
    return gd_concat_dispatch(backend, src, dst, args, true);
}

gd_status gd_backend_concat_from_full(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      const gd_backend_concat_args *args)
{
    return gd_concat_dispatch(backend, src, dst, args, false);
}
