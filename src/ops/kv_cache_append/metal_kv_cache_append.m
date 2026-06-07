#include "../../backends/metal/metal_backend_internal.h"
#include "metal_kv_cache_append_types.h"

#include <gradients/tensor.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_kv_cache_append_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->kv_cache_append_pso;
}

static id<MTLBuffer> gd_kv_cache_append_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_kv_cache_append_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        *out_size = 2U;
        return true;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static bool gd_kv_cache_append_byte_range_valid(const gd_backend_buffer *buffer,
                                                size_t offset,
                                                size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_kv_cache_append_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_kv_cache_append_dtype_size(view->dtype, &elem_size) ||
        view->count > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = view->count * elem_size;
    return gd_kv_cache_append_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_kv_cache_append_contiguous_valid(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride) {
            return false;
        }
        if (view->shape[dim] != 0 && stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_kv_cache_append_shapes_valid(const gd_backend_tensor_view *k_cache,
                                            const gd_backend_tensor_view *v_cache,
                                            const gd_backend_tensor_view *k_new,
                                            const gd_backend_tensor_view *v_new,
                                            const gd_backend_kv_cache_append_args *args)
{
    uint32_t i;
    if (!gd_kv_cache_append_view_range_valid(k_cache) ||
        !gd_kv_cache_append_view_range_valid(v_cache) ||
        !gd_kv_cache_append_view_range_valid(k_new) ||
        !gd_kv_cache_append_view_range_valid(v_new) ||
        !gd_kv_cache_append_contiguous_valid(k_cache) ||
        !gd_kv_cache_append_contiguous_valid(v_cache) ||
        !gd_kv_cache_append_contiguous_valid(k_new) ||
        !gd_kv_cache_append_contiguous_valid(v_new) || args == NULL) {
        return false;
    }
    if ((k_cache->dtype != (uint32_t)GD_DTYPE_F16 && k_cache->dtype != (uint32_t)GD_DTYPE_F32) ||
        v_cache->dtype != k_cache->dtype || k_new->dtype != k_cache->dtype ||
        v_new->dtype != k_cache->dtype || k_cache->rank != 4U || v_cache->rank != 4U ||
        k_new->rank != 4U || v_new->rank != 4U || args->cache_pos < 0) {
        return false;
    }
    for (i = 0U; i < 4U; ++i) {
        if (k_cache->shape[i] != v_cache->shape[i] || k_new->shape[i] != v_new->shape[i]) {
            return false;
        }
    }
    return k_cache->shape[0] == k_new->shape[0] &&
           k_cache->shape[2] == k_new->shape[2] &&
           k_cache->shape[3] == k_new->shape[3] &&
           k_cache->shape[1] > 0 && k_new->shape[1] > 0 &&
           (int64_t)args->cache_pos <= k_cache->shape[1] &&
           k_new->shape[1] <= k_cache->shape[1] - (int64_t)args->cache_pos &&
           k_cache->shape[0] <= (int64_t)UINT32_MAX &&
           k_cache->shape[1] <= (int64_t)UINT32_MAX &&
           k_new->shape[1] <= (int64_t)UINT32_MAX &&
           k_cache->shape[2] <= (int64_t)UINT32_MAX &&
           k_cache->shape[3] <= (int64_t)UINT32_MAX;
}

static uint32_t gd_kv_cache_append_copy_unit(const gd_backend_tensor_view *k_cache,
                                            const gd_backend_tensor_view *v_cache,
                                            const gd_backend_tensor_view *k_new,
                                            const gd_backend_tensor_view *v_new,
                                            size_t row_bytes,
                                            size_t total_bytes)
{
    const size_t offsets_or = k_cache->offset | v_cache->offset | k_new->offset | v_new->offset;
    if (((offsets_or | row_bytes | total_bytes) & 15U) == 0U) {
        return 16U;
    }
    if (((offsets_or | row_bytes | total_bytes) & 3U) == 0U) {
        return 4U;
    }
    return 1U;
}

gd_status gd_backend_kv_cache_append_at(gd_backend *backend,
                                        const gd_backend_tensor_view *k_cache,
                                        const gd_backend_tensor_view *v_cache,
                                        const gd_backend_tensor_view *k_new,
                                        const gd_backend_tensor_view *v_new,
                                        const gd_backend_kv_cache_append_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_kv_cache_append_args p;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    size_t elem_size;
    size_t row_elems;
    size_t row_bytes;
    size_t total_rows;
    size_t total_bytes;
    uint32_t copy_unit;
    if (backend == NULL || !gd_kv_cache_append_shapes_valid(k_cache, v_cache, k_new, v_new, args) ||
        !gd_kv_cache_append_dtype_size(k_cache->dtype, &elem_size)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if ((uint64_t)k_cache->shape[2] > (uint64_t)(SIZE_MAX / (uint64_t)k_cache->shape[3])) {
        return GD_ERR_UNSUPPORTED;
    }
    row_elems = (size_t)((uint64_t)k_cache->shape[2] * (uint64_t)k_cache->shape[3]);
    if (row_elems > SIZE_MAX / elem_size) {
        return GD_ERR_UNSUPPORTED;
    }
    row_bytes = row_elems * elem_size;
    if ((uint64_t)k_new->shape[0] > (uint64_t)(SIZE_MAX / (uint64_t)k_new->shape[1])) {
        return GD_ERR_UNSUPPORTED;
    }
    total_rows = (size_t)((uint64_t)k_new->shape[0] * (uint64_t)k_new->shape[1]);
    if (total_rows > SIZE_MAX / row_bytes || row_bytes > (size_t)UINT32_MAX) {
        return GD_ERR_UNSUPPORTED;
    }
    total_bytes = total_rows * row_bytes;
    copy_unit = gd_kv_cache_append_copy_unit(k_cache, v_cache, k_new, v_new, row_bytes, total_bytes);
    if (copy_unit == 0U || total_bytes / copy_unit > (size_t)UINT32_MAX) {
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
    memset(&p, 0, sizeof(p));
    p.k_cache_offset = (uint64_t)k_cache->offset;
    p.v_cache_offset = (uint64_t)v_cache->offset;
    p.k_new_offset = (uint64_t)k_new->offset;
    p.v_new_offset = (uint64_t)v_new->offset;
    p.total_units = (uint64_t)(total_bytes / copy_unit);
    p.tmax = (uint32_t)k_cache->shape[1];
    p.tnew = (uint32_t)k_new->shape[1];
    p.row_bytes = (uint32_t)row_bytes;
    p.copy_unit = copy_unit;
    p.cache_pos = (uint32_t)args->cache_pos;

    [encoder setComputePipelineState:gd_kv_cache_append_pso(backend)];
    [encoder setBuffer:gd_kv_cache_append_buffer(k_cache->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_kv_cache_append_buffer(v_cache->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_kv_cache_append_buffer(k_new->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_kv_cache_append_buffer(v_new->buffer) offset:0U atIndex:3U];
    [encoder setBytes:&p length:sizeof(p) atIndex:4U];
    threads = MTLSizeMake(256U, 1U, 1U);
    groups = MTLSizeMake((NSUInteger)((p.total_units + 255U) / 256U), 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
