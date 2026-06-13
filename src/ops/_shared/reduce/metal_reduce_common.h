#ifndef GD_OPS_SHARED_REDUCE_METAL_COMMON_H
#define GD_OPS_SHARED_REDUCE_METAL_COMMON_H

#include "../../../backends/metal/metal_backend_internal.h"
#include "metal_reduce_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

#define GD_METAL_REDUCE_MAX_THREADS_PER_GROUP 256U

static inline id<MTLBuffer> gd_metal_reduce_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static inline bool gd_metal_reduce_byte_range_valid(const gd_backend_buffer *buffer,
                                                    size_t offset,
                                                    size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static inline bool gd_metal_reduce_count_bytes(size_t count,
                                               uint32_t dtype,
                                               size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        elem_size = 2U;
    } else if (dtype == (uint32_t)GD_DTYPE_F32) {
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

static inline bool gd_metal_reduce_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_metal_reduce_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_metal_reduce_byte_range_valid(view->buffer, view->offset, nbytes);
}

static inline bool gd_metal_broadcast_dim(const gd_backend_tensor_view *src,
                                          const gd_backend_tensor_view *dst,
                                          uint32_t dim,
                                          uint64_t *out_stride)
{
    uint32_t prefix;
    int64_t src_dim;
    int64_t dst_dim;
    int64_t stride;
    if (src == NULL || dst == NULL || out_stride == NULL || dim >= dst->rank ||
        dst->rank > GD_MAX_DIMS || src->rank > dst->rank) {
        return false;
    }
    prefix = dst->rank - src->rank;
    dst_dim = dst->shape[dim];
    if (dst_dim <= 0) {
        return false;
    }
    if (dim < prefix) {
        *out_stride = 0U;
        return true;
    }
    src_dim = src->shape[dim - prefix];
    stride = src->strides[dim - prefix];
    if (src_dim == dst_dim && stride >= 0) {
        *out_stride = (uint64_t)stride;
        return true;
    }
    if (src_dim == 1) {
        *out_stride = 0U;
        return true;
    }
    return false;
}

#endif /* GD_OPS_SHARED_REDUCE_METAL_COMMON_H */
