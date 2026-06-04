#ifndef GD_OP_COMMON_H
#define GD_OP_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/tensor.h>

#include "../core/backend.h"
#include "../core/memory_internal.h"

static inline bool gd_op_i64_to_u32(int64_t value, uint32_t *out)
{
    if (out == NULL || value <= 0 || value > (int64_t)UINT32_MAX) {
        return false;
    }
    *out = (uint32_t)value;
    return true;
}

static inline bool gd_op_matrix_view_from_tensor(const gd_tensor *tensor,
                                                 gd_backend_matrix_view *out)
{
    size_t elem_size;
    size_t row_bytes;
    size_t offset;
    uint32_t rows;
    uint32_t cols;
    if (tensor == NULL || out == NULL || tensor->rank != 2U ||
        tensor->strides[1] != 1 || tensor->strides[0] <= 0) {
        return false;
    }
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U || tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return false;
    }
    if ((uint64_t)tensor->strides[0] > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    if (!gd_op_i64_to_u32(tensor->shape[0], &rows) ||
        !gd_op_i64_to_u32(tensor->shape[1], &cols)) {
        return false;
    }
    row_bytes = (size_t)tensor->strides[0] * elem_size;
    if (row_bytes < (size_t)cols * elem_size) {
        return false;
    }
    offset = tensor->storage.offset + tensor->view_offset;
    out->buffer = (gd_backend_buffer *)tensor->storage.buffer;
    out->offset = offset;
    out->rows = rows;
    out->cols = cols;
    out->row_bytes = row_bytes;
    out->dtype = (uint32_t)tensor->dtype;
    return out->buffer != NULL;
}

static inline bool gd_op_tensor_view_from_tensor(const gd_tensor *tensor,
                                                 gd_backend_tensor_view *out)
{
    int64_t numel;
    size_t elem_size;
    size_t offset;
    uint32_t i;
    if (tensor == NULL || out == NULL || tensor->rank > GD_MAX_DIMS) {
        return false;
    }
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U || !gd_tensor_is_contiguous(tensor) ||
        tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return false;
    }
    numel = 1;
    for (i = 0U; i < tensor->rank; ++i) {
        if (tensor->shape[i] <= 0 || numel > INT64_MAX / tensor->shape[i]) {
            return false;
        }
        numel *= tensor->shape[i];
    }
    offset = tensor->storage.offset + tensor->view_offset;
    out->buffer = (gd_backend_buffer *)tensor->storage.buffer;
    out->offset = offset;
    out->count = (size_t)numel;
    out->dtype = (uint32_t)tensor->dtype;
    return out->buffer != NULL;
}

static inline bool gd_op_vector_view_from_tensor(const gd_tensor *tensor,
                                                 gd_backend_vector_view *out)
{
    size_t offset;
    uint32_t length;
    if (tensor == NULL || out == NULL || tensor->rank != 1U || tensor->strides[0] != 1) {
        return false;
    }
    if (gd_dtype_size(tensor->dtype) == 0U ||
        tensor->storage.offset > SIZE_MAX - tensor->view_offset ||
        !gd_op_i64_to_u32(tensor->shape[0], &length)) {
        return false;
    }
    offset = tensor->storage.offset + tensor->view_offset;
    out->buffer = (gd_backend_buffer *)tensor->storage.buffer;
    out->offset = offset;
    out->length = length;
    out->dtype = (uint32_t)tensor->dtype;
    return out->buffer != NULL;
}

#endif /* GD_OP_COMMON_H */
