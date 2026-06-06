#ifndef GD_OPS_MATMUL_IMPL_H
#define GD_OPS_MATMUL_IMPL_H

#include <gradients/ops.h>

#include "../op_common.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GD_MATMUL_MAX_BATCH_DIMS (GD_MAX_DIMS - 2U)

typedef struct gd_matmul_shape_info {
    uint32_t batch_rank;
    uint32_t out_rank;
    uint32_t x_batch_rank;
    uint32_t w_batch_rank;
    uint32_t batch_count;
    int64_t batch_shape[GD_MATMUL_MAX_BATCH_DIMS];
    int64_t out_shape[GD_MAX_DIMS];
    int64_t m;
    int64_t k;
    int64_t n;
} gd_matmul_shape_info;

static inline bool gd_matmul_i64_mul_overflow(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL || a < 0 || b < 0) {
        return true;
    }
    if (a != 0 && b > INT64_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static inline int64_t gd_matmul_aligned_batch_dim(const gd_tensor *tensor,
                                                  uint32_t batch_rank,
                                                  uint32_t out_batch_rank,
                                                  uint32_t out_axis)
{
    uint32_t pad;
    if (tensor == NULL || out_axis >= out_batch_rank || batch_rank > out_batch_rank) {
        return 1;
    }
    pad = out_batch_rank - batch_rank;
    if (out_axis < pad) {
        return 1;
    }
    return tensor->shape[out_axis - pad];
}

static inline gd_status gd_matmul_build_shape_info(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   const gd_tensor *w,
                                                   gd_matmul_shape_info *info)
{
    uint32_t axis;
    int64_t batch_count_i64 = 1;
    if (ctx == NULL || x == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(info, 0, sizeof(*info));
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "matmul currently supports f16 tensors only");
    }
    if (x->rank < 2U || w->rank < 2U || x->rank > GD_MAX_DIMS || w->rank > GD_MAX_DIMS) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "matmul expects tensors with rank >= 2");
    }
    info->x_batch_rank = x->rank - 2U;
    info->w_batch_rank = w->rank - 2U;
    info->batch_rank = info->x_batch_rank > info->w_batch_rank ? info->x_batch_rank : info->w_batch_rank;
    if (info->batch_rank > GD_MATMUL_MAX_BATCH_DIMS) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul batch rank exceeds backend limit");
    }
    info->m = x->shape[x->rank - 2U];
    info->k = x->shape[x->rank - 1U];
    info->n = w->shape[w->rank - 1U];
    if (info->m <= 0 || info->k <= 0 || info->n <= 0 || w->shape[w->rank - 2U] <= 0 ||
        info->k != w->shape[w->rank - 2U]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul shape mismatch");
    }
    for (axis = 0U; axis < info->batch_rank; ++axis) {
        int64_t x_dim = gd_matmul_aligned_batch_dim(x, info->x_batch_rank, info->batch_rank, axis);
        int64_t w_dim = gd_matmul_aligned_batch_dim(w, info->w_batch_rank, info->batch_rank, axis);
        int64_t out_dim;
        if (x_dim <= 0 || w_dim <= 0) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul batch dimensions must be positive");
        }
        if (x_dim != w_dim && x_dim != 1 && w_dim != 1) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul batch dimensions are not broadcastable");
        }
        out_dim = x_dim > w_dim ? x_dim : w_dim;
        if (out_dim > (int64_t)UINT32_MAX ||
            gd_matmul_i64_mul_overflow(batch_count_i64, out_dim, &batch_count_i64) ||
            batch_count_i64 > (int64_t)UINT32_MAX) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "matmul batch size overflow");
        }
        info->batch_shape[axis] = out_dim;
        info->out_shape[axis] = out_dim;
    }
    info->out_shape[info->batch_rank] = info->m;
    info->out_shape[info->batch_rank + 1U] = info->n;
    info->out_rank = info->batch_rank + 2U;
    info->batch_count = (uint32_t)batch_count_i64;
    return GD_OK;
}

static inline gd_status gd_matmul_validate_common(gd_context *ctx,
                                                  const gd_tensor *x,
                                                  const gd_tensor *w,
                                                  gd_matmul_shape_info *info)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, w);
    if (st != GD_OK) {
        return st;
    }
    st = gd_matmul_build_shape_info(ctx, x, w, info);
    if (st != GD_OK) {
        return st;
    }
    if (x->strides[x->rank - 1U] != 1 || w->strides[w->rank - 1U] != 1 ||
        x->strides[x->rank - 2U] <= 0 || w->strides[w->rank - 2U] <= 0) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "matmul requires row-strided inputs");
    }
    return GD_OK;
}

static inline bool gd_matmul_shape_matches(const gd_tensor *tensor,
                                           uint32_t rank,
                                           const int64_t *shape)
{
    uint32_t i;
    if (tensor == NULL || shape == NULL || tensor->rank != rank) {
        return false;
    }
    for (i = 0U; i < rank; ++i) {
        if (tensor->shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

static inline bool gd_matmul_stride_bytes(int64_t stride, size_t elem_size, size_t *out)
{
    if (out == NULL || stride < 0 || (uint64_t)stride > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    *out = (size_t)stride * elem_size;
    return true;
}

static inline bool gd_matmul_flat_matrix_view_from_tensor(const gd_tensor *tensor,
                                                          int64_t rows_i64,
                                                          int64_t cols_i64,
                                                          gd_backend_matrix_view *out)
{
    size_t elem_size;
    size_t offset;
    uint32_t rows;
    uint32_t cols;
    int64_t tensor_count = 1;
    int64_t matrix_count;
    uint32_t dim;
    if (tensor == NULL || out == NULL || rows_i64 <= 0 || cols_i64 <= 0) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    elem_size = gd_dtype_size(tensor->dtype);
    for (dim = 0U; dim < tensor->rank; ++dim) {
        if (tensor->shape[dim] <= 0 || gd_matmul_i64_mul_overflow(tensor_count, tensor->shape[dim], &tensor_count)) {
            return false;
        }
    }
    if (gd_matmul_i64_mul_overflow(rows_i64, cols_i64, &matrix_count) ||
        matrix_count != tensor_count || elem_size == 0U || !gd_tensor_is_contiguous(tensor) ||
        tensor->storage.offset > SIZE_MAX - tensor->view_offset ||
        !gd_op_i64_to_u32(rows_i64, &rows) || !gd_op_i64_to_u32(cols_i64, &cols)) {
        return false;
    }
    offset = tensor->storage.offset + tensor->view_offset;
    out->buffer = (gd_backend_buffer *)tensor->storage.buffer;
    out->offset = offset;
    out->rows = rows;
    out->cols = cols;
    out->row_bytes = (size_t)cols * elem_size;
    out->dtype = (uint32_t)tensor->dtype;
    return out->buffer != NULL;
}

static inline bool gd_matmul_batched_matrix_view_from_tensor(const gd_tensor *tensor,
                                                             const gd_matmul_shape_info *info,
                                                             gd_backend_batched_matrix_view *out)
{
    size_t elem_size;
    size_t offset;
    size_t row_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t tensor_batch_rank;
    uint32_t pad;
    uint32_t axis;
    if (tensor == NULL || info == NULL || out == NULL || tensor->rank < 2U ||
        tensor->rank > GD_MAX_DIMS || info->batch_rank > GD_BACKEND_MAX_BATCH_DIMS) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U || tensor->storage.offset > SIZE_MAX - tensor->view_offset ||
        tensor->strides[tensor->rank - 1U] != 1 || tensor->strides[tensor->rank - 2U] <= 0) {
        return false;
    }
    if (!gd_matmul_stride_bytes(tensor->strides[tensor->rank - 2U], elem_size, &row_bytes) ||
        !gd_op_i64_to_u32(tensor->shape[tensor->rank - 2U], &rows) ||
        !gd_op_i64_to_u32(tensor->shape[tensor->rank - 1U], &cols)) {
        return false;
    }
    tensor_batch_rank = tensor->rank - 2U;
    if (tensor_batch_rank > info->batch_rank) {
        return false;
    }
    pad = info->batch_rank - tensor_batch_rank;
    offset = tensor->storage.offset + tensor->view_offset;
    out->buffer = (gd_backend_buffer *)tensor->storage.buffer;
    out->offset = offset;
    out->rows = rows;
    out->cols = cols;
    out->row_bytes = row_bytes;
    out->dtype = (uint32_t)tensor->dtype;
    out->batch_rank = info->batch_rank;
    out->batch_count = info->batch_count;
    for (axis = 0U; axis < info->batch_rank; ++axis) {
        out->batch_shape[axis] = (uint32_t)info->batch_shape[axis];
        if (axis < pad) {
            out->batch_strides[axis] = 0U;
        } else {
            uint32_t tensor_axis = axis - pad;
            if (tensor->shape[tensor_axis] == info->batch_shape[axis]) {
                if (!gd_matmul_stride_bytes(tensor->strides[tensor_axis], elem_size,
                                            &out->batch_strides[axis])) {
                    return false;
                }
            } else if (tensor->shape[tensor_axis] == 1) {
                out->batch_strides[axis] = 0U;
            } else {
                return false;
            }
        }
    }
    return out->buffer != NULL;
}

static inline bool gd_matmul_operand_needs_batch_reduce(const gd_tensor *operand,
                                                        const gd_matmul_shape_info *info)
{
    uint32_t operand_batch_rank;
    uint32_t pad;
    uint32_t axis;
    if (operand == NULL || info == NULL || operand->rank < 2U ||
        operand->rank - 2U > info->batch_rank) {
        return true;
    }
    operand_batch_rank = operand->rank - 2U;
    pad = info->batch_rank - operand_batch_rank;
    for (axis = 0U; axis < info->batch_rank; ++axis) {
        if (axis < pad) {
            if (info->batch_shape[axis] != 1) {
                return true;
            }
        } else {
            int64_t dim = operand->shape[axis - pad];
            if (dim == 1 && info->batch_shape[axis] != 1) {
                return true;
            }
        }
    }
    return false;
}

static inline gd_status gd_matmul_reduce_batch_gradient(gd_context *ctx,
                                                        const gd_tensor *partial,
                                                        const gd_tensor *operand,
                                                        const gd_matmul_shape_info *info,
                                                        gd_tensor *out)
{
    gd_status st;
    gd_tensor current;
    gd_tensor reduced;
    uint32_t operand_batch_rank;
    uint32_t pad;
    uint32_t axis;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || partial == NULL || operand == NULL || info == NULL || out == NULL ||
        operand->rank < 2U || operand->rank - 2U > info->batch_rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    current = *partial;
    operand_batch_rank = operand->rank - 2U;
    pad = info->batch_rank - operand_batch_rank;
    for (axis = 0U; axis < info->batch_rank; ++axis) {
        bool reduce_axis;
        if (axis < pad) {
            reduce_axis = info->batch_shape[axis] != 1;
        } else {
            reduce_axis = operand->shape[axis - pad] == 1 && info->batch_shape[axis] != 1;
        }
        if (reduce_axis) {
            st = gd_reduce_sum_axis(ctx, &current, (int32_t)axis, true, &reduced);
            if (st != GD_OK) {
                return st;
            }
            current = reduced;
        }
    }
    if (gd_matmul_shape_matches(&current, operand->rank, operand->shape)) {
        *out = current;
        return GD_OK;
    }
    return gd_reshape(ctx, &current, gd_shape_make(operand->rank, operand->shape), out);
}

#endif /* GD_OPS_MATMUL_IMPL_H */
