#ifndef GD_OPS_LINEAR_IMPL_H
#define GD_OPS_LINEAR_IMPL_H

#include <gradients/ops.h>

#include "../op_common.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct gd_linear_shape_info {
    uint32_t rank;
    int64_t out_shape[GD_MAX_DIMS];
    int64_t rows;
    int64_t k;
    int64_t n;
} gd_linear_shape_info;

static inline bool gd_linear_i64_mul_overflow(int64_t a, int64_t b, int64_t *out)
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

static inline gd_status gd_linear_build_shape_info(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   const gd_tensor *w,
                                                   const gd_tensor *bias,
                                                   gd_linear_shape_info *info)
{
    uint32_t axis;
    int64_t rows = 1;
    if (ctx == NULL || x == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(info, 0, sizeof(*info));
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 ||
        (bias != NULL && bias->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear currently supports f16 tensors only");
    }
    if (x->rank < 1U || x->rank > GD_MAX_DIMS || w->rank != 2U ||
        (bias != NULL && bias->rank != 1U)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "linear expects x [..., K], w [K, N], bias [N]");
    }
    info->rank = x->rank;
    info->k = x->shape[x->rank - 1U];
    info->n = w->shape[1];
    if (info->k <= 0 || info->n <= 0 || w->shape[0] <= 0 || info->k != w->shape[0] ||
        (bias != NULL && bias->shape[0] != info->n)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear shape mismatch");
    }
    if (info->k > (int64_t)UINT32_MAX || info->n > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "linear matrix dimension overflow");
    }
    for (axis = 0U; axis + 1U < x->rank; ++axis) {
        if (x->shape[axis] <= 0 || gd_linear_i64_mul_overflow(rows, x->shape[axis], &rows)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "linear flattened row count overflow");
        }
        info->out_shape[axis] = x->shape[axis];
    }
    if (rows > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "linear flattened row count overflow");
    }
    info->out_shape[x->rank - 1U] = info->n;
    info->rows = rows;
    return GD_OK;
}

static inline bool gd_linear_shape_matches(const gd_tensor *tensor,
                                           uint32_t rank,
                                           const int64_t *shape)
{
    uint32_t axis;
    if (tensor == NULL || shape == NULL || tensor->rank != rank) {
        return false;
    }
    for (axis = 0U; axis < rank; ++axis) {
        if (tensor->shape[axis] != shape[axis]) {
            return false;
        }
    }
    return true;
}

static inline gd_status gd_linear_validate_layout(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   const gd_tensor *w,
                                                   const gd_tensor *bias,
                                                   const char *row_message,
                                                   const char *rank_message)
{
    if (ctx == NULL || x == NULL || w == NULL || row_message == NULL || rank_message == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->strides[x->rank - 1U] != 1 || w->strides[1] != 1 || w->strides[0] <= 0 ||
        (bias != NULL && bias->strides[0] != 1)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, row_message);
    }
    if (x->rank == 2U) {
        if (x->strides[0] <= 0) {
            return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, row_message);
        }
    } else if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, rank_message);
    }
    return GD_OK;
}

static inline gd_status gd_linear_validate_common(gd_context *ctx,
                                                  const gd_tensor *x,
                                                  const gd_tensor *w,
                                                  const gd_tensor *bias,
                                                  gd_linear_shape_info *info)
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
    if (bias != NULL) {
        st = gd_tensor_validate(ctx, bias);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_linear_build_shape_info(ctx, x, w, bias, info);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_validate_layout(ctx,
                                     x,
                                     w,
                                     bias,
                                     "linear requires row-strided inputs",
                                     "linear rank-N input must be contiguous");
}

static inline gd_status gd_linear_transposed_weight_build_shape_info(gd_context *ctx,
                                                                     const gd_tensor *x,
                                                                     const gd_tensor *w,
                                                                     const gd_tensor *bias,
                                                                     gd_linear_shape_info *info)
{
    uint32_t axis;
    int64_t rows = 1;
    if (ctx == NULL || x == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(info, 0, sizeof(*info));
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 ||
        (bias != NULL && bias->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear_transposed_weight currently supports f16 tensors only");
    }
    if (x->rank < 1U || x->rank > GD_MAX_DIMS || w->rank != 2U ||
        (bias != NULL && bias->rank != 1U)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "linear_transposed_weight expects x [..., K], w [N, K], bias [N]");
    }
    info->rank = x->rank;
    info->k = x->shape[x->rank - 1U];
    info->n = w->shape[0];
    if (info->k <= 0 || info->n <= 0 || w->shape[1] <= 0 || info->k != w->shape[1] ||
        (bias != NULL && bias->shape[0] != info->n)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "linear_transposed_weight shape mismatch");
    }
    if (info->k > (int64_t)UINT32_MAX || info->n > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "linear_transposed_weight matrix dimension overflow");
    }
    for (axis = 0U; axis + 1U < x->rank; ++axis) {
        if (x->shape[axis] <= 0 || gd_linear_i64_mul_overflow(rows, x->shape[axis], &rows)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "linear_transposed_weight flattened row count overflow");
        }
        info->out_shape[axis] = x->shape[axis];
    }
    if (rows > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "linear_transposed_weight flattened row count overflow");
    }
    info->out_shape[x->rank - 1U] = info->n;
    info->rows = rows;
    return GD_OK;
}

static inline gd_status gd_linear_transposed_weight_validate_common(gd_context *ctx,
                                                                    const gd_tensor *x,
                                                                    const gd_tensor *w,
                                                                    const gd_tensor *bias,
                                                                    gd_linear_shape_info *info)
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
    if (bias != NULL) {
        st = gd_tensor_validate(ctx, bias);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_linear_transposed_weight_build_shape_info(ctx, x, w, bias, info);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_validate_layout(ctx,
                                     x,
                                     w,
                                     bias,
                                     "linear_transposed_weight requires row-strided inputs",
                                     "linear_transposed_weight rank-N input must be contiguous");
}

static inline gd_status gd_linear_validate_grad_out(gd_context *ctx,
                                                    const gd_tensor *grad_out,
                                                    const gd_linear_shape_info *info)
{
    gd_status st;
    if (ctx == NULL || grad_out == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear backward currently supports f16 tensors only");
    }
    if (!gd_linear_shape_matches(grad_out, info->rank, info->out_shape)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward shape mismatch");
    }
    if (grad_out->strides[grad_out->rank - 1U] != 1) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear backward requires row-strided grad_out");
    }
    if (grad_out->rank == 2U) {
        if (grad_out->strides[0] <= 0) {
            return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                        "linear backward requires row-strided grad_out");
        }
    } else if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear backward rank-N grad_out must be contiguous");
    }
    return GD_OK;
}

static inline bool gd_linear_flat_matrix_view_from_tensor(const gd_tensor *tensor,
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
    uint32_t axis;
    if (tensor == NULL || out == NULL || rows_i64 <= 0 || cols_i64 <= 0) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (tensor->rank == 2U && tensor->shape[0] == rows_i64 && tensor->shape[1] == cols_i64) {
        return gd_op_matrix_view_from_tensor(tensor, out);
    }
    elem_size = gd_dtype_size(tensor->dtype);
    for (axis = 0U; axis < tensor->rank; ++axis) {
        if (tensor->shape[axis] <= 0 ||
            gd_linear_i64_mul_overflow(tensor_count, tensor->shape[axis], &tensor_count)) {
            return false;
        }
    }
    if (gd_linear_i64_mul_overflow(rows_i64, cols_i64, &matrix_count) ||
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

#endif /* GD_OPS_LINEAR_IMPL_H */
