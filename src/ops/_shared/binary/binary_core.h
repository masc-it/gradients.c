#ifndef GD_OPS_SHARED_BINARY_CORE_H
#define GD_OPS_SHARED_BINARY_CORE_H

#include <gradients/ops.h>

#include "../../autograd_impl.h"
#include "../../op_common.h"

#include <stddef.h>
#include <string.h>

typedef gd_status (*gd_binary_backend_fn)(gd_backend *backend,
                                          const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *y,
                                          const gd_backend_tensor_view *out);

static inline bool gd_binary_shapes_equal(const gd_tensor *x, const gd_tensor *y)
{
    uint32_t i;
    if (x == NULL || y == NULL || x->rank != y->rank) {
        return false;
    }
    for (i = 0U; i < x->rank; ++i) {
        if (x->shape[i] != y->shape[i]) {
            return false;
        }
    }
    return true;
}

static inline bool gd_binary_tensor_matches_shape(const gd_tensor *tensor,
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

static inline gd_status gd_binary_count_tensor(gd_context *ctx,
                                               const gd_tensor *tensor,
                                               size_t *out_count)
{
    int64_t numel;
    gd_status st;
    if (ctx == NULL || tensor == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "binary op invalid tensor shape");
    }
    if ((uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "binary op tensor count overflow");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static inline gd_status gd_binary_broadcast_shape(gd_context *ctx,
                                                  const gd_tensor *x,
                                                  const gd_tensor *y,
                                                  uint32_t *out_rank,
                                                  int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t rank;
    uint32_t i;
    if (ctx == NULL || x == NULL || y == NULL || out_rank == NULL || out_shape == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->rank > GD_MAX_DIMS || y->rank > GD_MAX_DIMS) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "binary op invalid rank");
    }
    rank = x->rank > y->rank ? x->rank : y->rank;
    for (i = 0U; i < rank; ++i) {
        int64_t xd = 1;
        int64_t yd = 1;
        if (i + x->rank >= rank) {
            xd = x->shape[i + x->rank - rank];
        }
        if (i + y->rank >= rank) {
            yd = y->shape[i + y->rank - rank];
        }
        if (xd <= 0 || yd <= 0 || (xd != yd && xd != 1 && yd != 1)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "binary op shapes are not broadcast-compatible");
        }
        out_shape[i] = xd > yd ? xd : yd;
    }
    *out_rank = rank;
    return GD_OK;
}

static inline gd_status gd_binary_validate_inputs(gd_context *ctx,
                                                  const gd_tensor *x,
                                                  const gd_tensor *y,
                                                  uint32_t *out_rank,
                                                  int64_t out_shape[GD_MAX_DIMS])
{
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, y);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 && x->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "binary op currently supports f16/f32 tensors only");
    }
    if (x->dtype != y->dtype) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "binary op dtype mismatch");
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(y)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "binary op requires contiguous tensors");
    }
    return gd_binary_broadcast_shape(ctx, x, y, out_rank, out_shape);
}

static inline gd_status gd_binary_validate_backward(gd_context *ctx,
                                                    const gd_tensor *x,
                                                    const gd_tensor *y,
                                                    const gd_tensor *grad_out)
{
    gd_status st;
    uint32_t out_rank;
    int64_t out_shape[GD_MAX_DIMS];
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_binary_validate_inputs(ctx, x, y, &out_rank, out_shape);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != x->dtype || !gd_binary_tensor_matches_shape(grad_out, out_rank, out_shape)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "binary backward gradient shape/dtype mismatch");
    }
    if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "binary backward requires contiguous grad_out");
    }
    return GD_OK;
}

static inline gd_status gd_binary_apply_impl(gd_context *ctx,
                                             const gd_tensor *x,
                                             const gd_tensor *y,
                                             gd_tensor *out,
                                             gd_binary_backend_fn backend_fn,
                                             gd_op_kind op_kind,
                                             const char *op_name,
                                             bool record)
{
    gd_status st;
    gd_tensor result;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view ov;
    uint32_t out_rank;
    int64_t out_shape[GD_MAX_DIMS];
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || y == NULL || out == NULL || backend_fn == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_binary_validate_inputs(ctx, x, y, &out_rank, out_shape);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(out_rank, out_shape), 256U, &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(&result, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "binary op invalid tensor view");
    }
    st = backend_fn(gd_context_backend(ctx), &xv, &yv, &ov);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend binary op failed");
    }
    if (record) {
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        inputs[0] = x;
        inputs[1] = y;
        outputs[0] = &result;
        st = gd_autograd_record(ctx,
                                op_kind,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    (void)op_name;
    *out = result;
    return GD_OK;
}

static inline gd_status gd_binary_scale_to_shape(gd_context *ctx,
                                                 const gd_tensor *src,
                                                 const gd_tensor *shape_like,
                                                 float scale,
                                                 gd_tensor *out)
{
    gd_status st;
    gd_tensor result;
    gd_backend_tensor_view sv;
    size_t src_count;
    size_t dst_count;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || src == NULL || shape_like == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, shape_like);
    if (st != GD_OK) {
        return st;
    }
    if (src->dtype != shape_like->dtype ||
        (src->dtype != GD_DTYPE_F16 && src->dtype != GD_DTYPE_F32) ||
        !gd_tensor_is_contiguous(src) || !gd_tensor_is_contiguous(shape_like) || !(scale == scale)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "binary backward scale requires contiguous f16/f32 tensors");
    }
    st = gd_binary_count_tensor(ctx, src, &src_count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_binary_count_tensor(ctx, shape_like, &dst_count);
    if (st != GD_OK) {
        return st;
    }
    if (src_count != dst_count) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "binary backward flat scale count mismatch");
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, shape_like->dtype, gd_shape_make(shape_like->rank, shape_like->shape), 256U, &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(src, &sv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "binary backward invalid tensor view");
    }
    st = gd_backend_scale(gd_context_backend(ctx),
                          (gd_backend_buffer *)result.storage.buffer,
                          gd_tensor_storage_offset(&result),
                          (gd_backend_buffer *)src->storage.buffer,
                          gd_tensor_storage_offset(src),
                          sv.count,
                          (uint32_t)src->dtype,
                          scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend binary backward scale failed");
    }
    *out = result;
    return GD_OK;
}

static inline gd_status gd_binary_sum_to_shape(gd_context *ctx,
                                               const gd_tensor *src,
                                               const gd_tensor *shape_like,
                                               float scale,
                                               gd_tensor *out)
{
    gd_status st;
    gd_tensor result;
    gd_backend_tensor_view sv;
    gd_backend_tensor_view dv;
    size_t src_count;
    size_t dst_count;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || src == NULL || shape_like == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, shape_like);
    if (st != GD_OK) {
        return st;
    }
    if (src->dtype != shape_like->dtype ||
        (src->dtype != GD_DTYPE_F16 && src->dtype != GD_DTYPE_F32) ||
        !gd_tensor_is_contiguous(src) || !gd_tensor_is_contiguous(shape_like) || !(scale == scale)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "binary sum_to_shape requires contiguous f16/f32 tensors");
    }
    st = gd_binary_count_tensor(ctx, src, &src_count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_binary_count_tensor(ctx, shape_like, &dst_count);
    if (st != GD_OK) {
        return st;
    }
    if (src_count == dst_count) {
        return gd_binary_scale_to_shape(ctx, src, shape_like, scale, out);
    }
    if (shape_like->rank > src->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "binary sum_to_shape rank mismatch");
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, shape_like->dtype, gd_shape_make(shape_like->rank, shape_like->shape), 256U, &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(src, &sv) || !gd_op_tensor_view_from_tensor(&result, &dv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "binary sum_to_shape invalid tensor view");
    }
    st = gd_backend_reduce_broadcast(gd_context_backend(ctx), &sv, &dv, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend binary sum_to_shape failed");
    }
    *out = result;
    return GD_OK;
}

#endif /* GD_OPS_SHARED_BINARY_CORE_H */
