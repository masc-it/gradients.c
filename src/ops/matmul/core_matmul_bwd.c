#include "matmul_impl.h"

#include <string.h>

static gd_status gd_matmul_backward_validate(gd_context *ctx,
                                             const gd_tensor *x,
                                             const gd_tensor *w,
                                             const gd_tensor *grad_out,
                                             bool need_grad_x,
                                             bool need_grad_w,
                                             gd_matmul_shape_info *info)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || grad_out == NULL || info == NULL ||
        (!need_grad_x && !need_grad_w)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_matmul_validate_common(ctx, x, w, info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "matmul backward currently supports f16 tensors only");
    }
    if (!gd_matmul_shape_matches(grad_out, info->out_rank, info->out_shape)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward shape mismatch");
    }
    if (grad_out->strides[grad_out->rank - 1U] != 1 ||
        grad_out->strides[grad_out->rank - 2U] <= 0) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "matmul backward requires row-strided grad_out");
    }
    return GD_OK;
}

static gd_status gd_matmul_make_partial(gd_context *ctx,
                                        const gd_matmul_shape_info *info,
                                        int64_t rows,
                                        int64_t cols,
                                        gd_tensor *out)
{
    int64_t shape[GD_MAX_DIMS];
    uint32_t axis;
    if (ctx == NULL || info == NULL || out == NULL || rows <= 0 || cols <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (axis = 0U; axis < info->batch_rank; ++axis) {
        shape[axis] = info->batch_shape[axis];
    }
    shape[info->batch_rank] = rows;
    shape[info->batch_rank + 1U] = cols;
    return gd_tensor_empty(ctx,
                           GD_ARENA_SCRATCH,
                           GD_DTYPE_F16,
                           gd_shape_make(info->out_rank, shape),
                           256U,
                           out);
}

gd_status gd_matmul_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w)
{
    gd_status st;
    gd_matmul_shape_info info;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor dx_partial;
    gd_tensor dw_partial;
    gd_backend_batched_matrix_view xv;
    gd_backend_batched_matrix_view wv;
    gd_backend_batched_matrix_view gv;
    gd_backend_batched_matrix_view dxv;
    gd_backend_batched_matrix_view dwv;
    gd_backend_matrix_view x_flat;
    gd_backend_matrix_view w_flat;
    gd_backend_matrix_view g_flat;
    gd_backend_matrix_view dx_flat;
    gd_backend_matrix_view dw_flat;
    int64_t flat_rows;
    bool need_grad_x = grad_x != NULL;
    bool need_grad_w = grad_w != NULL;
    bool reduce_x;
    bool reduce_w;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    memset(&dx_partial, 0, sizeof(dx_partial));
    memset(&dw_partial, 0, sizeof(dw_partial));
    st = gd_matmul_backward_validate(ctx, x, w, grad_out, need_grad_x, need_grad_w, &info);
    if (st != GD_OK) {
        return st;
    }
    if (gd_matmul_i64_mul_overflow((int64_t)info.batch_count, info.m, &flat_rows)) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "matmul backward flattened row count overflow");
    }
    if (w->rank == 2U && gd_tensor_is_contiguous(x) && gd_tensor_is_contiguous(grad_out) &&
        flat_rows <= (int64_t)UINT32_MAX) {
        if (!gd_matmul_flat_matrix_view_from_tensor(grad_out, flat_rows, info.n, &g_flat) ||
            !gd_op_matrix_view_from_tensor(w, &w_flat)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "matmul backward invalid flattened input view");
        }
        if (need_grad_x) {
            st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16,
                                 gd_shape_make(x->rank, x->shape), 256U, &dx);
            if (st != GD_OK) {
                return st;
            }
            dx.is_leaf = false;
            if (!gd_matmul_flat_matrix_view_from_tensor(&dx, flat_rows, info.k, &dx_flat)) {
                return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                            "matmul backward invalid flattened grad_x view");
            }
            st = gd_backend_matmul_nt(gd_context_backend(ctx), &g_flat, &w_flat, &dx_flat);
            if (st != GD_OK) {
                return gd_context_set_error(ctx, st, "backend matmul backward flattened grad_x failed");
            }
        }
        if (need_grad_w) {
            st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16,
                                 gd_shape_make(w->rank, w->shape), 256U, &dw);
            if (st != GD_OK) {
                return st;
            }
            dw.is_leaf = false;
            if (!gd_matmul_flat_matrix_view_from_tensor(x, flat_rows, info.k, &x_flat) ||
                !gd_matmul_flat_matrix_view_from_tensor(&dw, info.k, info.n, &dw_flat)) {
                return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                            "matmul backward invalid flattened grad_w view");
            }
            st = gd_backend_matmul_tn(gd_context_backend(ctx), &x_flat, &g_flat, &dw_flat);
            if (st != GD_OK) {
                return gd_context_set_error(ctx, st, "backend matmul backward flattened grad_w failed");
            }
        }
        if (need_grad_x) {
            *grad_x = dx;
        }
        if (need_grad_w) {
            *grad_w = dw;
        }
        return GD_OK;
    }
    if (!gd_matmul_batched_matrix_view_from_tensor(x, &info, &xv) ||
        !gd_matmul_batched_matrix_view_from_tensor(w, &info, &wv) ||
        !gd_matmul_batched_matrix_view_from_tensor(grad_out, &info, &gv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid input view");
    }
    reduce_x = need_grad_x && gd_matmul_operand_needs_batch_reduce(x, &info);
    reduce_w = need_grad_w && gd_matmul_operand_needs_batch_reduce(w, &info);
    if (need_grad_x) {
        gd_tensor *dx_target = reduce_x ? &dx_partial : &dx;
        if (reduce_x) {
            st = gd_matmul_make_partial(ctx, &info, info.m, info.k, &dx_partial);
        } else {
            st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16,
                                 gd_shape_make(x->rank, x->shape), 256U, &dx);
        }
        if (st != GD_OK) {
            return st;
        }
        dx_target->is_leaf = false;
        if (!gd_matmul_batched_matrix_view_from_tensor(dx_target, &info, &dxv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid grad_x view");
        }
        st = gd_backend_batched_matmul_nt(gd_context_backend(ctx), &gv, &wv, &dxv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend batched matmul backward grad_x failed");
        }
        if (reduce_x) {
            st = gd_matmul_reduce_batch_gradient(ctx, &dx_partial, x, &info, &dx);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    if (need_grad_w) {
        gd_tensor *dw_target = reduce_w ? &dw_partial : &dw;
        if (reduce_w) {
            st = gd_matmul_make_partial(ctx, &info, info.k, info.n, &dw_partial);
        } else {
            st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16,
                                 gd_shape_make(w->rank, w->shape), 256U, &dw);
        }
        if (st != GD_OK) {
            return st;
        }
        dw_target->is_leaf = false;
        if (!gd_matmul_batched_matrix_view_from_tensor(dw_target, &info, &dwv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid grad_w view");
        }
        st = gd_backend_batched_matmul_tn(gd_context_backend(ctx), &xv, &gv, &dwv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend batched matmul backward grad_w failed");
        }
        if (reduce_w) {
            st = gd_matmul_reduce_batch_gradient(ctx, &dw_partial, w, &info, &dw);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    if (need_grad_x) {
        *grad_x = dx;
    }
    if (need_grad_w) {
        *grad_w = dw;
    }
    return GD_OK;
}
