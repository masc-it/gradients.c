#include <gradients/ops.h>

#include "../op_common.h"

#include <string.h>

static gd_status gd_linear_backward_validate(gd_context *ctx,
                                             const gd_tensor *x,
                                             const gd_tensor *w,
                                             const gd_tensor *bias,
                                             const gd_tensor *grad_out,
                                             bool need_grad_x,
                                             bool need_grad_w,
                                             bool need_grad_bias)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || grad_out == NULL ||
        (!need_grad_x && !need_grad_w && !need_grad_bias)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (need_grad_bias && bias == NULL) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "linear backward grad_bias requested without bias");
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, w);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (bias != NULL) {
        st = gd_tensor_validate(ctx, bias);
        if (st != GD_OK) {
            return st;
        }
    }
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 || grad_out->dtype != GD_DTYPE_F16 ||
        (bias != NULL && bias->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear backward currently supports f16 tensors only");
    }
    if (x->rank != 2U || w->rank != 2U || grad_out->rank != 2U ||
        (bias != NULL && bias->rank != 1U) ||
        x->shape[1] != w->shape[0] || grad_out->shape[0] != x->shape[0] ||
        grad_out->shape[1] != w->shape[1] ||
        (bias != NULL && bias->shape[0] != w->shape[1])) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward shape mismatch");
    }
    if (x->strides[1] != 1 || w->strides[1] != 1 || grad_out->strides[1] != 1 ||
        (bias != NULL && bias->strides[0] != 1)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "linear backward requires row-strided inputs");
    }
    return GD_OK;
}

gd_status gd_linear_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *bias,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w,
                             gd_tensor *grad_bias)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
    int64_t dx_shape[2];
    int64_t dw_shape[2];
    int64_t db_shape[1];
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view gv;
    gd_backend_matrix_view dxv;
    gd_backend_matrix_view dwv;
    gd_backend_vector_view dbv;
    bool need_grad_x = grad_x != NULL;
    bool need_grad_w = grad_w != NULL;
    bool need_grad_bias = grad_bias != NULL;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    if (grad_bias != NULL) {
        memset(grad_bias, 0, sizeof(*grad_bias));
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    memset(&db, 0, sizeof(db));
    st = gd_linear_backward_validate(ctx, x, w, bias, grad_out,
                                     need_grad_x, need_grad_w, need_grad_bias);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_matrix_view_from_tensor(x, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_op_matrix_view_from_tensor(grad_out, &gv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward invalid input view");
    }
    if (need_grad_x) {
        dx_shape[0] = x->shape[0];
        dx_shape[1] = x->shape[1];
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, dx_shape), 256U, &dx);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_matrix_view_from_tensor(&dx, &dxv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward invalid grad_x view");
        }
        st = gd_backend_matmul_nt(gd_context_backend(ctx), &gv, &wv, &dxv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend linear backward grad_x failed");
        }
    }
    if (need_grad_w) {
        dw_shape[0] = w->shape[0];
        dw_shape[1] = w->shape[1];
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, dw_shape), 256U, &dw);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_matrix_view_from_tensor(&dw, &dwv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward invalid grad_w view");
        }
        st = gd_backend_matmul_tn(gd_context_backend(ctx), &xv, &gv, &dwv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend linear backward grad_w failed");
        }
    }
    if (need_grad_bias) {
        db_shape[0] = w->shape[1];
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(1U, db_shape), 256U, &db);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_vector_view_from_tensor(&db, &dbv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear backward invalid grad_bias view");
        }
        st = gd_backend_reduce_rows(gd_context_backend(ctx), &gv, &dbv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend linear backward grad_bias failed");
        }
    }
    if (need_grad_x) {
        *grad_x = dx;
    }
    if (need_grad_w) {
        *grad_w = dw;
    }
    if (need_grad_bias) {
        *grad_bias = db;
    }
    return GD_OK;
}
