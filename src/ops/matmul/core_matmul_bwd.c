#include <gradients/ops.h>

#include "../op_common.h"

#include <string.h>

static gd_status gd_matmul_backward_validate(gd_context *ctx,
                                             const gd_tensor *x,
                                             const gd_tensor *w,
                                             const gd_tensor *grad_out,
                                             bool need_grad_x,
                                             bool need_grad_w)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || grad_out == NULL ||
        (!need_grad_x && !need_grad_w)) {
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
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 || grad_out->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "matmul backward currently supports f16 tensors only");
    }
    if (x->rank != 2U || w->rank != 2U || grad_out->rank != 2U ||
        x->shape[1] != w->shape[0] || grad_out->shape[0] != x->shape[0] ||
        grad_out->shape[1] != w->shape[1]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward shape mismatch");
    }
    if (x->strides[1] != 1 || w->strides[1] != 1 || grad_out->strides[1] != 1) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "matmul backward requires row-strided inputs");
    }
    return GD_OK;
}

gd_status gd_matmul_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dw;
    int64_t dx_shape[2];
    int64_t dw_shape[2];
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view gv;
    gd_backend_matrix_view dxv;
    gd_backend_matrix_view dwv;
    bool need_grad_x = grad_x != NULL;
    bool need_grad_w = grad_w != NULL;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    st = gd_matmul_backward_validate(ctx, x, w, grad_out, need_grad_x, need_grad_w);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_matrix_view_from_tensor(x, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_op_matrix_view_from_tensor(grad_out, &gv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid input view");
    }
    if (need_grad_x) {
        dx_shape[0] = x->shape[0];
        dx_shape[1] = x->shape[1];
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, 2U, dx_shape, 256U, &dx);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_matrix_view_from_tensor(&dx, &dxv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid grad_x view");
        }
        st = gd_backend_matmul_nt(gd_context_backend(ctx), &gv, &wv, &dxv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend matmul backward grad_x failed");
        }
    }
    if (need_grad_w) {
        dw_shape[0] = w->shape[0];
        dw_shape[1] = w->shape[1];
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, 2U, dw_shape, 256U, &dw);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_matrix_view_from_tensor(&dw, &dwv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul backward invalid grad_w view");
        }
        st = gd_backend_matmul_tn(gd_context_backend(ctx), &xv, &gv, &dwv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend matmul backward grad_w failed");
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
