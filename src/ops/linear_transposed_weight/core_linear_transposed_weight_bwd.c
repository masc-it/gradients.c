#include "../linear/linear_impl.h"

#include <string.h>

static gd_status gd_linear_transposed_weight_backward_validate(gd_context *ctx,
                                                               const gd_tensor *x,
                                                               const gd_tensor *w,
                                                               const gd_tensor *bias,
                                                               const gd_tensor *grad_out,
                                                               bool need_grad_x,
                                                               bool need_grad_w,
                                                               bool need_grad_bias,
                                                               gd_linear_shape_info *info)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || grad_out == NULL || info == NULL ||
        (!need_grad_x && !need_grad_w && !need_grad_bias)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (need_grad_bias && bias == NULL) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "linear_transposed_weight backward grad_bias requested without bias");
    }
    st = gd_linear_transposed_weight_validate_common(ctx, x, w, bias, info);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_validate_grad_out(ctx, grad_out, info);
}

gd_status gd_linear_transposed_weight_backward(gd_context *ctx,
                                               const gd_tensor *x,
                                               const gd_tensor *w,
                                               const gd_tensor *bias,
                                               const gd_tensor *grad_out,
                                               gd_tensor *grad_x,
                                               gd_tensor *grad_w,
                                               gd_tensor *grad_bias)
{
    gd_status st;
    gd_linear_shape_info info;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
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
    st = gd_linear_transposed_weight_backward_validate(ctx,
                                                       x,
                                                       w,
                                                       bias,
                                                       grad_out,
                                                       need_grad_x,
                                                       need_grad_w,
                                                       need_grad_bias,
                                                       &info);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_linear_flat_matrix_view_from_tensor(x, info.rows, info.k, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_linear_flat_matrix_view_from_tensor(grad_out, info.rows, info.n, &gv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "linear_transposed_weight backward invalid input view");
    }
    if (need_grad_x) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(x->rank, x->shape),
                             256U,
                             &dx);
        if (st != GD_OK) {
            return st;
        }
        dx.is_leaf = false;
        if (!gd_linear_flat_matrix_view_from_tensor(&dx, info.rows, info.k, &dxv)) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "linear_transposed_weight backward invalid grad_x view");
        }
        st = gd_backend_matmul(gd_context_backend(ctx), &gv, &wv, &dxv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx,
                                        st,
                                        "backend linear_transposed_weight grad_x failed");
        }
    }
    if (need_grad_w) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(w->rank, w->shape),
                             256U,
                             &dw);
        if (st != GD_OK) {
            return st;
        }
        dw.is_leaf = false;
        if (!gd_op_matrix_view_from_tensor(&dw, &dwv)) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "linear_transposed_weight backward invalid grad_w view");
        }
        st = gd_backend_matmul_tn(gd_context_backend(ctx), &gv, &xv, &dwv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx,
                                        st,
                                        "backend linear_transposed_weight grad_w failed");
        }
    }
    if (need_grad_bias) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(bias->rank, bias->shape),
                             256U,
                             &db);
        if (st != GD_OK) {
            return st;
        }
        db.is_leaf = false;
        if (!gd_op_vector_view_from_tensor(&db, &dbv)) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "linear_transposed_weight backward invalid grad_bias view");
        }
        st = gd_backend_reduce_rows(gd_context_backend(ctx), &gv, &dbv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx,
                                        st,
                                        "backend linear_transposed_weight grad_bias failed");
        }
    }
    if (grad_x != NULL) {
        *grad_x = dx;
    }
    if (grad_w != NULL) {
        *grad_w = dw;
    }
    if (grad_bias != NULL) {
        *grad_bias = db;
    }
    return GD_OK;
}
