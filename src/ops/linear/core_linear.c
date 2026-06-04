#include <gradients/ops.h>

#include "../op_common.h"

#include <string.h>

static gd_status gd_linear_validate(gd_context *ctx,
                                    const gd_tensor *x,
                                    const gd_tensor *w,
                                    const gd_tensor *bias)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL || bias == NULL) {
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
    st = gd_tensor_validate(ctx, bias);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 || bias->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "linear currently supports f16 tensors only");
    }
    if (x->rank != 2U || w->rank != 2U || bias->rank != 1U ||
        x->shape[1] != w->shape[0] || bias->shape[0] != w->shape[1]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear shape mismatch");
    }
    if (x->strides[1] != 1 || w->strides[1] != 1 || bias->strides[0] != 1) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "linear requires row-strided inputs");
    }
    return GD_OK;
}

gd_status gd_linear(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    const gd_tensor *bias,
                    gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    int64_t y_shape[2];
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view yv;
    gd_backend_vector_view bv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || w == NULL || bias == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_linear_validate(ctx, x, w, bias);
    if (st != GD_OK) {
        return st;
    }
    y_shape[0] = x->shape[0];
    y_shape[1] = w->shape[1];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, 2U, y_shape, 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_matrix_view_from_tensor(x, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_op_matrix_view_from_tensor(&y, &yv) ||
        !gd_op_vector_view_from_tensor(bias, &bv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear invalid backend view");
    }
    st = gd_backend_linear(gd_context_backend(ctx), &xv, &wv, &bv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend linear failed");
    }
    *out = y;
    return GD_OK;
}
