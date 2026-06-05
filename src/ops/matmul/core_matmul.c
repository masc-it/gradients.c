#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <string.h>

static gd_status gd_matmul_validate(gd_context *ctx, const gd_tensor *x, const gd_tensor *w)
{
    gd_status st;
    if (ctx == NULL || x == NULL || w == NULL) {
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
    if (x->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "matmul currently supports f16 tensors only");
    }
    if (x->rank != 2U || w->rank != 2U || x->shape[1] != w->shape[0]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul shape mismatch");
    }
    if (x->strides[1] != 1 || w->strides[1] != 1) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "matmul requires row-strided inputs");
    }
    return GD_OK;
}

gd_status gd_matmul(gd_context *ctx, const gd_tensor *x, const gd_tensor *w, gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    int64_t y_shape[2];
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view yv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_matmul_validate(ctx, x, w);
    if (st != GD_OK) {
        return st;
    }
    y_shape[0] = x->shape[0];
    y_shape[1] = w->shape[1];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, y_shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_matrix_view_from_tensor(x, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_op_matrix_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul invalid matrix view");
    }
    st = gd_backend_matmul(gd_context_backend(ctx), &xv, &wv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend matmul failed");
    }
    y.is_leaf = false;
    {
        const gd_tensor *inputs[2] = {x, w};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_MATMUL,
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
    *out = y;
    return GD_OK;
}
