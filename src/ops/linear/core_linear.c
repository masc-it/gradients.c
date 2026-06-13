#include "linear_impl.h"

#include "../autograd_impl.h"

#include <string.h>

gd_status gd_linear(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    const gd_tensor *bias,
                    gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_linear_shape_info info;
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view yv;
    gd_backend_vector_view bv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_linear_validate_common(ctx, x, w, bias, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(info.rank, info.out_shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    memset(&bv, 0, sizeof(bv));
    if (!gd_linear_flat_matrix_view_from_tensor(x, info.rows, info.k, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_linear_flat_matrix_view_from_tensor(&y, info.rows, info.n, &yv) ||
        (bias != NULL && !gd_op_vector_view_from_tensor(bias, &bv))) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "linear invalid backend view");
    }
    if (bias == NULL) {
        st = gd_backend_matmul(gd_context_backend(ctx), &xv, &wv, &yv);
    } else {
        st = gd_backend_linear(gd_context_backend(ctx), &xv, &wv, &bv, &yv);
    }
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend linear failed");
    }
    y.is_leaf = false;
    {
        const gd_tensor *inputs[3] = {x, w, bias};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_LINEAR,
                                inputs,
                                bias != NULL ? 3U : 2U,
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
