#include "../linear/linear_impl.h"

#include "../autograd_impl.h"

#include <string.h>

gd_status gd_linear_transposed_weight(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *w,
                                      const gd_tensor *bias,
                                      gd_tensor *out)
{
    gd_status st;
    gd_tensor matmul_y;
    gd_tensor y;
    gd_linear_shape_info info;
    gd_backend_matrix_view xv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view yv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&matmul_y, 0, sizeof(matmul_y));
    memset(&y, 0, sizeof(y));
    if (ctx == NULL || x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_linear_transposed_weight_validate_common(ctx, x, w, bias, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(info.rank, info.out_shape),
                         256U,
                         bias == NULL ? &y : &matmul_y);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_linear_flat_matrix_view_from_tensor(x, info.rows, info.k, &xv) ||
        !gd_op_matrix_view_from_tensor(w, &wv) ||
        !gd_linear_flat_matrix_view_from_tensor(bias == NULL ? &y : &matmul_y,
                                                info.rows,
                                                info.n,
                                                &yv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "linear_transposed_weight invalid backend view");
    }
    st = gd_backend_matmul_nt(gd_context_backend(ctx), &xv, &wv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend linear_transposed_weight matmul failed");
    }
    if (bias != NULL) {
        gd_backend_tensor_view mv;
        gd_backend_tensor_view bv;
        gd_backend_tensor_view ov;
        matmul_y.is_leaf = false;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(info.rank, info.out_shape),
                             256U,
                             &y);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_op_tensor_view_from_tensor(&matmul_y, &mv) ||
            !gd_op_tensor_view_from_tensor(bias, &bv) ||
            !gd_op_tensor_view_from_tensor(&y, &ov)) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "linear_transposed_weight invalid bias view");
        }
        st = gd_backend_add(gd_context_backend(ctx), &mv, &bv, &ov);
        if (st != GD_OK) {
            return gd_context_set_error(ctx,
                                        st,
                                        "backend linear_transposed_weight bias add failed");
        }
    }
    y.is_leaf = false;
    {
        const gd_tensor *inputs[3] = {x, w, bias};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_LINEAR_TRANSPOSED_WEIGHT,
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
