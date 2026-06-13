#include "matmul_impl.h"

#include "../autograd_impl.h"

#include <string.h>

gd_status gd_matmul(gd_context *ctx, const gd_tensor *x, const gd_tensor *w, gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_matmul_shape_info info;
    gd_backend_batched_matrix_view xv;
    gd_backend_batched_matrix_view wv;
    gd_backend_batched_matrix_view yv;
    gd_backend_matrix_view x_flat;
    gd_backend_matrix_view w_flat;
    gd_backend_matrix_view y_flat;
    int64_t flat_rows;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_matmul_validate_common(ctx, x, w, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(info.out_rank, info.out_shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    if (gd_matmul_i64_mul_overflow((int64_t)info.batch_count, info.m, &flat_rows)) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "matmul flattened row count overflow");
    }
    if (w->rank == 2U && gd_tensor_is_contiguous(x) && flat_rows <= (int64_t)UINT32_MAX) {
        if (!gd_matmul_flat_matrix_view_from_tensor(x, flat_rows, info.k, &x_flat) ||
            !gd_op_matrix_view_from_tensor(w, &w_flat) ||
            !gd_matmul_flat_matrix_view_from_tensor(&y, flat_rows, info.n, &y_flat)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul invalid flattened matrix view");
        }
        st = gd_backend_matmul(gd_context_backend(ctx), &x_flat, &w_flat, &y_flat);
    } else {
        if (!gd_matmul_batched_matrix_view_from_tensor(x, &info, &xv) ||
            !gd_matmul_batched_matrix_view_from_tensor(w, &info, &wv) ||
            !gd_matmul_batched_matrix_view_from_tensor(&y, &info, &yv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "matmul invalid batched matrix view");
        }
        st = gd_backend_batched_matmul(gd_context_backend(ctx), &xv, &wv, &yv);
    }
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
