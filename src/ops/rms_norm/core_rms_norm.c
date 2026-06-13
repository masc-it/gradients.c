#include "rms_norm_impl.h"

#include "../autograd_impl.h"
#include "../op_common.h"

#include <stddef.h>
#include <string.h>

#define GD_RMS_NORM_WGRAD_ROW_BLOCK_SMALL 64U
#define GD_RMS_NORM_WGRAD_ROW_BLOCK_LARGE 128U
#define GD_RMS_NORM_WGRAD_LARGE_ROWS_THRESHOLD 4096U

static bool gd_rms_norm_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static uint32_t gd_rms_norm_simdgroups(uint64_t cols)
{
    if (cols <= 64U) {
        return 1U;
    }
    if (cols <= 256U) {
        return 2U;
    }
    if (cols <= 1024U) {
        return 4U;
    }
    return 8U;
}

static uint32_t gd_rms_norm_wgrad_simdgroups(uint64_t row_blocks)
{
    if (row_blocks <= 64U) {
        return 1U;
    }
    if (row_blocks <= 256U) {
        return 4U;
    }
    return 8U;
}

static gd_status gd_rms_norm_validate_forward(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *weight,
                                              float eps,
                                              int64_t *out_rows,
                                              int64_t *out_cols)
{
    gd_status st;
    int64_t numel;
    int64_t cols;
    if (ctx == NULL || x == NULL || weight == NULL || !(eps > 0.0f) || !(eps == eps)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, weight);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_rms_norm_dtype_supported(x->dtype) || weight->dtype != x->dtype) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "rms_norm requires matching f16/f32 input and weight dtypes");
    }
    if (x->rank < 1U || weight->rank != 1U) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm expects x rank >= 1 and weight [last_dim]");
    }
    cols = x->shape[x->rank - 1U];
    if (cols <= 0 || weight->shape[0] != cols) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm weight length must match x last dimension");
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(weight)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "rms_norm requires contiguous x and weight tensors");
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "rms_norm invalid input shape");
    }
    if (numel <= 0 || numel % cols != 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm invalid row/column shape");
    }
    if (out_rows != NULL) {
        *out_rows = numel / cols;
    }
    if (out_cols != NULL) {
        *out_cols = cols;
    }
    return GD_OK;
}

static bool gd_rms_norm_same_shape_dtype(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->dtype != b->dtype || a->rank != b->rank) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_rms_norm_validate_grad_out(gd_context *ctx,
                                               const gd_tensor *x,
                                               const gd_tensor *grad_out)
{
    gd_status st;
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_rms_norm_same_shape_dtype(x, grad_out) || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm backward requires contiguous grad_out matching x");
    }
    return GD_OK;
}

static gd_status gd_rms_norm_validate_inv(gd_context *ctx,
                                          const gd_tensor *inv_rms,
                                          int64_t rows)
{
    gd_status st;
    if (ctx == NULL || inv_rms == NULL || rows <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, inv_rms);
    if (st != GD_OK) {
        return st;
    }
    if (inv_rms->dtype != GD_DTYPE_F32 || inv_rms->rank != 1U || inv_rms->shape[0] != rows ||
        !gd_tensor_is_contiguous(inv_rms)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm inv_rms stats must be contiguous f32 [rows]");
    }
    return GD_OK;
}

static gd_status gd_rms_norm_make_args(gd_context *ctx,
                                       int64_t rows,
                                       int64_t cols,
                                       float eps,
                                       gd_backend_rms_norm_args *out_args)
{
    uint64_t urows;
    uint64_t ucols;
    uint64_t row_block;
    uint64_t row_blocks;
    if (ctx == NULL || out_args == NULL || rows <= 0 || cols <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    urows = (uint64_t)rows;
    ucols = (uint64_t)cols;
    row_block = urows >= (uint64_t)GD_RMS_NORM_WGRAD_LARGE_ROWS_THRESHOLD
                    ? (uint64_t)GD_RMS_NORM_WGRAD_ROW_BLOCK_LARGE
                    : (uint64_t)GD_RMS_NORM_WGRAD_ROW_BLOCK_SMALL;
    row_blocks = (urows + row_block - 1U) / row_block;
    if (urows > UINT64_MAX / ucols || row_blocks > (uint64_t)INT64_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "rms_norm shape overflow");
    }
    memset(out_args, 0, sizeof(*out_args));
    out_args->rows = urows;
    out_args->cols = ucols;
    out_args->row_blocks = row_blocks;
    out_args->eps = eps;
    out_args->simdgroups = gd_rms_norm_simdgroups(ucols);
    out_args->wgrad_simdgroups = gd_rms_norm_wgrad_simdgroups(row_blocks);
    out_args->wgrad_row_block = (uint32_t)row_block;
    return GD_OK;
}

static gd_status gd_rms_norm_dispatch_forward(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *weight,
                                              const gd_tensor *out,
                                              const gd_tensor *inv_rms,
                                              const gd_backend_rms_norm_args *args)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view wv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view iv;
    gd_status st;
    if (ctx == NULL || x == NULL || weight == NULL || out == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(weight, &wv) ||
        !gd_op_tensor_view_from_tensor(out, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm forward invalid tensor view");
    }
    if (inv_rms != NULL) {
        if (!gd_op_tensor_view_from_tensor(inv_rms, &iv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "rms_norm forward invalid inv_rms view");
        }
        st = gd_backend_rms_norm_forward_stats(gd_context_backend(ctx), &xv, &wv, &yv, &iv, args);
    } else {
        st = gd_backend_rms_norm_forward(gd_context_backend(ctx), &xv, &wv, &yv, args);
    }
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rms_norm forward failed");
    }
    return GD_OK;
}

static gd_status gd_rms_norm_dispatch_inv(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *inv_rms,
                                          const gd_backend_rms_norm_args *args)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view iv;
    gd_status st;
    if (ctx == NULL || x == NULL || inv_rms == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(inv_rms, &iv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm inv invalid tensor view");
    }
    st = gd_backend_rms_norm_inv(gd_context_backend(ctx), &xv, &iv, args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rms_norm inv failed");
    }
    return GD_OK;
}

static gd_status gd_rms_norm_dispatch_backward(gd_context *ctx,
                                               const gd_tensor *x,
                                               const gd_tensor *weight,
                                               const gd_tensor *inv_rms,
                                               const gd_tensor *grad_out,
                                               const gd_tensor *grad_x,
                                               const gd_backend_rms_norm_args *args)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view wv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_status st;
    if (ctx == NULL || x == NULL || weight == NULL || grad_out == NULL || grad_x == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(weight, &wv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_x, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm backward invalid tensor view");
    }
    if (inv_rms != NULL) {
        if (!gd_op_tensor_view_from_tensor(inv_rms, &iv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "rms_norm backward invalid inv_rms view");
        }
        st = gd_backend_rms_norm_backward_stats(gd_context_backend(ctx), &xv, &wv, &iv, &gv, &dxv, args);
    } else {
        st = gd_backend_rms_norm_backward(gd_context_backend(ctx), &xv, &wv, &gv, &dxv, args);
    }
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rms_norm backward dx failed");
    }
    return GD_OK;
}

static gd_status gd_rms_norm_dispatch_weight_backward_stats(gd_context *ctx,
                                                            const gd_tensor *x,
                                                            const gd_tensor *inv_rms,
                                                            const gd_tensor *grad_out,
                                                            const gd_tensor *grad_weight,
                                                            const gd_backend_rms_norm_args *args)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dwv;
    gd_backend_tensor_view pv;
    gd_tensor partial;
    gd_status st;
    int64_t partial_shape[2];
    if (ctx == NULL || x == NULL || inv_rms == NULL || grad_out == NULL || grad_weight == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (args->row_blocks == 0U || args->row_blocks > (uint64_t)INT64_MAX ||
        args->cols == 0U || args->cols > (uint64_t)INT64_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "rms_norm weight backward partial shape overflow");
    }
    partial_shape[0] = (int64_t)args->row_blocks;
    partial_shape[1] = (int64_t)args->cols;
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(2U, partial_shape), 256U, &partial);
    if (st != GD_OK) {
        return st;
    }
    partial.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(inv_rms, &iv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_weight, &dwv) ||
        !gd_op_tensor_view_from_tensor(&partial, &pv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "rms_norm weight backward invalid tensor view");
    }
    st = gd_backend_rms_norm_weight_backward_stats(gd_context_backend(ctx), &xv, &iv, &gv, &dwv, &pv, args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rms_norm backward weight failed");
    }
    return GD_OK;
}

gd_status gd_rms_norm(gd_context *ctx,
                      const gd_tensor *x,
                      const gd_tensor *weight,
                      float eps,
                      gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_tensor inv_rms;
    gd_backend_rms_norm_args args;
    int64_t rows;
    int64_t cols;
    bool need_stats;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&y, 0, sizeof(y));
    memset(&inv_rms, 0, sizeof(inv_rms));
    if (ctx == NULL || x == NULL || weight == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_forward(ctx, x, weight, eps, &rows, &cols);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_make_args(ctx, rows, cols, eps, &args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    need_stats = (x->requires_grad || weight->requires_grad) &&
                 gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
    if (need_stats) {
        int64_t stats_shape[1];
        stats_shape[0] = rows;
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, stats_shape), 256U, &inv_rms);
        if (st != GD_OK) {
            return st;
        }
        inv_rms.is_leaf = false;
        st = gd_rms_norm_dispatch_forward(ctx, x, weight, &y, &inv_rms, &args);
    } else {
        st = gd_rms_norm_dispatch_forward(ctx, x, weight, &y, NULL, &args);
    }
    if (st != GD_OK) {
        return st;
    }
    {
        gd_rms_norm_attrs attrs;
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        const gd_tensor *saved[1];
        memset(&attrs, 0, sizeof(attrs));
        attrs.eps = eps;
        inputs[0] = x;
        inputs[1] = weight;
        outputs[0] = &y;
        saved[0] = &inv_rms;
        st = gd_autograd_record(ctx,
                                GD_OP_RMS_NORM,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                need_stats ? saved : NULL,
                                need_stats ? 1U : 0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_rms_norm_backward_with_stats(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *weight,
                                          const gd_tensor *inv_rms,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x,
                                          gd_tensor *grad_weight)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dw;
    gd_backend_rms_norm_args args;
    int64_t rows;
    int64_t cols;
    bool need_grad_x = grad_x != NULL;
    bool need_grad_weight = grad_weight != NULL;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_weight != NULL) {
        memset(grad_weight, 0, sizeof(*grad_weight));
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    if (ctx == NULL || x == NULL || weight == NULL || inv_rms == NULL || grad_out == NULL ||
        (!need_grad_x && !need_grad_weight)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_forward(ctx, x, weight, 1.0f, &rows, &cols);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_grad_out(ctx, x, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_inv(ctx, inv_rms, rows);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_make_args(ctx, rows, cols, 1.0f, &args);
    if (st != GD_OK) {
        return st;
    }
    if (need_grad_x) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &dx);
        if (st != GD_OK) {
            return st;
        }
        dx.is_leaf = false;
        st = gd_rms_norm_dispatch_backward(ctx, x, weight, inv_rms, grad_out, &dx, &args);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_grad_weight) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, weight->dtype, gd_shape_make(weight->rank, weight->shape), 256U, &dw);
        if (st != GD_OK) {
            return st;
        }
        dw.is_leaf = false;
        st = gd_rms_norm_dispatch_weight_backward_stats(ctx, x, inv_rms, grad_out, &dw, &args);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_grad_x) {
        *grad_x = dx;
    }
    if (need_grad_weight) {
        *grad_weight = dw;
    }
    return GD_OK;
}

gd_status gd_rms_norm_backward(gd_context *ctx,
                               const gd_tensor *x,
                               const gd_tensor *weight,
                               const gd_tensor *grad_out,
                               float eps,
                               gd_tensor *grad_x,
                               gd_tensor *grad_weight)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor inv_rms;
    gd_backend_rms_norm_args args;
    int64_t rows;
    int64_t cols;
    bool need_grad_x = grad_x != NULL;
    bool need_grad_weight = grad_weight != NULL;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_weight != NULL) {
        memset(grad_weight, 0, sizeof(*grad_weight));
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    memset(&inv_rms, 0, sizeof(inv_rms));
    if (ctx == NULL || x == NULL || weight == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_forward(ctx, x, weight, eps, &rows, &cols);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_grad_out(ctx, x, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_make_args(ctx, rows, cols, eps, &args);
    if (st != GD_OK) {
        return st;
    }
    if (!need_grad_x && !need_grad_weight) {
        return GD_OK;
    }
    if (need_grad_weight) {
        int64_t stats_shape[1];
        stats_shape[0] = rows;
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, stats_shape), 256U, &inv_rms);
        if (st != GD_OK) {
            return st;
        }
        inv_rms.is_leaf = false;
        st = gd_rms_norm_dispatch_inv(ctx, x, &inv_rms, &args);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_grad_x) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &dx);
        if (st != GD_OK) {
            return st;
        }
        dx.is_leaf = false;
        st = gd_rms_norm_dispatch_backward(ctx,
                                           x,
                                           weight,
                                           need_grad_weight ? &inv_rms : NULL,
                                           grad_out,
                                           &dx,
                                           &args);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_grad_weight) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, weight->dtype, gd_shape_make(weight->rank, weight->shape), 256U, &dw);
        if (st != GD_OK) {
            return st;
        }
        dw.is_leaf = false;
        st = gd_rms_norm_dispatch_weight_backward_stats(ctx, x, &inv_rms, grad_out, &dw, &args);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_grad_x) {
        *grad_x = dx;
    }
    if (need_grad_weight) {
        *grad_weight = dw;
    }
    return GD_OK;
}
