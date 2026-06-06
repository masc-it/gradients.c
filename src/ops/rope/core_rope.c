#include "rope_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define GD_ROPE_DEFAULT_THETA 10000.0f

static bool gd_rope_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_rope_same_shape_dtype(const gd_tensor *a, const gd_tensor *b)
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

static gd_status gd_rope_validate_base(gd_context *ctx,
                                       const gd_tensor *x,
                                       const gd_tensor *pos_ids,
                                       int64_t *out_rows,
                                       int64_t *out_lanes)
{
    gd_status st;
    int64_t pos_count;
    int64_t lead_count = 1;
    int64_t total_count;
    int64_t head_dim;
    int64_t heads;
    uint32_t i;
    if (ctx == NULL || x == NULL || pos_ids == NULL || out_rows == NULL || out_lanes == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_rows = 0;
    *out_lanes = 0;
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, pos_ids);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_rope_dtype_supported(x->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "rope supports f16/f32 inputs only");
    }
    if (pos_ids->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "rope position ids must be i32");
    }
    if (x->rank < 2U) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "rope input must be shaped [.., heads, head_dim]");
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(pos_ids)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "rope requires contiguous tensors");
    }
    head_dim = x->shape[x->rank - 1U];
    heads = x->shape[x->rank - 2U];
    if (head_dim <= 0 || heads <= 0 || head_dim > (int64_t)INT32_MAX || heads > (int64_t)INT32_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "rope heads/head_dim exceed kernel limits");
    }
    for (i = 0U; i + 2U < x->rank; ++i) {
        if (x->shape[i] <= 0 || lead_count > INT64_MAX / x->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope invalid leading shape");
        }
        lead_count *= x->shape[i];
    }
    st = gd_tensor_numel(pos_ids, &pos_count);
    if (st != GD_OK) {
        return st;
    }
    if (pos_count != lead_count) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "rope position count must equal product of leading dims");
    }
    st = gd_tensor_numel(x, &total_count);
    if (st != GD_OK) {
        return st;
    }
    if (total_count <= 0 || total_count % head_dim != 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope invalid input shape");
    }
    *out_rows = total_count / head_dim;
    *out_lanes = 1;
    return GD_OK;
}

static gd_status gd_rope_resolve_config(gd_context *ctx,
                                        const gd_tensor *x,
                                        const gd_rope_config *config,
                                        gd_rope_attrs *attrs)
{
    int64_t head_dim;
    int64_t n_dims;
    float theta;
    if (ctx == NULL || x == NULL || attrs == NULL || x->rank < 2U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    head_dim = x->shape[x->rank - 1U];
    n_dims = (config != NULL && config->n_dims > 0) ? (int64_t)config->n_dims : head_dim;
    if (n_dims <= 0 || n_dims > (int64_t)INT32_MAX || (n_dims & 1) != 0 || n_dims > head_dim) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "rope n_dims must be even and <= head_dim");
    }
    theta = (config != NULL && config->theta > 0.0f) ? config->theta : GD_ROPE_DEFAULT_THETA;
    if (!isfinite(theta) || theta <= 0.0f) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope theta must be finite and positive");
    }
    attrs->theta = theta;
    attrs->n_dims = (int32_t)n_dims;
    attrs->interleaved = (config != NULL && config->interleaved) ? 1 : 0;
    return GD_OK;
}

static gd_status gd_rope_validate_attrs(gd_context *ctx,
                                        const gd_tensor *x,
                                        const gd_rope_attrs *attrs,
                                        int64_t rows,
                                        int64_t *out_lanes)
{
    int64_t head_dim;
    int64_t half;
    int64_t tail;
    int64_t lanes;
    if (ctx == NULL || x == NULL || attrs == NULL || out_lanes == NULL || x->rank < 2U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    head_dim = x->shape[x->rank - 1U];
    if (!isfinite(attrs->theta) || attrs->theta <= 0.0f || attrs->n_dims <= 0 ||
        (attrs->n_dims & 1) != 0 || (int64_t)attrs->n_dims > head_dim ||
        (attrs->interleaved != 0 && attrs->interleaved != 1)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope invalid attributes");
    }
    half = (int64_t)attrs->n_dims / 2;
    tail = head_dim - (int64_t)attrs->n_dims;
    lanes = half > tail ? half : tail;
    if (rows <= 0 || lanes <= 0 || rows > (int64_t)UINT32_MAX ||
        lanes > (int64_t)UINT32_MAX || rows > (int64_t)UINT32_MAX / lanes) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "rope launch dimensions exceed kernel limits");
    }
    *out_lanes = lanes;
    return GD_OK;
}

static gd_status gd_rope_validate_forward(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *pos_ids,
                                          const gd_rope_config *config,
                                          gd_rope_attrs *attrs)
{
    gd_status st;
    int64_t rows;
    int64_t lanes;
    st = gd_rope_validate_base(ctx, x, pos_ids, &rows, &lanes);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rope_resolve_config(ctx, x, config, attrs);
    if (st != GD_OK) {
        return st;
    }
    return gd_rope_validate_attrs(ctx, x, attrs, rows, &lanes);
}

static gd_status gd_rope_validate_backward_attrs(gd_context *ctx,
                                                 const gd_tensor *x,
                                                 const gd_tensor *pos_ids,
                                                 const gd_tensor *grad_out,
                                                 const gd_rope_attrs *attrs)
{
    gd_status st;
    int64_t rows;
    int64_t lanes;
    st = gd_rope_validate_base(ctx, x, pos_ids, &rows, &lanes);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_rope_same_shape_dtype(x, grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "rope backward grad_out must match x shape/dtype");
    }
    if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "rope backward requires contiguous grad_out");
    }
    return gd_rope_validate_attrs(ctx, x, attrs, rows, &lanes);
}

static gd_backend_rope_args gd_rope_backend_args_from_attrs(const gd_rope_attrs *attrs)
{
    gd_backend_rope_args args;
    memset(&args, 0, sizeof(args));
    if (attrs != NULL) {
        args.theta = attrs->theta;
        args.n_dims = (uint32_t)attrs->n_dims;
        args.interleaved = attrs->interleaved != 0 ? 1U : 0U;
    }
    return args;
}

static gd_status gd_rope_dispatch_forward(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *pos_ids,
                                          const gd_tensor *out,
                                          const gd_rope_attrs *attrs)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view pv;
    gd_backend_tensor_view yv;
    gd_backend_rope_args args;
    gd_status st;
    if (ctx == NULL || x == NULL || pos_ids == NULL || out == NULL || attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(pos_ids, &pv) ||
        !gd_op_tensor_view_from_tensor(out, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope invalid tensor view");
    }
    args = gd_rope_backend_args_from_attrs(attrs);
    st = gd_backend_rope(gd_context_backend(ctx), &xv, &pv, &yv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rope failed");
    }
    return GD_OK;
}

static gd_status gd_rope_dispatch_backward(gd_context *ctx,
                                           const gd_tensor *grad_out,
                                           const gd_tensor *pos_ids,
                                           const gd_tensor *grad_x,
                                           const gd_rope_attrs *attrs)
{
    gd_backend_tensor_view gov;
    gd_backend_tensor_view pv;
    gd_backend_tensor_view dxv;
    gd_backend_rope_args args;
    gd_status st;
    if (ctx == NULL || grad_out == NULL || pos_ids == NULL || grad_x == NULL || attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(grad_out, &gov) || !gd_op_tensor_view_from_tensor(pos_ids, &pv) ||
        !gd_op_tensor_view_from_tensor(grad_x, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "rope backward invalid tensor view");
    }
    args = gd_rope_backend_args_from_attrs(attrs);
    st = gd_backend_rope_backward(gd_context_backend(ctx), &gov, &pv, &dxv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend rope backward failed");
    }
    return GD_OK;
}

gd_status gd_rope(gd_context *ctx,
                  const gd_tensor *x,
                  const gd_tensor *pos_ids,
                  const gd_rope_config *config,
                  gd_tensor *out)
{
    gd_status st;
    gd_rope_attrs attrs;
    gd_tensor y;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || pos_ids == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rope_validate_forward(ctx, x, pos_ids, config, &attrs);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    st = gd_rope_dispatch_forward(ctx, x, pos_ids, &y, &attrs);
    if (st != GD_OK) {
        return st;
    }
    if (x->requires_grad) {
        const gd_tensor *inputs[2] = {x, pos_ids};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_ROPE,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_rope_backward_from_attrs(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *pos_ids,
                                      const gd_tensor *grad_out,
                                      const gd_rope_attrs *attrs,
                                      gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || pos_ids == NULL || grad_out == NULL || attrs == NULL ||
        grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rope_validate_backward_attrs(ctx, x, pos_ids, grad_out, attrs);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_rope_dispatch_backward(ctx, grad_out, pos_ids, &dx, attrs);
    if (st != GD_OK) {
        return st;
    }
    *grad_x = dx;
    return GD_OK;
}

gd_status gd_rope_backward(gd_context *ctx,
                           const gd_tensor *x,
                           const gd_tensor *pos_ids,
                           const gd_tensor *grad_out,
                           const gd_rope_config *config,
                           gd_tensor *grad_x)
{
    gd_status st;
    gd_rope_attrs attrs;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || pos_ids == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rope_validate_forward(ctx, x, pos_ids, config, &attrs);
    if (st != GD_OK) {
        return st;
    }
    return gd_rope_backward_from_attrs(ctx, x, pos_ids, grad_out, &attrs, grad_x);
}
