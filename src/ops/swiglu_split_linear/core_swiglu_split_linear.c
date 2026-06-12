#include <gradients/ops.h>

#include "swiglu_split_linear_impl.h"
#include "../autograd_impl.h"
#include "../linear/linear_impl.h"
#include "../op_common.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct gd_swiglu_split_linear_shape_info {
    uint32_t rank;
    int64_t out_shape[GD_MAX_DIMS];
    int64_t act_shape[GD_MAX_DIMS];
    int64_t rows;
    int64_t hidden;
    int64_t out_cols;
} gd_swiglu_split_linear_shape_info;

static gd_status gd_swiglu_split_linear_build_shape(gd_context *ctx,
                                                    const gd_tensor *x12,
                                                    const gd_tensor *w,
                                                    const gd_tensor *bias,
                                                    gd_swiglu_split_linear_shape_info *info)
{
    uint32_t axis;
    int64_t rows = 1;
    if (ctx == NULL || x12 == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(info, 0, sizeof(*info));
    if (x12->dtype != GD_DTYPE_F16 || w->dtype != GD_DTYPE_F16 ||
        (bias != NULL && bias->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "swiglu_split_linear currently supports f16 tensors only");
    }
    if (x12->rank < 1U || x12->rank > GD_MAX_DIMS || w->rank != 2U ||
        (bias != NULL && bias->rank != 1U)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear expects x12 [..., 2H], w [H, N], bias [N]");
    }
    if (x12->shape[x12->rank - 1U] <= 0 ||
        (x12->shape[x12->rank - 1U] & 1) != 0 || w->shape[0] <= 0 ||
        w->shape[1] <= 0) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear shape mismatch");
    }
    info->rank = x12->rank;
    info->hidden = x12->shape[x12->rank - 1U] / 2;
    info->out_cols = w->shape[1];
    if (info->hidden != w->shape[0] || (bias != NULL && bias->shape[0] != info->out_cols)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear shape mismatch");
    }
    if (info->hidden > (int64_t)UINT32_MAX || info->out_cols > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_OUT_OF_MEMORY,
                                    "swiglu_split_linear matrix dimension overflow");
    }
    for (axis = 0U; axis + 1U < x12->rank; ++axis) {
        if (x12->shape[axis] <= 0 || gd_linear_i64_mul_overflow(rows, x12->shape[axis], &rows)) {
            return gd_context_set_error(ctx,
                                        GD_ERR_OUT_OF_MEMORY,
                                        "swiglu_split_linear flattened row count overflow");
        }
        info->out_shape[axis] = x12->shape[axis];
        info->act_shape[axis] = x12->shape[axis];
    }
    if (rows > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_OUT_OF_MEMORY,
                                    "swiglu_split_linear flattened row count overflow");
    }
    info->out_shape[x12->rank - 1U] = info->out_cols;
    info->act_shape[x12->rank - 1U] = info->hidden;
    info->rows = rows;
    return GD_OK;
}

static gd_status gd_swiglu_split_linear_validate_common(gd_context *ctx,
                                                        const gd_tensor *x12,
                                                        const gd_tensor *w,
                                                        const gd_tensor *bias,
                                                        gd_swiglu_split_linear_shape_info *info)
{
    gd_status st;
    if (ctx == NULL || x12 == NULL || w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x12);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, w);
    if (st != GD_OK) {
        return st;
    }
    if (bias != NULL) {
        st = gd_tensor_validate(ctx, bias);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_swiglu_split_linear_build_shape(ctx, x12, w, bias, info);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(x12) || w->strides[1] != 1 || w->strides[0] <= 0 ||
        (bias != NULL && bias->strides[0] != 1)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "swiglu_split_linear requires contiguous x12 and row-strided weight/bias");
    }
    return GD_OK;
}

static gd_status gd_swiglu_split_linear_validate_activation(
    gd_context *ctx,
    const gd_tensor *activation,
    const gd_swiglu_split_linear_shape_info *info)
{
    uint32_t axis;
    gd_status st;
    if (ctx == NULL || activation == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, activation);
    if (st != GD_OK) {
        return st;
    }
    if (activation->dtype != GD_DTYPE_F16 || activation->rank != info->rank ||
        !gd_tensor_is_contiguous(activation)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear activation shape/layout mismatch");
    }
    for (axis = 0U; axis < info->rank; ++axis) {
        if (activation->shape[axis] != info->act_shape[axis]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "swiglu_split_linear activation shape/layout mismatch");
        }
    }
    return GD_OK;
}

static gd_status gd_swiglu_split_linear_validate_grad_out(
    gd_context *ctx,
    const gd_tensor *grad_out,
    const gd_swiglu_split_linear_shape_info *info)
{
    uint32_t axis;
    gd_status st;
    if (ctx == NULL || grad_out == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F16 || grad_out->rank != info->rank ||
        grad_out->strides[grad_out->rank - 1U] != 1) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear backward grad_out shape/layout mismatch");
    }
    for (axis = 0U; axis < info->rank; ++axis) {
        if (grad_out->shape[axis] != info->out_shape[axis]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "swiglu_split_linear backward grad_out shape/layout mismatch");
        }
    }
    if (grad_out->rank == 2U) {
        if (grad_out->strides[0] <= 0) {
            return gd_context_set_error(ctx,
                                        GD_ERR_UNSUPPORTED,
                                        "swiglu_split_linear backward requires row-strided grad_out");
        }
    } else if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "swiglu_split_linear backward rank-N grad_out must be contiguous");
    }
    return GD_OK;
}

static gd_status gd_swiglu_split_linear_recompute_activation(gd_context *ctx,
                                                             const gd_tensor *x12,
                                                             gd_tensor *activation)
{
    return gd_swiglu_split(ctx, x12, activation);
}

static bool gd_swiglu_split_linear_recompute_enabled(void)
{
    const char *env = getenv("GD_SWIGLU_SPLIT_LINEAR_RECOMPUTE");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

static gd_status gd_swiglu_split_linear_ensure_activation(gd_context *ctx,
                                                          const gd_tensor *x12,
                                                          const gd_tensor *provided,
                                                          gd_tensor *computed,
                                                          const gd_tensor **out_activation)
{
    gd_status st;
    if (provided != NULL) {
        *out_activation = provided;
        return GD_OK;
    }
    st = gd_swiglu_split_linear_recompute_activation(ctx, x12, computed);
    if (st != GD_OK) {
        return st;
    }
    *out_activation = computed;
    return GD_OK;
}

gd_status gd_swiglu_split_linear(gd_context *ctx,
                                 const gd_tensor *x12,
                                 const gd_tensor *w,
                                 const gd_tensor *bias,
                                 gd_tensor *out)
{
    gd_status st;
    gd_status restore_st = GD_OK;
    gd_swiglu_split_linear_shape_info info;
    gd_tensor activation;
    gd_tensor y;
    bool was_recording;
    bool save_activation;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&activation, 0, sizeof(activation));
    memset(&y, 0, sizeof(y));
    if (ctx == NULL || x12 == NULL || w == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_swiglu_split_linear_validate_common(ctx, x12, w, bias, &info);
    if (st != GD_OK) {
        return st;
    }

    was_recording = gd_is_grad_enabled(ctx);
    save_activation = was_recording && !gd_swiglu_split_linear_recompute_enabled();
    if (was_recording) {
        st = gd_set_grad_enabled(ctx, false);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_swiglu_split(ctx, x12, &activation);
    if (st == GD_OK) {
        st = gd_linear(ctx, &activation, w, bias, &y);
    }
    if (was_recording) {
        restore_st = gd_set_grad_enabled(ctx, true);
    }
    if (st != GD_OK) {
        return st;
    }
    if (restore_st != GD_OK) {
        return restore_st;
    }
    if (!save_activation) {
        st = gd_context_free_span(ctx, &activation.storage);
        if (st != GD_OK) {
            return st;
        }
    }

    y.is_leaf = false;
    {
        const gd_tensor *inputs[3] = {x12, w, bias};
        gd_tensor *outputs[1] = {&y};
        const gd_tensor *saved[1] = {&activation};
        st = gd_autograd_record(ctx,
                                GD_OP_SWIGLU_SPLIT_LINEAR,
                                inputs,
                                bias != NULL ? 3U : 2U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                save_activation ? saved : NULL,
                                save_activation ? 1U : 0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_swiglu_split_linear_backward_with_activation(gd_context *ctx,
                                                          const gd_tensor *x12,
                                                          const gd_tensor *activation,
                                                          const gd_tensor *w,
                                                          const gd_tensor *bias,
                                                          const gd_tensor *grad_out,
                                                          gd_tensor *grad_x12,
                                                          gd_tensor *grad_w,
                                                          gd_tensor *grad_bias)
{
    gd_status st;
    gd_swiglu_split_linear_shape_info info;
    gd_tensor dx12;
    gd_tensor dw;
    gd_tensor db;
    gd_tensor dact;
    gd_tensor computed_activation;
    const gd_tensor *act = activation;
    bool need_x12 = grad_x12 != NULL;
    bool need_w = grad_w != NULL;
    bool need_bias = grad_bias != NULL;
    if (grad_x12 != NULL) {
        memset(grad_x12, 0, sizeof(*grad_x12));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    if (grad_bias != NULL) {
        memset(grad_bias, 0, sizeof(*grad_bias));
    }
    memset(&dx12, 0, sizeof(dx12));
    memset(&dw, 0, sizeof(dw));
    memset(&db, 0, sizeof(db));
    memset(&dact, 0, sizeof(dact));
    memset(&computed_activation, 0, sizeof(computed_activation));
    if (ctx == NULL || x12 == NULL || w == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (need_bias && bias == NULL) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "swiglu_split_linear grad_bias requested without bias");
    }
    st = gd_swiglu_split_linear_validate_common(ctx, x12, w, bias, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_swiglu_split_linear_validate_grad_out(ctx, grad_out, &info);
    if (st != GD_OK) {
        return st;
    }
    if (!need_x12 && !need_w && !need_bias) {
        return GD_OK;
    }
    if (act != NULL) {
        st = gd_swiglu_split_linear_validate_activation(ctx, act, &info);
        if (st != GD_OK) {
            return st;
        }
    }

    if (need_w || need_bias) {
        st = gd_swiglu_split_linear_ensure_activation(ctx, x12, act, &computed_activation, &act);
        if (st != GD_OK) {
            return st;
        }
        st = gd_linear_backward(ctx,
                                act,
                                w,
                                bias,
                                grad_out,
                                NULL,
                                need_w ? &dw : NULL,
                                need_bias ? &db : NULL);
        if (st != GD_OK) {
            return st;
        }
    }

    if (need_x12) {
        st = gd_swiglu_split_linear_ensure_activation(ctx, x12, act, &computed_activation, &act);
        if (st != GD_OK) {
            return st;
        }
        st = gd_linear_backward(ctx, act, w, bias, grad_out, &dact, NULL, NULL);
        if (st != GD_OK) {
            return st;
        }
        st = gd_swiglu_split_backward(ctx, x12, &dact, &dx12);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "swiglu_split_linear backward x12 failed");
        }
    }

    if (need_x12) {
        *grad_x12 = dx12;
    }
    if (need_w) {
        *grad_w = dw;
    }
    if (need_bias) {
        *grad_bias = db;
    }
    return GD_OK;
}

gd_status gd_swiglu_split_linear_backward(gd_context *ctx,
                                          const gd_tensor *x12,
                                          const gd_tensor *w,
                                          const gd_tensor *bias,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x12,
                                          gd_tensor *grad_w,
                                          gd_tensor *grad_bias)
{
    return gd_swiglu_split_linear_backward_with_activation(ctx,
                                                           x12,
                                                           NULL,
                                                           w,
                                                           bias,
                                                           grad_out,
                                                           grad_x12,
                                                           grad_w,
                                                           grad_bias);
}
