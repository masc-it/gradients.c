#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"
#include "sigmoid_impl.h"

#include <string.h>

static bool gd_sigmoid_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static gd_status gd_sigmoid_validate(gd_context *ctx, const gd_tensor *x)
{
    gd_status st;
    if (ctx == NULL || x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_sigmoid_dtype_supported(x->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sigmoid supports f16 and f32 tensors only");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sigmoid requires contiguous input");
    }
    return GD_OK;
}

static gd_status gd_sigmoid_validate_like(gd_context *ctx,
                                          const gd_tensor *ref,
                                          const gd_tensor *tensor,
                                          const char *what)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || ref == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (tensor->dtype != ref->dtype || tensor->rank != ref->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, what);
    }
    for (i = 0U; i < ref->rank; ++i) {
        if (tensor->shape[i] != ref->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, what);
        }
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sigmoid backward requires contiguous tensors");
    }
    return GD_OK;
}

static gd_status gd_sigmoid_backward_validate(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *grad_out)
{
    gd_status st;
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    return gd_sigmoid_validate_like(ctx, x, grad_out, "sigmoid backward gradient shape mismatch");
}

gd_status gd_sigmoid(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, x->rank, x->shape, 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sigmoid invalid tensor view");
    }
    st = gd_backend_sigmoid(gd_context_backend(ctx), &xv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sigmoid failed");
    }
    {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_SIGMOID,
                                inputs,
                                1U,
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

gd_status gd_sigmoid_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_backward_validate(ctx, x, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, x->rank, x->shape, 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(&dx, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sigmoid backward invalid tensor view");
    }
    st = gd_backend_sigmoid_backward(gd_context_backend(ctx), &xv, &gv, &dxv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sigmoid backward failed");
    }
    *grad_x = dx;
    return GD_OK;
}

gd_status gd_sigmoid_backward_from_output(gd_context *ctx,
                                          const gd_tensor *sigmoid_out,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || sigmoid_out == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sigmoid_validate(ctx, sigmoid_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_sigmoid_validate_like(ctx,
                                  sigmoid_out,
                                  grad_out,
                                  "sigmoid saved backward gradient shape mismatch");
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         sigmoid_out->dtype,
                         sigmoid_out->rank,
                         sigmoid_out->shape,
                         256U,
                         &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(sigmoid_out, &yv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(&dx, &dxv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sigmoid saved backward invalid tensor view");
    }
    st = gd_backend_sigmoid_backward_from_output(gd_context_backend(ctx), &yv, &gv, &dxv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sigmoid saved backward failed");
    }
    *grad_x = dx;
    return GD_OK;
}
