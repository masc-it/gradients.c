#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <string.h>

static gd_status gd_relu_validate(gd_context *ctx, const gd_tensor *x)
{
    gd_status st;
    if (ctx == NULL || x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 && x->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "relu currently supports f16/f32 tensors only");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "relu requires contiguous input");
    }
    return GD_OK;
}

static gd_status gd_relu_backward_validate(gd_context *ctx,
                                           const gd_tensor *x,
                                           const gd_tensor *grad_out)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_relu_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != x->dtype || grad_out->rank != x->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "relu backward gradient shape mismatch");
    }
    for (i = 0U; i < x->rank; ++i) {
        if (grad_out->shape[i] != x->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "relu backward gradient shape mismatch");
        }
    }
    if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "relu backward requires contiguous grad_out");
    }
    return GD_OK;
}

gd_status gd_relu(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
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
    st = gd_relu_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, x->rank, x->shape, 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "relu invalid tensor view");
    }
    st = gd_backend_relu(gd_context_backend(ctx), &xv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend relu failed");
    }
    {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_RELU,
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

gd_status gd_relu_backward(gd_context *ctx,
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
    st = gd_relu_backward_validate(ctx, x, grad_out);
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
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "relu backward invalid tensor view");
    }
    st = gd_backend_relu_backward(gd_context_backend(ctx), &xv, &gv, &dxv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend relu backward failed");
    }
    *grad_x = dx;
    return GD_OK;
}
