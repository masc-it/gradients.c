#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"
#include "powlu_impl.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static gd_status gd_powlu_validate_m(gd_context *ctx, float m)
{
    if (!isfinite(m) || m <= 0.0f || m >= 10.0f) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "powlu m must satisfy 0 < m < 10");
    }
    return GD_OK;
}

static bool gd_powlu_same_shape(const gd_tensor *x1, const gd_tensor *x2)
{
    uint32_t axis;
    if (x1 == NULL || x2 == NULL || x1->rank != x2->rank) {
        return false;
    }
    for (axis = 0U; axis < x1->rank; ++axis) {
        if (x1->shape[axis] != x2->shape[axis]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_powlu_validate_inputs(gd_context *ctx,
                                          const gd_tensor *x1,
                                          const gd_tensor *x2,
                                          float m,
                                          size_t *out_count)
{
    gd_status st;
    int64_t numel;
    if (ctx == NULL || x1 == NULL || x2 == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_count = 0U;
    st = gd_powlu_validate_m(ctx, m);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, x1);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, x2);
    if (st != GD_OK) {
        return st;
    }
    if (x1->dtype != GD_DTYPE_F16 || x2->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "powlu currently supports f16 tensors only");
    }
    if (!gd_powlu_same_shape(x1, x2)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "powlu inputs must have equal shape");
    }
    if (!gd_tensor_is_contiguous(x1) || !gd_tensor_is_contiguous(x2)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "powlu requires contiguous inputs");
    }
    st = gd_tensor_numel(x1, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "powlu invalid input shape");
    }
    if (numel <= 0 || (uint64_t)numel > (uint64_t)UINT32_MAX ||
        (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx,
                                    numel <= 0 ? GD_ERR_INVALID_ARGUMENT : GD_ERR_OUT_OF_MEMORY,
                                    "powlu element count unsupported");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static gd_status gd_powlu_validate_grad_out(gd_context *ctx,
                                            const gd_tensor *shape_like,
                                            const gd_tensor *grad_out)
{
    gd_status st;
    if (ctx == NULL || shape_like == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != shape_like->dtype || !gd_powlu_same_shape(shape_like, grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "powlu backward gradient shape/dtype mismatch");
    }
    if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "powlu backward requires contiguous grad_out");
    }
    return GD_OK;
}

static gd_status gd_powlu_dispatch_forward(gd_context *ctx,
                                           const gd_tensor *x1,
                                           const gd_tensor *x2,
                                           gd_tensor *out,
                                           float m)
{
    gd_backend_tensor_view x1v;
    gd_backend_tensor_view x2v;
    gd_backend_tensor_view ov;
    gd_status st;
    if (!gd_op_tensor_view_from_tensor(x1, &x1v) ||
        !gd_op_tensor_view_from_tensor(x2, &x2v) ||
        !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "powlu invalid tensor view");
    }
    st = gd_backend_powlu_forward(gd_context_backend(ctx), &x1v, &x2v, &ov, m);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend powlu forward failed");
    }
    return GD_OK;
}

static gd_status gd_powlu_dispatch_backward(gd_context *ctx,
                                            const gd_tensor *x1,
                                            const gd_tensor *x2,
                                            const gd_tensor *grad_out,
                                            gd_tensor *grad_x1,
                                            gd_tensor *grad_x2,
                                            float m)
{
    gd_backend_tensor_view x1v;
    gd_backend_tensor_view x2v;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dx1v;
    gd_backend_tensor_view dx2v;
    gd_status st;
    memset(&dx1v, 0, sizeof(dx1v));
    memset(&dx2v, 0, sizeof(dx2v));
    if (!gd_op_tensor_view_from_tensor(x1, &x1v) ||
        !gd_op_tensor_view_from_tensor(x2, &x2v) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        (grad_x1 != NULL && !gd_op_tensor_view_from_tensor(grad_x1, &dx1v)) ||
        (grad_x2 != NULL && !gd_op_tensor_view_from_tensor(grad_x2, &dx2v))) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "powlu backward invalid tensor view");
    }
    st = gd_backend_powlu_backward(gd_context_backend(ctx),
                                   &x1v,
                                   &x2v,
                                   &gv,
                                   grad_x1 != NULL ? &dx1v : NULL,
                                   grad_x2 != NULL ? &dx2v : NULL,
                                   m);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend powlu backward failed");
    }
    return GD_OK;
}

gd_status gd_powlu(gd_context *ctx,
                   const gd_tensor *x1,
                   const gd_tensor *x2,
                   float m,
                   gd_tensor *out)
{
    gd_status st;
    gd_tensor result;
    size_t count = 0U;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x1 == NULL || x2 == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_powlu_validate_inputs(ctx, x1, x2, m, &count);
    if (st != GD_OK) {
        return st;
    }
    (void)count;
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         x1->dtype,
                         gd_shape_make(x1->rank, x1->shape),
                         256U,
                         &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    st = gd_powlu_dispatch_forward(ctx, x1, x2, &result, m);
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_powlu_attrs attrs = {.m = m};
        const gd_tensor *inputs[2] = {x1, x2};
        gd_tensor *outputs[1] = {&result};
        st = gd_autograd_record(ctx,
                                GD_OP_POWLU,
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
    *out = result;
    return GD_OK;
}

gd_status gd_powlu_backward(gd_context *ctx,
                            const gd_tensor *x1,
                            const gd_tensor *x2,
                            const gd_tensor *grad_out,
                            float m,
                            gd_tensor *grad_x1,
                            gd_tensor *grad_x2)
{
    gd_status st;
    gd_tensor dx1;
    gd_tensor dx2;
    size_t count = 0U;
    if (grad_x1 != NULL) {
        memset(grad_x1, 0, sizeof(*grad_x1));
    }
    if (grad_x2 != NULL) {
        memset(grad_x2, 0, sizeof(*grad_x2));
    }
    if (ctx == NULL || x1 == NULL || x2 == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_powlu_validate_inputs(ctx, x1, x2, m, &count);
    if (st != GD_OK) {
        return st;
    }
    (void)count;
    st = gd_powlu_validate_grad_out(ctx, x1, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x1 == NULL && grad_x2 == NULL) {
        return GD_OK;
    }
    memset(&dx1, 0, sizeof(dx1));
    memset(&dx2, 0, sizeof(dx2));
    if (grad_x1 != NULL) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             x1->dtype,
                             gd_shape_make(x1->rank, x1->shape),
                             256U,
                             &dx1);
        if (st != GD_OK) {
            return st;
        }
        dx1.is_leaf = false;
    }
    if (grad_x2 != NULL) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             x2->dtype,
                             gd_shape_make(x2->rank, x2->shape),
                             256U,
                             &dx2);
        if (st != GD_OK) {
            return st;
        }
        dx2.is_leaf = false;
    }
    st = gd_powlu_dispatch_backward(ctx,
                                    x1,
                                    x2,
                                    grad_out,
                                    grad_x1 != NULL ? &dx1 : NULL,
                                    grad_x2 != NULL ? &dx2 : NULL,
                                    m);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x1 != NULL) {
        *grad_x1 = dx1;
    }
    if (grad_x2 != NULL) {
        *grad_x2 = dx2;
    }
    return GD_OK;
}
