#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"
#include "dropout_impl.h"

#include <stdint.h>
#include <string.h>

static bool gd_dropout_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_dropout_probability_valid(float p)
{
    return p >= 0.0f && p < 1.0f;
}

static gd_status gd_dropout_validate_tensor(gd_context *ctx,
                                            const gd_tensor *x,
                                            size_t *out_count)
{
    gd_status st;
    int64_t numel;
    if (ctx == NULL || x == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_count = 0U;
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_dropout_dtype_supported(x->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout supports f16 and f32 tensors only");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout requires contiguous tensors");
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK || numel <= 0) {
        return gd_context_set_error(ctx,
                                    st == GD_OK ? GD_ERR_INVALID_ARGUMENT : st,
                                    "dropout invalid tensor shape");
    }
    if ((uint64_t)numel > (uint64_t)UINT32_MAX ||
        (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_OUT_OF_MEMORY,
                                    "dropout element count unsupported");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static bool gd_dropout_same_shape(const gd_tensor *a, const gd_tensor *b)
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

static gd_status gd_dropout_validate_grad(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *grad_out,
                                          size_t *out_count)
{
    gd_status st;
    if (ctx == NULL || x == NULL || grad_out == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_dropout_validate_tensor(ctx, x, out_count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_dropout_same_shape(x, grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout backward gradient shape mismatch");
    }
    if (!gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout backward requires contiguous grad_out");
    }
    return GD_OK;
}

static gd_status gd_dropout_validate_mask(gd_context *ctx,
                                          const gd_tensor *mask,
                                          const gd_tensor *grad_out,
                                          size_t *out_count)
{
    gd_status st;
    int64_t numel;
    uint32_t i;
    if (ctx == NULL || mask == NULL || grad_out == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_count = 0U;
    st = gd_tensor_validate(ctx, mask);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_dropout_dtype_supported(grad_out->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout backward supports f16 and f32 gradients only");
    }
    if (mask->dtype != GD_DTYPE_U8 || mask->rank != grad_out->rank) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout backward mask shape mismatch");
    }
    for (i = 0U; i < grad_out->rank; ++i) {
        if (mask->shape[i] != grad_out->shape[i]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "dropout backward mask shape mismatch");
        }
    }
    if (!gd_tensor_is_contiguous(mask) || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout backward requires contiguous mask/grad_out");
    }
    st = gd_tensor_numel(grad_out, &numel);
    if (st != GD_OK || numel <= 0) {
        return gd_context_set_error(ctx,
                                    st == GD_OK ? GD_ERR_INVALID_ARGUMENT : st,
                                    "dropout backward invalid gradient shape");
    }
    if ((uint64_t)numel > (uint64_t)UINT32_MAX ||
        (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_OUT_OF_MEMORY,
                                    "dropout backward element count unsupported");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static gd_status gd_dropout_dispatch_forward(gd_context *ctx,
                                             const gd_tensor *x,
                                             const gd_tensor *y,
                                             const gd_tensor *mask,
                                             float p,
                                             uint64_t seed)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view mv;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || mask == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(mask, &mv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout forward invalid tensor view");
    }
    st = gd_backend_dropout_forward(gd_context_backend(ctx), &xv, &yv, &mv, p, seed);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend dropout forward failed");
    }
    return GD_OK;
}

static gd_status gd_dropout_dispatch_add_forward(gd_context *ctx,
                                                  const gd_tensor *residual,
                                                  const gd_tensor *x,
                                                  const gd_tensor *y,
                                                  const gd_tensor *mask,
                                                  float p,
                                                  uint64_t seed)
{
    gd_backend_tensor_view rv;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view mv;
    gd_status st;
    if (ctx == NULL || residual == NULL || x == NULL || y == NULL || mask == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(residual, &rv) ||
        !gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(mask, &mv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout_add forward invalid tensor view");
    }
    st = gd_backend_dropout_add_forward(gd_context_backend(ctx), &rv, &xv, &yv, &mv, p, seed);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend dropout_add forward failed");
    }
    return GD_OK;
}

static gd_status gd_dropout_dispatch_backward(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *grad_out,
                                              const gd_tensor *grad_x,
                                              float p,
                                              uint64_t seed)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_status st;
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_x, &dxv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout backward invalid tensor view");
    }
    (void)xv;
    st = gd_backend_dropout_backward(gd_context_backend(ctx), &gv, &dxv, p, seed);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend dropout backward failed");
    }
    return GD_OK;
}

static gd_status gd_dropout_dispatch_backward_mask(gd_context *ctx,
                                                   const gd_tensor *mask,
                                                   const gd_tensor *grad_out,
                                                   const gd_tensor *grad_x,
                                                   float scale)
{
    gd_backend_tensor_view mv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_status st;
    if (ctx == NULL || mask == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(mask, &mv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_x, &dxv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout saved backward invalid tensor view");
    }
    st = gd_backend_dropout_backward_mask(gd_context_backend(ctx), &mv, &gv, &dxv, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend dropout saved backward failed");
    }
    return GD_OK;
}

gd_status gd_dropout(gd_context *ctx,
                     const gd_tensor *x,
                     float p,
                     bool training,
                     uint64_t seed,
                     gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_tensor mask;
    gd_dropout_attrs attrs;
    size_t count = 0U;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dropout_probability_valid(p)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout probability must satisfy 0 <= p < 1");
    }
    st = gd_dropout_validate_tensor(ctx, x, &count);
    if (st != GD_OK) {
        return st;
    }
    (void)count;
    if (!training || p == 0.0f) {
        *out = *x;
        return GD_OK;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_U8, gd_shape_make(x->rank, x->shape), 256U, &mask);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    mask.requires_grad = false;
    mask.is_leaf = false;
    st = gd_dropout_dispatch_forward(ctx, x, &y, &mask, p, seed);
    if (st != GD_OK) {
        return st;
    }
    attrs.p = p;
    attrs.scale = 1.0f / (1.0f - p);
    {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        const gd_tensor *saved[1] = {&mask};
        st = gd_autograd_record(ctx,
                                GD_OP_DROPOUT,
                                inputs,
                                1U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                saved,
                                1U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_dropout_add(gd_context *ctx,
                         const gd_tensor *residual,
                         const gd_tensor *x,
                         float p,
                         bool training,
                         uint64_t seed,
                         gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_tensor mask;
    gd_dropout_attrs attrs;
    size_t count = 0U;
    size_t residual_count = 0U;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || residual == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dropout_probability_valid(p)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout_add probability must satisfy 0 <= p < 1");
    }
    st = gd_dropout_validate_tensor(ctx, x, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_dropout_validate_tensor(ctx, residual, &residual_count);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 || residual->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "dropout_add currently supports f16 tensors only");
    }
    if (count != residual_count || !gd_dropout_same_shape(residual, x)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout_add inputs must have equal shape");
    }
    if (!training || p == 0.0f) {
        return gd_add(ctx, residual, x, out);
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_U8, gd_shape_make(x->rank, x->shape), 256U, &mask);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    mask.requires_grad = false;
    mask.is_leaf = false;
    st = gd_dropout_dispatch_add_forward(ctx, residual, x, &y, &mask, p, seed);
    if (st != GD_OK) {
        return st;
    }

    attrs.p = p;
    attrs.scale = 1.0f / (1.0f - p);
    {
        const gd_tensor *inputs[2] = {residual, x};
        gd_tensor *outputs[1] = {&y};
        const gd_tensor *saved[1] = {&mask};
        st = gd_autograd_record(ctx,
                                GD_OP_DROPOUT,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                saved,
                                1U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_dropout_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              float p,
                              uint64_t seed,
                              gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    size_t count = 0U;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dropout_probability_valid(p)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout probability must satisfy 0 <= p < 1");
    }
    st = gd_dropout_validate_grad(ctx, x, grad_out, &count);
    if (st != GD_OK) {
        return st;
    }
    (void)count;
    if (p == 0.0f) {
        *grad_x = *grad_out;
        return GD_OK;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_dropout_dispatch_backward(ctx, x, grad_out, &dx, p, seed);
    if (st != GD_OK) {
        return st;
    }
    *grad_x = dx;
    return GD_OK;
}

gd_status gd_dropout_backward_from_mask(gd_context *ctx,
                                        const gd_tensor *mask,
                                        const gd_tensor *grad_out,
                                        float scale,
                                        gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    size_t count = 0U;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || mask == NULL || grad_out == NULL || grad_x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!(scale > 0.0f) || scale != scale) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "dropout saved backward requires positive finite scale");
    }
    st = gd_dropout_validate_mask(ctx, mask, grad_out, &count);
    if (st != GD_OK) {
        return st;
    }
    (void)count;
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         grad_out->dtype,
                         gd_shape_make(grad_out->rank, grad_out->shape),
                         256U,
                         &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_dropout_dispatch_backward_mask(ctx, mask, grad_out, &dx, scale);
    if (st != GD_OK) {
        return st;
    }
    *grad_x = dx;
    return GD_OK;
}
