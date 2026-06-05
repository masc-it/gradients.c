#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"
#include "metal_mse_types.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static gd_status gd_mse_validate_inputs(gd_context *ctx,
                                        const gd_tensor *x,
                                        const gd_tensor *y,
                                        size_t *count_out)
{
    gd_status st;
    int64_t numel;
    uint32_t i;
    if (ctx == NULL || x == NULL || y == NULL || count_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *count_out = 0U;
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, y);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 && x->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "mse supports f16/f32 tensors only");
    }
    if (y->dtype != x->dtype) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse input dtypes must match");
    }
    if (x->rank != y->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse input shapes must match");
    }
    for (i = 0U; i < x->rank; ++i) {
        if (x->shape[i] != y->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "mse input shapes must match");
        }
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(y)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "mse requires contiguous inputs");
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "mse invalid input shape");
    }
    if (numel <= 0 || (uint64_t)numel > (uint64_t)UINT32_MAX ||
        (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx,
                                    numel <= 0 ? GD_ERR_INVALID_ARGUMENT : GD_ERR_OUT_OF_MEMORY,
                                    "mse element count unsupported");
    }
    *count_out = (size_t)numel;
    return GD_OK;
}

static gd_status gd_mse_validate_grad_out(gd_context *ctx, const gd_tensor *grad_out)
{
    gd_status st;
    if (ctx == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F32 || grad_out->rank != 0U ||
        !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse backward requires scalar f32 grad_out");
    }
    return GD_OK;
}

static gd_status gd_mse_dispatch_forward(gd_context *ctx,
                                         const gd_tensor *x,
                                         const gd_tensor *y,
                                         const gd_tensor *out,
                                         size_t chunk_size,
                                         float scale)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view ov;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || out == NULL ||
        chunk_size == 0U || chunk_size > (size_t)UINT32_MAX || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse forward invalid tensor view");
    }
    st = gd_backend_mse_forward(gd_context_backend(ctx),
                                &xv,
                                &yv,
                                &ov,
                                (uint64_t)chunk_size,
                                scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend mse forward failed");
    }
    return GD_OK;
}

static gd_status gd_mse_dispatch_reduce(gd_context *ctx,
                                        const gd_tensor *partial,
                                        const gd_tensor *out,
                                        float scale)
{
    gd_backend_tensor_view pv;
    gd_backend_tensor_view ov;
    gd_status st;
    if (ctx == NULL || partial == NULL || out == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(partial, &pv) || !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse reduce invalid tensor view");
    }
    st = gd_backend_reduce_contiguous(gd_context_backend(ctx), &pv, &ov, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend mse reduce failed");
    }
    return GD_OK;
}

static gd_status gd_mse_dispatch_backward(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *y,
                                          const gd_tensor *grad_out,
                                          const gd_tensor *grad_x,
                                          const gd_tensor *grad_y,
                                          float scale)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_backend_tensor_view dyv;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL ||
        (grad_x == NULL && grad_y == NULL) || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&dxv, 0, sizeof(dxv));
    memset(&dyv, 0, sizeof(dyv));
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        (grad_x != NULL && !gd_op_tensor_view_from_tensor(grad_x, &dxv)) ||
        (grad_y != NULL && !gd_op_tensor_view_from_tensor(grad_y, &dyv))) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "mse backward invalid tensor view");
    }
    st = gd_backend_mse_backward(gd_context_backend(ctx),
                                 &xv,
                                 &yv,
                                 &gv,
                                 grad_x != NULL ? &dxv : NULL,
                                 grad_y != NULL ? &dyv : NULL,
                                 scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend mse backward failed");
    }
    return GD_OK;
}

gd_status gd_mse(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    gd_status st;
    gd_tensor result;
    size_t count = 0U;
    size_t chunk_count;
    float inv_count;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_mse_validate_inputs(ctx, x, y, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(0U, NULL), 256U, &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    inv_count = 1.0f / (float)count;
    chunk_count = (count + (size_t)GD_METAL_MSE_CHUNK_SIZE - 1U) /
                  (size_t)GD_METAL_MSE_CHUNK_SIZE;
    if (chunk_count <= 1U) {
        st = gd_mse_dispatch_forward(ctx, x, y, &result, count, inv_count);
    } else {
        gd_tensor partial;
        int64_t partial_shape[1];
        if (chunk_count > (size_t)INT64_MAX) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "mse partial count overflow");
        }
        partial_shape[0] = (int64_t)chunk_count;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, partial_shape),
                             256U,
                             &partial);
        if (st == GD_OK) {
            partial.is_leaf = false;
            st = gd_mse_dispatch_forward(ctx,
                                         x,
                                         y,
                                         &partial,
                                         (size_t)GD_METAL_MSE_CHUNK_SIZE,
                                         1.0f);
        }
        if (st == GD_OK) {
            st = gd_mse_dispatch_reduce(ctx, &partial, &result, inv_count);
        }
    }
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        inputs[0] = x;
        inputs[1] = y;
        outputs[0] = &result;
        st = gd_autograd_record(ctx,
                                GD_OP_MSE,
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
    *out = result;
    return GD_OK;
}

gd_status gd_mse_backward(gd_context *ctx,
                          const gd_tensor *x,
                          const gd_tensor *y,
                          const gd_tensor *grad_out,
                          gd_tensor *grad_x,
                          gd_tensor *grad_y)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dy;
    size_t count = 0U;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_y != NULL) {
        memset(grad_y, 0, sizeof(*grad_y));
    }
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_mse_validate_inputs(ctx, x, y, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_mse_validate_grad_out(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL && grad_y == NULL) {
        return GD_OK;
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dy, 0, sizeof(dy));
    if (grad_x != NULL) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, gd_shape_make(x->rank, x->shape), 256U, &dx);
        if (st != GD_OK) {
            return st;
        }
        dx.is_leaf = false;
    }
    if (grad_y != NULL) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, y->dtype, gd_shape_make(y->rank, y->shape), 256U, &dy);
        if (st != GD_OK) {
            return st;
        }
        dy.is_leaf = false;
    }
    st = gd_mse_dispatch_backward(ctx,
                                  x,
                                  y,
                                  grad_out,
                                  grad_x != NULL ? &dx : NULL,
                                  grad_y != NULL ? &dy : NULL,
                                  2.0f / (float)count);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x != NULL) {
        *grad_x = dx;
    }
    if (grad_y != NULL) {
        *grad_y = dy;
    }
    return GD_OK;
}
