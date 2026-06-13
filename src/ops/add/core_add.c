#include <gradients/ops.h>

#include "../_shared/binary/binary_core.h"

#include <string.h>

static gd_status gd_add_validate_f16(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *y,
                                      const gd_tensor *grad_out)
{
    if (ctx == NULL || x == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != GD_DTYPE_F16 || y->dtype != GD_DTYPE_F16 ||
        (grad_out != NULL && grad_out->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "add is optimized for f16 tensors only");
    }
    return GD_OK;
}

gd_status gd_add(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    gd_status st = gd_add_validate_f16(ctx, x, y, NULL);
    if (st != GD_OK) {
        return st;
    }
    return gd_binary_apply_impl(ctx, x, y, out, gd_backend_add, GD_OP_ADD, "add", true);
}

static void gd_add_alias_grad(const gd_tensor *grad_out, gd_tensor *grad)
{
    *grad = *grad_out;
    grad->requires_grad = false;
    grad->is_leaf = false;
}

gd_status gd_add_backward(gd_context *ctx,
                          const gd_tensor *x,
                          const gd_tensor *y,
                          const gd_tensor *grad_out,
                          gd_tensor *grad_x,
                          gd_tensor *grad_y)
{
    gd_status st;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_y != NULL) {
        memset(grad_y, 0, sizeof(*grad_y));
    }
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_add_validate_f16(ctx, x, y, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_binary_validate_backward(ctx, x, y, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x != NULL) {
        if (gd_binary_shapes_equal(grad_out, x)) {
            gd_add_alias_grad(grad_out, grad_x);
        } else {
            st = gd_binary_sum_to_shape(ctx, grad_out, x, 1.0f, grad_x);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    if (grad_y != NULL) {
        if (gd_binary_shapes_equal(grad_out, y)) {
            gd_add_alias_grad(grad_out, grad_y);
        } else {
            st = gd_binary_sum_to_shape(ctx, grad_out, y, 1.0f, grad_y);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    return GD_OK;
}
