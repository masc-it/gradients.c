#include <gradients/ops.h>

#include "../_shared/binary/binary_core.h"

#include <string.h>

gd_status gd_add(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    return gd_binary_apply_impl(ctx, x, y, out, gd_backend_add, GD_OP_ADD, "add", true);
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
    st = gd_binary_validate_backward(ctx, x, y, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x != NULL) {
        st = gd_binary_sum_to_shape(ctx, grad_out, x, 1.0f, grad_x);
        if (st != GD_OK) {
            return st;
        }
    }
    if (grad_y != NULL) {
        st = gd_binary_sum_to_shape(ctx, grad_out, y, 1.0f, grad_y);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}
