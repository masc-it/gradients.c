#include <gradients/ops.h>

#include "../_shared/binary/binary_core.h"

#include <string.h>

gd_status gd_mul(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    return gd_binary_apply_impl(ctx, x, y, out, gd_backend_mul, GD_OP_MUL, "mul", true);
}

gd_status gd_mul_backward(gd_context *ctx,
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
        gd_tensor full_dx;
        st = gd_binary_apply_impl(ctx, grad_out, y, &full_dx, gd_backend_mul, GD_OP_MUL, "mul backward dx", false);
        if (st != GD_OK) {
            return st;
        }
        if (gd_binary_shapes_equal(&full_dx, x)) {
            *grad_x = full_dx;
        } else {
            st = gd_binary_sum_to_shape(ctx, &full_dx, x, 1.0f, grad_x);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    if (grad_y != NULL) {
        gd_tensor full_dy;
        st = gd_binary_apply_impl(ctx, grad_out, x, &full_dy, gd_backend_mul, GD_OP_MUL, "mul backward dy", false);
        if (st != GD_OK) {
            return st;
        }
        if (gd_binary_shapes_equal(&full_dy, y)) {
            *grad_y = full_dy;
        } else {
            st = gd_binary_sum_to_shape(ctx, &full_dy, y, 1.0f, grad_y);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    return GD_OK;
}
