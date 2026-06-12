#include <gradients/ops.h>

#include "../_shared/reduce/reduce_core.h"

gd_status gd_reduce_mean(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    gd_status st;
    size_t count;
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_tensor_count(ctx, x, &count);
    if (st != GD_OK) {
        return st;
    }
    return gd_reduce_all_forward_impl_dtype(ctx,
                                            x,
                                            out,
                                            GD_OP_REDUCE_MEAN,
                                            x->dtype == GD_DTYPE_F16 ? GD_DTYPE_F32 : x->dtype,
                                            1.0f / (float)count);
}

gd_status gd_reduce_mean_axis(gd_context *ctx,
                              const gd_tensor *x,
                              int32_t axis,
                              bool keepdims,
                              gd_tensor *out)
{
    gd_status st;
    uint32_t normalized_axis;
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_normalize_axis(ctx, x, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    return gd_reduce_axis_forward_impl(ctx,
                                       x,
                                       out,
                                       GD_OP_REDUCE_MEAN,
                                       (int32_t)normalized_axis,
                                       keepdims,
                                       1.0f / (float)x->shape[normalized_axis]);
}

gd_status gd_reduce_mean_backward(gd_context *ctx,
                                  const gd_tensor *x,
                                  const gd_tensor *grad_out,
                                  gd_tensor *grad_x)
{
    gd_status st;
    size_t count;
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_tensor_count(ctx, x, &count);
    if (st != GD_OK) {
        return st;
    }
    return gd_reduce_all_backward_impl(ctx, x, grad_out, 1.0f / (float)count, grad_x);
}

gd_status gd_reduce_mean_axis_backward(gd_context *ctx,
                                       const gd_tensor *x,
                                       const gd_tensor *grad_out,
                                       int32_t axis,
                                       bool keepdims,
                                       gd_tensor *grad_x)
{
    gd_status st;
    uint32_t normalized_axis;
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_normalize_axis(ctx, x, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    return gd_reduce_axis_backward_impl(ctx,
                                        x,
                                        grad_out,
                                        (int32_t)normalized_axis,
                                        keepdims,
                                        1.0f / (float)x->shape[normalized_axis],
                                        grad_x);
}
