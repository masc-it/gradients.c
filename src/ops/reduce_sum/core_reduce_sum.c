#include <gradients/ops.h>

#include "../_shared/reduce/reduce_core.h"

gd_status gd_reduce_sum(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    return gd_reduce_all_forward_impl(ctx, x, out, GD_OP_REDUCE_SUM, 1.0f);
}

gd_status gd_reduce_sum_axis(gd_context *ctx,
                             const gd_tensor *x,
                             int32_t axis,
                             bool keepdims,
                             gd_tensor *out)
{
    return gd_reduce_axis_forward_impl(ctx, x, out, GD_OP_REDUCE_SUM, axis, keepdims, 1.0f);
}

gd_status gd_reduce_sum_backward(gd_context *ctx,
                                 const gd_tensor *x,
                                 const gd_tensor *grad_out,
                                 gd_tensor *grad_x)
{
    return gd_reduce_all_backward_impl(ctx, x, grad_out, 1.0f, grad_x);
}

gd_status gd_reduce_sum_axis_backward(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *grad_out,
                                      int32_t axis,
                                      bool keepdims,
                                      gd_tensor *grad_x)
{
    return gd_reduce_axis_backward_impl(ctx, x, grad_out, axis, keepdims, 1.0f, grad_x);
}
