#include <gradients/ops.h>

#include "../_shared/unary/unary_core.h"
#include "sigmoid_impl.h"

static bool gd_sigmoid_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static const gd_unary_op_spec GD_SIGMOID_UNARY_SPEC = {
    .name = "sigmoid",
    .op = GD_OP_SIGMOID,
    .dtype_supported = gd_sigmoid_dtype_supported,
    .backend_forward = gd_backend_sigmoid,
    .backend_backward = gd_backend_sigmoid_backward,
    .unsupported_dtype_message = "sigmoid supports f16 and f32 tensors only",
    .contiguous_input_message = "sigmoid requires contiguous input",
    .grad_shape_message = "sigmoid backward gradient shape mismatch",
    .grad_contiguous_message = "sigmoid backward requires contiguous tensors",
    .forward_view_message = "sigmoid invalid tensor view",
    .backward_view_message = "sigmoid backward invalid tensor view",
    .backend_forward_failed_message = "backend sigmoid failed",
    .backend_backward_failed_message = "backend sigmoid backward failed",
};

gd_status gd_sigmoid(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    return gd_unary_apply_impl(ctx, x, out, &GD_SIGMOID_UNARY_SPEC);
}

gd_status gd_sigmoid_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              gd_tensor *grad_x)
{
    return gd_unary_backward_impl(ctx, x, grad_out, grad_x, &GD_SIGMOID_UNARY_SPEC);
}

gd_status gd_sigmoid_backward_from_output(gd_context *ctx,
                                          const gd_tensor *sigmoid_out,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x)
{
    return gd_unary_backward_impl_with_backend(ctx,
                                               sigmoid_out,
                                               grad_out,
                                               grad_x,
                                               &GD_SIGMOID_UNARY_SPEC,
                                               gd_backend_sigmoid_backward_from_output,
                                               "sigmoid saved backward gradient shape mismatch",
                                               "sigmoid saved backward invalid tensor view",
                                               "backend sigmoid saved backward failed");
}
