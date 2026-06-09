#include <gradients/ops.h>

#include "../_shared/unary/unary_core.h"

static bool gd_relu_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16;
}

static const gd_unary_op_spec GD_RELU_UNARY_SPEC = {
    .name = "relu",
    .op = GD_OP_RELU,
    .dtype_supported = gd_relu_dtype_supported,
    .backend_forward = gd_backend_relu,
    .backend_backward = gd_backend_relu_backward,
    .unsupported_dtype_message = "relu supports f16 tensors only",
    .contiguous_input_message = "relu requires contiguous input",
    .grad_shape_message = "relu backward gradient shape mismatch",
    .grad_contiguous_message = "relu backward requires contiguous grad_out",
    .forward_view_message = "relu invalid tensor view",
    .backward_view_message = "relu backward invalid tensor view",
    .backend_forward_failed_message = "backend relu failed",
    .backend_backward_failed_message = "backend relu backward failed",
};

gd_status gd_relu(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    return gd_unary_apply_impl(ctx, x, out, &GD_RELU_UNARY_SPEC);
}

gd_status gd_relu_backward(gd_context *ctx,
                           const gd_tensor *x,
                           const gd_tensor *grad_out,
                           gd_tensor *grad_x)
{
    return gd_unary_backward_impl(ctx, x, grad_out, grad_x, &GD_RELU_UNARY_SPEC);
}
