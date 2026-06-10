#include <gradients/ops.h>

#include "../_shared/unary/unary_core.h"
#include "tanh_impl.h"

static bool gd_tanh_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static const gd_unary_op_spec GD_TANH_UNARY_SPEC = {
    .name = "tanh",
    .op = GD_OP_TANH,
    .dtype_supported = gd_tanh_dtype_supported,
    .backend_forward = gd_backend_tanh,
    .backend_backward = gd_backend_tanh_backward,
    .unsupported_dtype_message = "tanh supports f16 and f32 tensors only",
    .contiguous_input_message = "tanh requires contiguous input",
    .grad_shape_message = "tanh backward gradient shape mismatch",
    .grad_contiguous_message = "tanh backward requires contiguous tensors",
    .forward_view_message = "tanh invalid tensor view",
    .backward_view_message = "tanh backward invalid tensor view",
    .backend_forward_failed_message = "backend tanh failed",
    .backend_backward_failed_message = "backend tanh backward failed",
};

gd_status gd_tanh(gd_context *ctx, const gd_tensor *x, gd_tensor *out)
{
    return gd_unary_apply_impl(ctx, x, out, &GD_TANH_UNARY_SPEC);
}

gd_status gd_tanh_backward(gd_context *ctx,
                           const gd_tensor *x,
                           const gd_tensor *grad_out,
                           gd_tensor *grad_x)
{
    return gd_unary_backward_impl(ctx, x, grad_out, grad_x, &GD_TANH_UNARY_SPEC);
}

gd_status gd_tanh_backward_from_output(gd_context *ctx,
                                       const gd_tensor *tanh_out,
                                       const gd_tensor *grad_out,
                                       gd_tensor *grad_x)
{
    return gd_unary_backward_impl_with_backend(ctx,
                                               tanh_out,
                                               grad_out,
                                               grad_x,
                                               &GD_TANH_UNARY_SPEC,
                                               gd_backend_tanh_backward_from_output,
                                               "tanh saved backward gradient shape mismatch",
                                               "tanh saved backward invalid tensor view",
                                               "backend tanh saved backward failed");
}
