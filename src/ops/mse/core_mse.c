#include <gradients/ops.h>

#include "../_shared/loss/pairwise_loss_core.h"
#include "metal_mse_types.h"

static gd_status gd_mse_backend_forward(gd_backend *backend,
                                        const gd_backend_tensor_view *x,
                                        const gd_backend_tensor_view *y,
                                        const gd_backend_tensor_view *out,
                                        uint64_t chunk_size,
                                        float scale,
                                        const void *attrs)
{
    (void)attrs;
    return gd_backend_mse_forward(backend, x, y, out, chunk_size, scale);
}

static gd_status gd_mse_backend_backward(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_x,
                                         const gd_backend_tensor_view *grad_y,
                                         float scale,
                                         const void *attrs)
{
    (void)attrs;
    return gd_backend_mse_backward(backend, x, y, grad_out, grad_x, grad_y, scale);
}

static const gd_pairwise_loss_spec GD_MSE_LOSS_SPEC = {
    .name = "mse",
    .op = GD_OP_MSE,
    .chunk_size = (size_t)GD_METAL_MSE_CHUNK_SIZE,
    .backward_scale = 2.0f,
    .validate_attrs = NULL,
    .backend_forward = gd_mse_backend_forward,
    .backend_backward = gd_mse_backend_backward,
    .unsupported_dtype_message = "mse supports f16/f32 tensors only",
    .dtype_mismatch_message = "mse input dtypes must match",
    .shape_mismatch_message = "mse input shapes must match",
    .contiguous_message = "mse requires contiguous inputs",
    .invalid_shape_message = "mse invalid input shape",
    .element_count_message = "mse element count unsupported",
    .grad_out_message = "mse backward requires scalar f32 grad_out",
    .forward_view_message = "mse forward invalid tensor view",
    .reduce_view_message = "mse reduce invalid tensor view",
    .backward_view_message = "mse backward invalid tensor view",
    .forward_failed_message = "backend mse forward failed",
    .reduce_failed_message = "backend mse reduce failed",
    .backward_failed_message = "backend mse backward failed",
    .partial_overflow_message = "mse partial count overflow",
};

gd_status gd_mse(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    return gd_pairwise_loss_forward_mean(ctx, x, y, out, &GD_MSE_LOSS_SPEC, NULL);
}

gd_status gd_mse_backward(gd_context *ctx,
                          const gd_tensor *x,
                          const gd_tensor *y,
                          const gd_tensor *grad_out,
                          gd_tensor *grad_x,
                          gd_tensor *grad_y)
{
    return gd_pairwise_loss_backward_mean(ctx,
                                          x,
                                          y,
                                          grad_out,
                                          grad_x,
                                          grad_y,
                                          &GD_MSE_LOSS_SPEC,
                                          NULL);
}
