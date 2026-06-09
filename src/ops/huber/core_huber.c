#include <gradients/ops.h>

#include "../_shared/loss/pairwise_loss_core.h"
#include "metal_huber_types.h"

#define GD_HUBER_DEFAULT_DELTA 1.0f

typedef struct gd_huber_loss_attrs {
    float delta;
} gd_huber_loss_attrs;

static gd_status gd_huber_validate_attrs(gd_context *ctx, const void *attrs)
{
    const gd_huber_loss_attrs *huber_attrs = (const gd_huber_loss_attrs *)attrs;
    if (ctx == NULL || huber_attrs == NULL || !(huber_attrs->delta > 0.0f)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "huber delta must be positive");
    }
    return GD_OK;
}

static gd_status gd_huber_backend_forward(gd_backend *backend,
                                          const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *y,
                                          const gd_backend_tensor_view *out,
                                          uint64_t chunk_size,
                                          float scale,
                                          const void *attrs)
{
    const gd_huber_loss_attrs *huber_attrs = (const gd_huber_loss_attrs *)attrs;
    if (huber_attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_backend_huber_forward(backend,
                                    x,
                                    y,
                                    out,
                                    chunk_size,
                                    scale,
                                    huber_attrs->delta);
}

static gd_status gd_huber_backend_backward(gd_backend *backend,
                                           const gd_backend_tensor_view *x,
                                           const gd_backend_tensor_view *y,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_x,
                                           const gd_backend_tensor_view *grad_y,
                                           float scale,
                                           const void *attrs)
{
    const gd_huber_loss_attrs *huber_attrs = (const gd_huber_loss_attrs *)attrs;
    if (huber_attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_backend_huber_backward(backend,
                                     x,
                                     y,
                                     grad_out,
                                     grad_x,
                                     grad_y,
                                     scale,
                                     huber_attrs->delta);
}

static const gd_pairwise_loss_spec GD_HUBER_LOSS_SPEC = {
    .name = "huber",
    .op = GD_OP_HUBER,
    .chunk_size = (size_t)GD_METAL_HUBER_CHUNK_SIZE,
    .backward_scale = 1.0f,
    .validate_attrs = gd_huber_validate_attrs,
    .backend_forward = gd_huber_backend_forward,
    .backend_backward = gd_huber_backend_backward,
    .unsupported_dtype_message = "huber supports f16/f32 tensors only",
    .dtype_mismatch_message = "huber input dtypes must match",
    .shape_mismatch_message = "huber input shapes must match",
    .contiguous_message = "huber requires contiguous inputs",
    .invalid_shape_message = "huber invalid input shape",
    .element_count_message = "huber element count unsupported",
    .grad_out_message = "huber backward requires scalar f32 grad_out",
    .forward_view_message = "huber forward invalid tensor view",
    .reduce_view_message = "huber reduce invalid tensor view",
    .backward_view_message = "huber backward invalid tensor view",
    .forward_failed_message = "backend huber forward failed",
    .reduce_failed_message = "backend huber reduce failed",
    .backward_failed_message = "backend huber backward failed",
    .partial_overflow_message = "huber partial count overflow",
};

gd_status gd_huber(gd_context *ctx,
                   const gd_tensor *x,
                   const gd_tensor *y,
                   gd_tensor *out)
{
    const gd_huber_loss_attrs attrs = {.delta = GD_HUBER_DEFAULT_DELTA};
    return gd_pairwise_loss_forward_mean(ctx, x, y, out, &GD_HUBER_LOSS_SPEC, &attrs);
}

gd_status gd_huber_backward(gd_context *ctx,
                            const gd_tensor *x,
                            const gd_tensor *y,
                            const gd_tensor *grad_out,
                            gd_tensor *grad_x,
                            gd_tensor *grad_y)
{
    const gd_huber_loss_attrs attrs = {.delta = GD_HUBER_DEFAULT_DELTA};
    return gd_pairwise_loss_backward_mean(ctx,
                                          x,
                                          y,
                                          grad_out,
                                          grad_x,
                                          grad_y,
                                          &GD_HUBER_LOSS_SPEC,
                                          &attrs);
}
