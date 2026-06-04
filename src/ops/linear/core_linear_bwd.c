#include <gradients/ops.h>

#include "../../core/memory_internal.h"

#include <string.h>

gd_status gd_linear_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *bias,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w,
                             gd_tensor *grad_bias)
{
    (void)x;
    (void)w;
    (void)bias;
    (void)grad_out;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    if (grad_bias != NULL) {
        memset(grad_bias, 0, sizeof(*grad_bias));
    }
    return gd_context_set_error(ctx, GD_ERR_NOT_IMPLEMENTED, "linear backward not implemented");
}
