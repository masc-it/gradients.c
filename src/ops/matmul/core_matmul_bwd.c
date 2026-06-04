#include <gradients/ops.h>

#include "../../core/memory_internal.h"

#include <string.h>

gd_status gd_matmul_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w)
{
    (void)x;
    (void)w;
    (void)grad_out;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_w != NULL) {
        memset(grad_w, 0, sizeof(*grad_w));
    }
    return gd_context_set_error(ctx, GD_ERR_NOT_IMPLEMENTED, "matmul backward not implemented");
}
