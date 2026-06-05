#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_mse_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *y = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dy;
    bool need_x;
    bool need_y;
    if (x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x = x->requires_grad;
    need_y = y->requires_grad;
    if (!need_x && !need_y) {
        return GD_OK;
    }
    GD_TRY(gd_mse_backward(bwd->ctx,
                           x,
                           y,
                           &grad_out,
                           need_x ? &dx : NULL,
                           need_y ? &dy : NULL));
    if (need_x) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
    }
    if (need_y) {
        GD_TRY(gd_autograd_accumulate(bwd, y->id, &dy));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_mse = {
    .kind = GD_OP_MSE,
    .name = "mse",
    .backward = gd_mse_autograd_backward,
};
