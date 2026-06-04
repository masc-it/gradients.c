#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_relu_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;
    if (x == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    GD_TRY(gd_relu_backward(bwd->ctx, x, &grad_out, &dx));
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_relu = {
    .kind = GD_OP_RELU,
    .name = "relu",
    .backward = gd_relu_autograd_backward,
};
