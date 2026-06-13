#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_matmul_autograd_backward(gd_bwd_ctx *bwd,
                                             const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *w = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dw;
    bool need_x;
    bool need_w;
    if (x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x = x->requires_grad;
    need_w = w->requires_grad;
    if (!need_x && !need_w) {
        return GD_OK;
    }
    GD_TRY(gd_matmul_backward(bwd->ctx,
                              x,
                              w,
                              &grad_out,
                              need_x ? &dx : NULL,
                              need_w ? &dw : NULL));
    if (need_x) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
    }
    if (need_w) {
        GD_TRY(gd_autograd_accumulate(bwd, w->id, &dw));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_matmul = {
    .kind = GD_OP_MATMUL,
    .name = "matmul",
    .backward = gd_matmul_autograd_backward,
};
