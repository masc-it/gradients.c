#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_linear_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *w = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *bias = node->n_inputs == 3U ? gd_tape_input(bwd->tape, node, 2U) : NULL;
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
    bool need_x;
    bool need_w;
    bool need_b;
    if (x == NULL || w == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x = x->requires_grad;
    need_w = w->requires_grad;
    need_b = bias != NULL && bias->requires_grad;
    if (!need_x && !need_w && !need_b) {
        return GD_OK;
    }
    GD_TRY(gd_linear_backward(bwd->ctx,
                              x,
                              w,
                              bias,
                              &grad_out,
                              need_x ? &dx : NULL,
                              need_w ? &dw : NULL,
                              need_b ? &db : NULL));
    if (need_x) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
    }
    if (need_w) {
        GD_TRY(gd_autograd_accumulate(bwd, w->id, &dw));
    }
    if (need_b) {
        GD_TRY(gd_autograd_accumulate(bwd, bias->id, &db));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_linear = {
    .kind = GD_OP_LINEAR,
    .name = "linear",
    .backward = gd_linear_autograd_backward,
};
