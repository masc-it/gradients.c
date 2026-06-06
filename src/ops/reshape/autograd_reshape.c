#include "reshape_impl.h"

#include "../autograd_impl.h"

static gd_status gd_reshape_autograd_backward(gd_bwd_ctx *bwd,
                                              const gd_tape_node *node)
{
    const gd_tensor *x;
    const gd_tensor *out;
    gd_tensor grad_out;
    gd_tensor dx;
    gd_status st;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    x = gd_tape_input(bwd->tape, node, 0U);
    out = gd_tape_output(bwd->tape, node, 0U);
    if (x == NULL || out == NULL || node->n_inputs != 1U || node->n_outputs != 1U) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    st = gd_reshape_backward(bwd->ctx, x, &grad_out, &dx);
    if (st != GD_OK) {
        return st;
    }
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_reshape = {
    .kind = GD_OP_RESHAPE,
    .name = "reshape",
    .backward = gd_reshape_autograd_backward,
};
