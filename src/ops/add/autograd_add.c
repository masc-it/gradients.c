#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_add_autograd_backward(gd_bwd_ctx *bwd,
                                          const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *y = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    if (x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (x->requires_grad) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &grad_out));
    }
    if (y->requires_grad) {
        GD_TRY(gd_autograd_accumulate(bwd, y->id, &grad_out));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_add = {
    .kind = GD_OP_ADD,
    .name = "add",
    .backward = gd_add_autograd_backward,
};
