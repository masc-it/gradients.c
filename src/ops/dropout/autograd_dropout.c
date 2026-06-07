#include "../autograd_impl.h"
#include "dropout_impl.h"

#include <gradients/ops.h>

static gd_status gd_dropout_autograd_backward(gd_bwd_ctx *bwd,
                                              const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_tensor *mask = gd_tape_saved(bwd->tape, node, 0U);
    const gd_dropout_attrs *attrs =
        (const gd_dropout_attrs *)gd_tape_attrs(bwd->tape,
                                                node,
                                                (uint32_t)sizeof(gd_dropout_attrs));
    gd_tensor grad_out;
    gd_tensor dx;
    if (x == NULL || out == NULL || mask == NULL || attrs == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (node->n_inputs == 2U) {
        const gd_tensor *residual = x;
        x = gd_tape_input(bwd->tape, node, 1U);
        if (x == NULL) {
            return GD_ERR_INTERNAL;
        }
        if (residual->requires_grad) {
            GD_TRY(gd_autograd_accumulate(bwd, residual->id, &grad_out));
        }
        if (x->requires_grad) {
            GD_TRY(gd_dropout_backward_from_mask(bwd->ctx, mask, &grad_out, attrs->scale, &dx));
            GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
        }
        return GD_OK;
    }
    if (node->n_inputs != 1U) {
        return GD_ERR_INTERNAL;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    GD_TRY(gd_dropout_backward_from_mask(bwd->ctx, mask, &grad_out, attrs->scale, &dx));
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_dropout = {
    .kind = GD_OP_DROPOUT,
    .name = "dropout",
    .backward = gd_dropout_autograd_backward,
};
