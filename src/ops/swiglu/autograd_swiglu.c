#include "../autograd_impl.h"
#include "swiglu_impl.h"

#include <gradients/ops.h>

static gd_status gd_swiglu_autograd_backward(gd_bwd_ctx *bwd,
                                             const gd_tape_node *node)
{
    const gd_tensor *x1 = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx1;
    gd_tensor dx2;
    bool need_x1;
    bool need_x2;
    if (x1 == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (node->n_inputs == 1U) {
        if (!x1->requires_grad) {
            return GD_OK;
        }
        GD_TRY(gd_swiglu_split_backward(bwd->ctx, x1, &grad_out, &dx1));
        return gd_autograd_accumulate(bwd, x1->id, &dx1);
    }
    if (node->n_inputs != 2U) {
        return GD_ERR_INTERNAL;
    }
    {
        const gd_tensor *x2 = gd_tape_input(bwd->tape, node, 1U);
        if (x2 == NULL) {
            return GD_ERR_INTERNAL;
        }
        need_x1 = x1->requires_grad;
        need_x2 = x2->requires_grad;
        if (!need_x1 && !need_x2) {
            return GD_OK;
        }
        GD_TRY(gd_swiglu_backward(bwd->ctx,
                                  x1,
                                  x2,
                                  &grad_out,
                                  need_x1 ? &dx1 : NULL,
                                  need_x2 ? &dx2 : NULL));
        if (need_x1) {
            GD_TRY(gd_autograd_accumulate(bwd, x1->id, &dx1));
        }
        if (need_x2) {
            GD_TRY(gd_autograd_accumulate(bwd, x2->id, &dx2));
        }
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_swiglu = {
    .kind = GD_OP_SWIGLU,
    .name = "swiglu",
    .backward = gd_swiglu_autograd_backward,
};
