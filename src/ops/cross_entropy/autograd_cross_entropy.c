#include "../autograd_impl.h"
#include "cross_entropy_impl.h"

#include <gradients/ops.h>

static gd_status gd_cross_entropy_autograd_backward(gd_bwd_ctx *bwd,
                                                    const gd_tape_node *node)
{
    const gd_tensor *logits = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *targets = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_tensor *row_max;
    const gd_tensor *row_inv_sum;
    gd_tensor grad_out;
    gd_tensor dlogits;
    if (logits == NULL || targets == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!logits->requires_grad) {
        return GD_OK;
    }
    row_max = node->n_saved == 2U ? gd_tape_saved(bwd->tape, node, 0U) : NULL;
    row_inv_sum = node->n_saved == 2U ? gd_tape_saved(bwd->tape, node, 1U) : NULL;
    if (row_max != NULL && row_inv_sum != NULL) {
        GD_TRY(gd_cross_entropy_backward_with_stats(bwd->ctx,
                                                    logits,
                                                    targets,
                                                    row_max,
                                                    row_inv_sum,
                                                    &grad_out,
                                                    &dlogits));
    } else {
        GD_TRY(gd_cross_entropy_backward(bwd->ctx,
                                         logits,
                                         targets,
                                         &grad_out,
                                         &dlogits,
                                         NULL));
    }
    return gd_autograd_accumulate(bwd, logits->id, &dlogits);
}

const gd_autograd_rule gd_bwd_rule_cross_entropy = {
    .kind = GD_OP_CROSS_ENTROPY,
    .name = "cross_entropy",
    .backward = gd_cross_entropy_autograd_backward,
};
