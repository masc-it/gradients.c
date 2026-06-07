#include "lm_cross_entropy_impl.h"

#include "../autograd_impl.h"

static gd_status gd_lm_cross_entropy_autograd_backward(gd_bwd_ctx *bwd,
                                                       const gd_tape_node *node)
{
    const gd_tensor *hidden = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *weight = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *targets = gd_tape_input(bwd->tape, node, 2U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_tensor *logits = gd_tape_saved(bwd->tape, node, 0U);
    const gd_tensor *row_max = gd_tape_saved(bwd->tape, node, 1U);
    const gd_tensor *row_inv_sum = gd_tape_saved(bwd->tape, node, 2U);
    gd_tensor grad_out;
    gd_tensor dhidden;
    gd_tensor dweight;
    bool need_hidden;
    bool need_weight;
    if (hidden == NULL || weight == NULL || targets == NULL || out == NULL || logits == NULL ||
        row_max == NULL || row_inv_sum == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_hidden = hidden->requires_grad;
    need_weight = weight->requires_grad;
    if (!need_hidden && !need_weight) {
        return GD_OK;
    }
    GD_TRY(gd_lm_cross_entropy_backward_with_stats(bwd->ctx,
                                                   hidden,
                                                   weight,
                                                   targets,
                                                   logits,
                                                   row_max,
                                                   row_inv_sum,
                                                   &grad_out,
                                                   need_hidden ? &dhidden : NULL,
                                                   need_weight ? &dweight : NULL));
    if (need_hidden) {
        GD_TRY(gd_autograd_accumulate(bwd, hidden->id, &dhidden));
    }
    if (need_weight) {
        GD_TRY(gd_autograd_accumulate(bwd, weight->id, &dweight));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_lm_cross_entropy = {
    .kind = GD_OP_LM_CROSS_ENTROPY,
    .name = "lm_cross_entropy",
    .backward = gd_lm_cross_entropy_autograd_backward,
};
