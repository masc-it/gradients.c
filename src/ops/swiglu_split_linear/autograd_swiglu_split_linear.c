#include "../autograd_impl.h"
#include "swiglu_split_linear_impl.h"

#include <gradients/ops.h>

static gd_status gd_swiglu_split_linear_autograd_backward(gd_bwd_ctx *bwd,
                                                          const gd_tape_node *node)
{
    const gd_tensor *x12 = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *w = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *bias = node->n_inputs == 3U ? gd_tape_input(bwd->tape, node, 2U) : NULL;
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_tensor *activation = node->n_saved == 1U ? gd_tape_saved(bwd->tape, node, 0U) : NULL;
    gd_tensor grad_out;
    gd_tensor dx12;
    gd_tensor dw;
    gd_tensor db;
    bool need_x12;
    bool need_w;
    bool need_bias;
    if (x12 == NULL || w == NULL || out == NULL ||
        (node->n_inputs != 2U && node->n_inputs != 3U) || node->n_saved > 1U) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x12 = x12->requires_grad;
    need_w = w->requires_grad;
    need_bias = bias != NULL && bias->requires_grad;
    if (!need_x12 && !need_w && !need_bias) {
        return GD_OK;
    }
    GD_TRY(gd_swiglu_split_linear_backward_with_activation(bwd->ctx,
                                                           x12,
                                                           activation,
                                                           w,
                                                           bias,
                                                           &grad_out,
                                                           need_x12 ? &dx12 : NULL,
                                                           need_w ? &dw : NULL,
                                                           need_bias ? &db : NULL));
    if (need_x12) {
        GD_TRY(gd_autograd_accumulate(bwd, x12->id, &dx12));
    }
    if (need_w) {
        GD_TRY(gd_autograd_accumulate(bwd, w->id, &dw));
    }
    if (need_bias) {
        GD_TRY(gd_autograd_accumulate(bwd, bias->id, &db));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_swiglu_split_linear = {
    .kind = GD_OP_SWIGLU_SPLIT_LINEAR,
    .name = "swiglu_split_linear",
    .backward = gd_swiglu_split_linear_autograd_backward,
};
