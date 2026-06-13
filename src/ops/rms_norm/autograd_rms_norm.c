#include "rms_norm_impl.h"

#include "../autograd_impl.h"

static gd_status gd_rms_norm_autograd_backward(gd_bwd_ctx *bwd,
                                                const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *weight = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_tensor *inv_rms;
    const gd_rms_norm_attrs *attrs;
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dw;
    if (x == NULL || weight == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad && !weight->requires_grad) {
        return GD_OK;
    }
    attrs = (const gd_rms_norm_attrs *)gd_tape_attrs(bwd->tape,
                                                     node,
                                                     (uint32_t)sizeof(gd_rms_norm_attrs));
    if (attrs == NULL) {
        return GD_ERR_INTERNAL;
    }
    inv_rms = node->n_saved == 1U ? gd_tape_saved(bwd->tape, node, 0U) : NULL;
    if (inv_rms != NULL) {
        GD_TRY(gd_rms_norm_backward_with_stats(bwd->ctx,
                                               x,
                                               weight,
                                               inv_rms,
                                               &grad_out,
                                               x->requires_grad ? &dx : NULL,
                                               weight->requires_grad ? &dw : NULL));
    } else {
        GD_TRY(gd_rms_norm_backward(bwd->ctx,
                                    x,
                                    weight,
                                    &grad_out,
                                    attrs->eps,
                                    x->requires_grad ? &dx : NULL,
                                    weight->requires_grad ? &dw : NULL));
    }
    if (x->requires_grad) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
    }
    if (weight->requires_grad) {
        GD_TRY(gd_autograd_accumulate(bwd, weight->id, &dw));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_rms_norm = {
    .kind = GD_OP_RMS_NORM,
    .name = "rms_norm",
    .backward = gd_rms_norm_autograd_backward,
};
