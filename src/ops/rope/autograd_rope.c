#include "rope_impl.h"

#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_rope_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *pos_ids = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_rope_attrs *attrs;
    gd_tensor grad_out;
    gd_tensor dx;
    if (x == NULL || pos_ids == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    attrs = (const gd_rope_attrs *)gd_tape_attrs(bwd->tape,
                                                 node,
                                                 (uint32_t)sizeof(gd_rope_attrs));
    if (attrs == NULL) {
        return GD_ERR_INTERNAL;
    }
    GD_TRY(gd_rope_backward_from_attrs(bwd->ctx, x, pos_ids, &grad_out, attrs, &dx));
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_rope = {
    .kind = GD_OP_ROPE,
    .name = "rope",
    .backward = gd_rope_autograd_backward,
};
