#include "permute_impl.h"

#include "../autograd_impl.h"

#include <stdint.h>

static gd_status gd_permute_autograd_backward(gd_bwd_ctx *bwd,
                                              const gd_tape_node *node)
{
    const gd_permute_attrs *attrs;
    const gd_tensor *x;
    const gd_tensor *out;
    gd_tensor grad_out;
    gd_tensor dx;
    gd_status st;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    attrs = (const gd_permute_attrs *)gd_tape_attrs(bwd->tape, node, (uint32_t)sizeof(*attrs));
    x = gd_tape_input(bwd->tape, node, 0U);
    out = gd_tape_output(bwd->tape, node, 0U);
    if (attrs == NULL || x == NULL || out == NULL || node->n_inputs != 1U ||
        node->n_outputs != 1U || attrs->n_axes != x->rank || attrs->n_axes != out->rank) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    st = gd_permute_backward(bwd->ctx, x, &grad_out, attrs->axes, attrs->n_axes, &dx);
    if (st != GD_OK) {
        return st;
    }
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_permute = {
    .kind = GD_OP_PERMUTE,
    .name = "permute",
    .backward = gd_permute_autograd_backward,
};
