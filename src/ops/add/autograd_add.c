#include "../autograd_impl.h"

#include <gradients/ops.h>

static bool gd_add_same_shape(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t dim;
    if (a == NULL || b == NULL || a->dtype != b->dtype || a->rank != b->rank) {
        return false;
    }
    for (dim = 0U; dim < a->rank; ++dim) {
        if (a->shape[dim] != b->shape[dim]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_add_autograd_backward(gd_bwd_ctx *bwd,
                                          const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *y = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dy;
    bool need_x;
    bool need_y;
    bool reduce_x;
    bool reduce_y;
    if (x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x = x->requires_grad;
    need_y = y->requires_grad;
    if (!need_x && !need_y) {
        return GD_OK;
    }
    reduce_x = need_x && !gd_add_same_shape(x, &grad_out);
    reduce_y = need_y && !gd_add_same_shape(y, &grad_out);
    if (reduce_x || reduce_y) {
        GD_TRY(gd_add_backward(bwd->ctx,
                               x,
                               y,
                               &grad_out,
                               reduce_x ? &dx : NULL,
                               reduce_y ? &dy : NULL));
    }
    if (need_x) {
        if (reduce_x || !gd_autograd_steal_grad_if_absent(bwd, out->id, x->id, NULL)) {
            GD_TRY(gd_autograd_accumulate(bwd, x->id, reduce_x ? &dx : &grad_out));
        }
    }
    if (need_y) {
        GD_TRY(gd_autograd_accumulate(bwd, y->id, reduce_y ? &dy : &grad_out));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_add = {
    .kind = GD_OP_ADD,
    .name = "add",
    .backward = gd_add_autograd_backward,
};
