#include "../autograd_impl.h"

#include "../_shared/reduce/reduce_core.h"

static gd_status gd_reduce_mean_autograd_backward(gd_bwd_ctx *bwd,
                                                  const gd_tape_node *node)
{
    const gd_tensor *x = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_reduce_op_attrs *attrs;
    gd_tensor grad_out;
    gd_tensor dx;
    if (x == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    attrs = (const gd_reduce_op_attrs *)gd_tape_attrs(bwd->tape,
                                                      node,
                                                      (uint32_t)sizeof(gd_reduce_op_attrs));
    if (attrs != NULL && attrs->axis != GD_REDUCE_AXIS_ALL) {
        GD_TRY(gd_reduce_mean_axis_backward(bwd->ctx,
                                            x,
                                            &grad_out,
                                            attrs->axis,
                                            attrs->keepdims != 0U,
                                            &dx));
    } else {
        GD_TRY(gd_reduce_mean_backward(bwd->ctx, x, &grad_out, &dx));
    }
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_reduce_mean = {
    .kind = GD_OP_REDUCE_MEAN,
    .name = "reduce_mean",
    .backward = gd_reduce_mean_autograd_backward,
};
