#include "concat_impl.h"

#include "../autograd_impl.h"

#include <stddef.h>
#include <stdint.h>

static gd_status gd_concat_autograd_backward(gd_bwd_ctx *bwd,
                                             const gd_tape_node *node)
{
    const gd_concat_attrs *attrs;
    const gd_tensor *out;
    gd_tensor grad_out;
    uint16_t i;
    int64_t axis_offset = 0;
    int64_t full_axis;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    attrs = (const gd_concat_attrs *)gd_tape_attrs(bwd->tape, node, (uint32_t)sizeof(*attrs));
    out = gd_tape_output(bwd->tape, node, 0U);
    if (attrs == NULL || out == NULL || attrs->axis < 0 || (uint32_t)attrs->axis >= out->rank ||
        attrs->n_inputs != (uint32_t)node->n_inputs) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    full_axis = out->shape[(uint32_t)attrs->axis];
    for (i = 0U; i < node->n_inputs; ++i) {
        const gd_tensor *input = gd_tape_input(bwd->tape, node, i);
        if (input == NULL || (uint32_t)attrs->axis >= input->rank) {
            return GD_ERR_INTERNAL;
        }
        if (input->requires_grad) {
            gd_tensor dx;
            gd_status st = gd_concat_backward_input_impl(bwd->ctx,
                                                         &grad_out,
                                                         input,
                                                         (uint32_t)attrs->axis,
                                                         axis_offset,
                                                         full_axis,
                                                         &dx);
            if (st != GD_OK) {
                return st;
            }
            st = gd_autograd_accumulate(bwd, input->id, &dx);
            if (st != GD_OK) {
                return st;
            }
        }
        axis_offset += input->shape[(uint32_t)attrs->axis];
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_concat = {
    .kind = GD_OP_CONCAT,
    .name = "concat",
    .backward = gd_concat_autograd_backward,
};
