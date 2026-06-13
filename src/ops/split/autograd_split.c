#include "split_impl.h"

#include "../autograd_impl.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

static gd_status gd_split_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    const gd_split_attrs *attrs;
    const gd_tensor *x;
    gd_tensor dx;
    uint16_t i;
    uint32_t present_grads = 0U;
    int64_t axis_offset;
    int64_t full_axis;
    bool any_missing = false;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    attrs = (const gd_split_attrs *)gd_tape_attrs(bwd->tape, node, (uint32_t)sizeof(*attrs));
    x = gd_tape_input(bwd->tape, node, 0U);
    if (attrs == NULL || x == NULL || attrs->axis < 0 || (uint32_t)attrs->axis >= x->rank ||
        attrs->n_outputs != (uint32_t)node->n_outputs || node->n_inputs != 1U) {
        return GD_ERR_INTERNAL;
    }
    if (!x->requires_grad) {
        return GD_OK;
    }
    full_axis = x->shape[(uint32_t)attrs->axis];
    axis_offset = 0;
    for (i = 0U; i < node->n_outputs; ++i) {
        const gd_tensor *out = gd_tape_output(bwd->tape, node, i);
        gd_tensor grad_out;
        uint32_t d;
        if (out == NULL || out->rank != x->rank || out->dtype != x->dtype ||
            (uint32_t)attrs->axis >= out->rank || out->shape[(uint32_t)attrs->axis] <= 0) {
            return GD_ERR_INTERNAL;
        }
        for (d = 0U; d < x->rank; ++d) {
            if (d != (uint32_t)attrs->axis && out->shape[d] != x->shape[d]) {
                return GD_ERR_INTERNAL;
            }
        }
        if (gd_autograd_get_grad(bwd, out->id, &grad_out)) {
            present_grads += 1U;
        } else {
            any_missing = true;
        }
        if (axis_offset > INT64_MAX - out->shape[(uint32_t)attrs->axis]) {
            return GD_ERR_INTERNAL;
        }
        axis_offset += out->shape[(uint32_t)attrs->axis];
    }
    if (axis_offset != full_axis) {
        return GD_ERR_INTERNAL;
    }
    if (present_grads == 0U) {
        return GD_OK;
    }
    if (any_missing) {
        gd_status st = gd_tensor_zeros(bwd->ctx,
                                       GD_ARENA_SCRATCH,
                                       x->dtype,
                                       gd_shape_make(x->rank, x->shape),
                                       256U,
                                       &dx);
        if (st != GD_OK) {
            return st;
        }
    } else {
        gd_status st = gd_tensor_empty(bwd->ctx,
                                       GD_ARENA_SCRATCH,
                                       x->dtype,
                                       gd_shape_make(x->rank, x->shape),
                                       256U,
                                       &dx);
        if (st != GD_OK) {
            return st;
        }
    }
    dx.requires_grad = false;
    dx.is_leaf = false;
    axis_offset = 0;
    for (i = 0U; i < node->n_outputs; ++i) {
        const gd_tensor *out = gd_tape_output(bwd->tape, node, i);
        gd_tensor grad_out;
        if (out == NULL) {
            return GD_ERR_INTERNAL;
        }
        if (gd_autograd_get_grad(bwd, out->id, &grad_out)) {
            gd_status st = gd_split_backward_output_to_full_impl(bwd->ctx,
                                                                 &grad_out,
                                                                 &dx,
                                                                 (uint32_t)attrs->axis,
                                                                 axis_offset,
                                                                 full_axis);
            if (st != GD_OK) {
                return st;
            }
        }
        axis_offset += out->shape[(uint32_t)attrs->axis];
    }
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

const gd_autograd_rule gd_bwd_rule_split = {
    .kind = GD_OP_SPLIT,
    .name = "split",
    .backward = gd_split_autograd_backward,
};
