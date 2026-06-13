#include "../_shared/unary/unary_core.h"
#include "sigmoid_impl.h"

#include <gradients/ops.h>

static gd_status gd_sigmoid_autograd_backward(gd_bwd_ctx *bwd,
                                              const gd_tape_node *node)
{
    return gd_unary_autograd_backward_from_output(bwd, node, gd_sigmoid_backward_from_output);
}

const gd_autograd_rule gd_bwd_rule_sigmoid = {
    .kind = GD_OP_SIGMOID,
    .name = "sigmoid",
    .backward = gd_sigmoid_autograd_backward,
};
