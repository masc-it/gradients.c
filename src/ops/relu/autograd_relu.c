#include "../_shared/unary/unary_core.h"

#include <gradients/ops.h>

static gd_status gd_relu_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    return gd_unary_autograd_backward(bwd, node, gd_relu_backward);
}

const gd_autograd_rule gd_bwd_rule_relu = {
    .kind = GD_OP_RELU,
    .name = "relu",
    .backward = gd_relu_autograd_backward,
};
