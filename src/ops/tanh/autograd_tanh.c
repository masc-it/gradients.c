#include "../_shared/unary/unary_core.h"
#include "tanh_impl.h"

#include <gradients/ops.h>

static gd_status gd_tanh_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    return gd_unary_autograd_backward_from_output(bwd, node, gd_tanh_backward_from_output);
}

const gd_autograd_rule gd_bwd_rule_tanh = {
    .kind = GD_OP_TANH,
    .name = "tanh",
    .backward = gd_tanh_autograd_backward,
};
