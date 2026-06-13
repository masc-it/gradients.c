#include "../_shared/loss/pairwise_loss_core.h"

#include <gradients/ops.h>

static gd_status gd_mse_autograd_backward(gd_bwd_ctx *bwd,
                                           const gd_tape_node *node)
{
    return gd_pairwise_loss_autograd_backward(bwd, node, gd_mse_backward);
}

const gd_autograd_rule gd_bwd_rule_mse = {
    .kind = GD_OP_MSE,
    .name = "mse",
    .backward = gd_mse_autograd_backward,
};
