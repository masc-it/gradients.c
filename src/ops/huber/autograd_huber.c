#include "../_shared/loss/pairwise_loss_core.h"

#include <gradients/ops.h>

static gd_status gd_huber_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    return gd_pairwise_loss_autograd_backward(bwd, node, gd_huber_backward);
}

const gd_autograd_rule gd_bwd_rule_huber = {
    .kind = GD_OP_HUBER,
    .name = "huber",
    .backward = gd_huber_autograd_backward,
};
