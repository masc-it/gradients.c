#include "../grad_impl.h"

static gd_status add_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_accumulate_broadcast(b, node->inputs[0], go);
    if (status != GD_OK) {
        return status;
    }
    return _gd_bwd_accumulate_broadcast(b, node->inputs[1], go);
}

const _gd_bwd_rule _gd_bwd_rule_add = {
    .op = _GD_OP_ADD,
    .fn = add_backward,
    .unsupported_reason = NULL,
};
