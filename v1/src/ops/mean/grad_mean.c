#include "../grad_impl.h"

static gd_status mean_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *dx = NULL;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_emit(b, _GD_OP_MEAN_BWD, &go, 1, &node->attrs,
                          _gd_bwd_value_desc(b, node->inputs[0]), &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_mean = {
    .op = _GD_OP_MEAN,
    .fn = mean_backward,
    .unsupported_reason = NULL,
};
