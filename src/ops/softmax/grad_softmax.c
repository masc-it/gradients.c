#include "../grad_impl.h"

static gd_status softmax_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *y = NULL;
    gd_tensor *dx = NULL;
    gd_tensor *inputs[2];

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->outputs[0], &y);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = y;
    inputs[1] = go;
    status = _gd_bwd_emit(b, _GD_OP_SOFTMAX_BWD, inputs, 2, &node->attrs,
                          _gd_bwd_value_desc(b, node->outputs[0]), &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_softmax = {
    .op = _GD_OP_SOFTMAX,
    .fn = softmax_backward,
    .unsupported_reason = NULL,
};
