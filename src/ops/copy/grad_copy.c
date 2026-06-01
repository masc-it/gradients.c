#include "../grad_impl.h"

static gd_status copy_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *din = NULL;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_emit(b, _GD_OP_COPY, &go, 1, NULL,
                          _gd_bwd_value_desc(b, node->inputs[0]), &din);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], din);
    gd_tensor_release(din);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_copy = {
    .op = _GD_OP_COPY,
    .fn = copy_backward,
    .unsupported_reason = NULL,
};
