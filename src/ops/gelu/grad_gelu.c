#include "../grad_impl.h"

static gd_status gelu_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *x = NULL;
    gd_tensor *dx = NULL;
    gd_tensor *inputs[2];
    _gd_op_attrs attrs = {0};

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &x);
    if (status != GD_OK) {
        return status;
    }
    attrs.gelu_tanh = node->attrs.gelu_tanh;
    inputs[0] = x;
    inputs[1] = go;
    status = _gd_bwd_emit(b, _GD_OP_GELU_BWD, inputs, 2, &attrs,
                          _gd_bwd_value_desc(b, node->inputs[0]), &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_gelu = {
    .op = _GD_OP_GELU,
    .fn = gelu_backward,
    .unsupported_reason = NULL,
};
