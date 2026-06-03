#include "../grad_impl.h"

static gd_status rope_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *freqs = NULL;
    gd_tensor *dx = NULL;
    gd_tensor *inputs[2];

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &freqs);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = go;
    inputs[1] = freqs;
    status = _gd_bwd_emit(b, _GD_OP_ROPE_BWD, inputs, 2, &node->attrs,
                          _gd_bwd_value_desc(b, node->inputs[0]), &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_rope = {
    .op = _GD_OP_ROPE,
    .fn = rope_backward,
    .unsupported_reason = NULL,
};
