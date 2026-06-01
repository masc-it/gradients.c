#include "../grad_impl.h"

static gd_status cross_entropy_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *logits = NULL;
    gd_tensor *target = NULL;
    gd_tensor *dlogits = NULL;
    gd_tensor *inputs[3];
    const gd_tensor_desc *desc = NULL;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &logits);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &target);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = logits;
    inputs[1] = target;
    inputs[2] = go;
    desc = _gd_bwd_value_desc(b, node->inputs[0]);
    status = _gd_bwd_emit(b, _GD_OP_CROSS_ENTROPY_BWD, inputs, 3, &node->attrs,
                          desc, &dlogits);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dlogits);
    gd_tensor_release(dlogits);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_cross_entropy = {
    .op = _GD_OP_CROSS_ENTROPY,
    .fn = cross_entropy_backward,
    .unsupported_reason = NULL,
};
