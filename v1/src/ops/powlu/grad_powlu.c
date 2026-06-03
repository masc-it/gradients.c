#include "../grad_impl.h"

static gd_status powlu_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *x1 = NULL;
    gd_tensor *x2 = NULL;
    gd_tensor *grads[2] = {NULL, NULL};
    gd_tensor *inputs[3];
    gd_tensor_desc out_descs[2];

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &x1);
    if (status == GD_OK) {
        status = _gd_bwd_fwd(b, node->inputs[1], &x2);
    }
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = x1;
    inputs[1] = x2;
    inputs[2] = go;
    out_descs[0] = *_gd_bwd_value_desc(b, node->inputs[0]);
    out_descs[1] = *_gd_bwd_value_desc(b, node->inputs[1]);
    status = _gd_bwd_emit_multi(b, _GD_OP_POWLU_BWD, inputs, 3, &node->attrs,
                                out_descs, 2, grads);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], grads[0]);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[1], grads[1]);
    }
    gd_tensor_release(grads[0]);
    gd_tensor_release(grads[1]);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_powlu = {
    .op = _GD_OP_POWLU,
    .fn = powlu_backward,
    .unsupported_reason = NULL,
};
