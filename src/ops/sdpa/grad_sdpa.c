#include "../grad_impl.h"

static gd_status sdpa_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *bias = NULL;
    gd_tensor *grads[3] = {NULL, NULL, NULL};
    gd_tensor *inputs[5];
    gd_tensor_desc out_descs[3];
    int n_inputs = 4;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &q);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &k);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[2], &v);
    if (status != GD_OK) {
        return status;
    }
    out_descs[0] = *_gd_bwd_value_desc(b, node->inputs[0]);
    out_descs[1] = *_gd_bwd_value_desc(b, node->inputs[1]);
    out_descs[2] = *_gd_bwd_value_desc(b, node->inputs[2]);
    inputs[0] = go;
    inputs[1] = q;
    inputs[2] = k;
    inputs[3] = v;
    if (node->attrs.has_bias) {
        status = _gd_bwd_fwd(b, node->inputs[3], &bias);
        if (status != GD_OK) {
            return status;
        }
        inputs[4] = bias;
        n_inputs = 5;
    }
    status = _gd_bwd_emit_multi(b, _GD_OP_SDPA_BWD, inputs, n_inputs, &node->attrs,
                                out_descs, 3, grads);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], grads[0]);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[1], grads[1]);
    }
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[2], grads[2]);
    }
    gd_tensor_release(grads[0]);
    gd_tensor_release(grads[1]);
    gd_tensor_release(grads[2]);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_sdpa = {
    .op = _GD_OP_SDPA,
    .fn = sdpa_backward,
    .unsupported_reason = NULL,
};
