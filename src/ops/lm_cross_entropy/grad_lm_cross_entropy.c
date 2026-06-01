#include "../grad_impl.h"

static gd_status lm_cross_entropy_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *hidden = NULL;
    gd_tensor *weight = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *row_max = NULL;
    gd_tensor *row_sum = NULL;
    gd_tensor *grads[2] = {NULL, NULL};
    gd_tensor *inputs[6];
    gd_tensor_desc out_descs[2];

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &hidden);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &weight);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[2], &targets);
    if (status != GD_OK) {
        return status;
    }
    if (node->n_outputs != 3) {
        return _gd_error(GD_ERR_INTERNAL, "lm_cross_entropy expects stats outputs");
    }
    status = _gd_bwd_fwd(b, node->outputs[1], &row_max);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->outputs[2], &row_sum);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = hidden;
    inputs[1] = weight;
    inputs[2] = targets;
    inputs[3] = go;
    inputs[4] = row_max;
    inputs[5] = row_sum;
    out_descs[0] = *_gd_bwd_value_desc(b, node->inputs[0]);
    out_descs[1] = *_gd_bwd_value_desc(b, node->inputs[1]);
    status = _gd_bwd_emit_multi(b, _GD_OP_LM_CROSS_ENTROPY_BWD, inputs, 6, NULL,
                                out_descs, 2, grads);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], grads[0]);
    gd_tensor_release(grads[0]);
    if (status != GD_OK) {
        gd_tensor_release(grads[1]);
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[1], grads[1]);
    gd_tensor_release(grads[1]);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_lm_cross_entropy = {
    .op = _GD_OP_LM_CROSS_ENTROPY,
    .fn = lm_cross_entropy_backward,
    .unsupported_reason = NULL,
};
