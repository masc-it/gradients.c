#include "../grad_impl.h"

static gd_status rms_norm_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *x = NULL;
    gd_tensor *weight = NULL;
    gd_tensor *dx = NULL;
    gd_tensor *dw = NULL;
    gd_tensor *dx_inputs[3];
    gd_tensor *dw_inputs[2];
    _gd_op_attrs attrs = {0};
    gd_tensor_desc x_desc;
    gd_tensor_desc w_desc;

    if (go == NULL) {
        return GD_OK;
    }
    x_desc = *_gd_bwd_value_desc(b, node->inputs[0]);
    w_desc = *_gd_bwd_value_desc(b, node->inputs[1]);
    attrs.eps = node->attrs.eps;
    status = _gd_bwd_fwd(b, node->inputs[0], &x);
    if (status == GD_OK) {
        status = _gd_bwd_fwd(b, node->inputs[1], &weight);
    }
    if (status != GD_OK) {
        return status;
    }
    dx_inputs[0] = x;
    dx_inputs[1] = weight;
    dx_inputs[2] = go;
    status = _gd_bwd_emit(b, _GD_OP_RMS_NORM_BWD, dx_inputs, 3, &attrs,
                          &x_desc, &dx);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    }
    gd_tensor_release(dx);
    if (status != GD_OK) {
        return status;
    }
    dw_inputs[0] = x;
    dw_inputs[1] = go;
    status = _gd_bwd_emit(b, _GD_OP_RMS_NORM_WBWD, dw_inputs, 2, &attrs,
                          &w_desc, &dw);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[1], dw);
    }
    gd_tensor_release(dw);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_rms_norm = {
    .op = _GD_OP_RMS_NORM,
    .fn = rms_norm_backward,
    .unsupported_reason = NULL,
};
