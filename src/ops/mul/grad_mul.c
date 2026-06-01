#include "../grad_impl.h"

static gd_status mul_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *a = NULL;
    gd_tensor *bb = NULL;
    gd_tensor *da = NULL;
    gd_tensor *db = NULL;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &a);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &bb);
    if (status != GD_OK) {
        return status;
    }
    status = gd_mul(_gd_bwd_context(b), go, bb, &da);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate_broadcast(b, node->inputs[0], da);
    gd_tensor_release(da);
    if (status != GD_OK) {
        return status;
    }
    status = gd_mul(_gd_bwd_context(b), go, a, &db);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate_broadcast(b, node->inputs[1], db);
    gd_tensor_release(db);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_mul = {
    .op = _GD_OP_MUL,
    .fn = mul_backward,
    .unsupported_reason = NULL,
};
