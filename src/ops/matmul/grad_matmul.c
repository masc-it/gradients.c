#include "../grad_impl.h"


static gd_status matmul_pair_backward(_gd_bwd_ctx *b,
                                      gd_tensor *go,
                                      int a_id,
                                      int b_id,
                                      bool trans_a,
                                      bool trans_b,
                                      gd_compute_policy compute)
{
    gd_status status = GD_OK;
    gd_tensor *a = NULL;
    gd_tensor *bb = NULL;
    gd_tensor *da = NULL;
    gd_tensor *db = NULL;
    gd_matmul_desc desc = {false, false, compute};

    status = _gd_bwd_fwd(b, a_id, &a);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_fwd(b, b_id, &bb);
    if (status != GD_OK) {
        return status;
    }

    if (trans_a) {
        desc.trans_a = trans_b;
        desc.trans_b = true;
        status = gd_matmul_ex(_gd_bwd_context(b), &desc, bb, go, &da);
    } else {
        desc.trans_a = false;
        desc.trans_b = !trans_b;
        status = gd_matmul_ex(_gd_bwd_context(b), &desc, go, bb, &da);
    }
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate_broadcast(b, a_id, da);
    gd_tensor_release(da);
    if (status != GD_OK) {
        return status;
    }

    if (trans_b) {
        desc.trans_a = true;
        desc.trans_b = trans_a;
        status = gd_matmul_ex(_gd_bwd_context(b), &desc, go, a, &db);
    } else {
        desc.trans_a = !trans_a;
        desc.trans_b = false;
        status = gd_matmul_ex(_gd_bwd_context(b), &desc, a, go, &db);
    }
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate_broadcast(b, b_id, db);
    gd_tensor_release(db);
    return status;
}

static gd_status matmul_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);

    if (go == NULL) {
        return GD_OK;
    }
    return matmul_pair_backward(b, go, node->inputs[0], node->inputs[1],
                                node->attrs.trans_a, node->attrs.trans_b,
                                node->attrs.compute);
}

const _gd_bwd_rule _gd_bwd_rule_matmul = {
    .op = _GD_OP_MATMUL,
    .fn = matmul_backward,
    .unsupported_reason = NULL,
};
