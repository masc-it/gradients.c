#include "../grad_impl.h"

static bool value_needs_f32_grad(_gd_bwd_ctx *b, int value_id)
{
    gd_tensor *leaf = NULL;

    if (value_id < 0 || value_id >= b->graph->n_values) {
        return false;
    }
    leaf = b->graph->values[value_id].external;
    return leaf != NULL && gd_tensor_requires_grad(leaf) &&
           b->graph->values[value_id].desc.dtype == GD_DTYPE_F16;
}

static gd_status matmul_grad_product(_gd_bwd_ctx *b,
                                     int target_id,
                                     const gd_matmul_desc *desc,
                                     gd_tensor *lhs,
                                     gd_tensor *rhs,
                                     gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_matmul_desc local = *desc;
    gd_tensor *l = lhs;
    gd_tensor *r = rhs;
    gd_tensor *lc = NULL;
    gd_tensor *rc = NULL;

    if (value_needs_f32_grad(b, target_id)) {
        local.compute.compute_dtype = GD_DTYPE_F32;
        local.compute.accum_dtype = GD_DTYPE_F32;
        if (gd_tensor_dtype(lhs) != GD_DTYPE_F32) {
            status = gd_cast(_gd_bwd_context(b), lhs, GD_DTYPE_F32, &lc);
            if (status != GD_OK) { return status; }
            l = lc;
        }
        if (gd_tensor_dtype(rhs) != GD_DTYPE_F32) {
            status = gd_cast(_gd_bwd_context(b), rhs, GD_DTYPE_F32, &rc);
            if (status != GD_OK) {
                gd_tensor_release(lc);
                return status;
            }
            r = rc;
        }
    }
    status = gd_matmul_ex(_gd_bwd_context(b), &local, l, r, out);
    gd_tensor_release(lc);
    gd_tensor_release(rc);
    return status;
}

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
        status = matmul_grad_product(b, a_id, &desc, bb, go, &da);
    } else {
        desc.trans_a = false;
        desc.trans_b = !trans_b;
        status = matmul_grad_product(b, a_id, &desc, go, bb, &da);
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
        status = matmul_grad_product(b, b_id, &desc, go, a, &db);
    } else {
        desc.trans_a = !trans_a;
        desc.trans_b = false;
        status = matmul_grad_product(b, b_id, &desc, a, go, &db);
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
