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

static gd_status accumulate_projection_weight_grad(_gd_bwd_ctx *b,
                                                   int weight_id,
                                                   gd_tensor *norm,
                                                   gd_tensor *go)
{
    gd_status status = GD_OK;
    gd_tensor *dw = NULL;
    gd_matmul_desc desc = {0};

    desc.trans_a = true;
    desc.trans_b = false;
    desc.compute = gd_context_compute_policy(_gd_bwd_context(b));
    status = matmul_grad_product(b, weight_id, &desc, norm, go, &dw);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate_broadcast(b, weight_id, dw);
    gd_tensor_release(dw);
    return status;
}

static gd_status add_owned_contrib(gd_context *ctx, gd_tensor **accum, gd_tensor *contrib)
{
    gd_status status = GD_OK;
    gd_tensor *sum = NULL;

    if (contrib == NULL) {
        return GD_OK;
    }
    if (*accum == NULL) {
        *accum = contrib;
        return GD_OK;
    }
    status = gd_add(ctx, *accum, contrib, &sum);
    gd_tensor_release(*accum);
    gd_tensor_release(contrib);
    if (status != GD_OK) {
        *accum = NULL;
        return status;
    }
    *accum = sum;
    return GD_OK;
}

static gd_status accumulate_norm_grad_from_projection(_gd_bwd_ctx *b,
                                                      int norm_id,
                                                      int weight_id,
                                                      gd_tensor *go,
                                                      gd_tensor **dn)
{
    gd_status status = GD_OK;
    gd_tensor *w = NULL;
    gd_tensor *contrib = NULL;
    gd_matmul_desc desc = {0};

    status = _gd_bwd_fwd(b, weight_id, &w);
    if (status != GD_OK) {
        return status;
    }
    desc.trans_a = false;
    desc.trans_b = true;
    desc.compute = gd_context_compute_policy(_gd_bwd_context(b));
    status = matmul_grad_product(b, norm_id, &desc, go, w, &contrib);
    if (status != GD_OK) {
        return status;
    }
    return add_owned_contrib(_gd_bwd_context(b), dn, contrib);
}

static gd_status rms_norm_qkv_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go_norm = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *go_q = _gd_bwd_grad(b, node->outputs[1]);
    gd_tensor *go_k = _gd_bwd_grad(b, node->outputs[2]);
    gd_tensor *go_v = _gd_bwd_grad(b, node->outputs[3]);
    gd_tensor *x = NULL;
    gd_tensor *weight = NULL;
    gd_tensor *norm = NULL;
    gd_tensor *dn = NULL;
    gd_tensor *dx = NULL;
    gd_tensor *dw_norm = NULL;
    gd_tensor *dn_inputs[3];
    gd_tensor *dw_inputs[2];
    _gd_op_attrs attrs = {0};
    gd_tensor_desc x_desc;
    gd_tensor_desc w_desc;

    if (go_norm == NULL && go_q == NULL && go_k == NULL && go_v == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[0], &x);
    if (status == GD_OK) { status = _gd_bwd_fwd(b, node->inputs[1], &weight); }
    if (status == GD_OK) { status = _gd_bwd_fwd(b, node->outputs[0], &norm); }
    if (status != GD_OK) {
        return status;
    }

    if (go_norm != NULL) {
        status = gd_tensor_retain(go_norm);
        if (status != GD_OK) {
            return status;
        }
        dn = go_norm;
    }
    if (go_q != NULL) {
        status = accumulate_projection_weight_grad(b, node->inputs[2], norm, go_q);
        if (status == GD_OK) {
            status = accumulate_norm_grad_from_projection(b, node->outputs[0], node->inputs[2], go_q, &dn);
        }
        if (status != GD_OK) { goto done; }
    }
    if (go_k != NULL) {
        status = accumulate_projection_weight_grad(b, node->inputs[3], norm, go_k);
        if (status == GD_OK) {
            status = accumulate_norm_grad_from_projection(b, node->outputs[0], node->inputs[3], go_k, &dn);
        }
        if (status != GD_OK) { goto done; }
    }
    if (go_v != NULL) {
        status = accumulate_projection_weight_grad(b, node->inputs[4], norm, go_v);
        if (status == GD_OK) {
            status = accumulate_norm_grad_from_projection(b, node->outputs[0], node->inputs[4], go_v, &dn);
        }
        if (status != GD_OK) { goto done; }
    }
    if (dn == NULL) {
        goto done;
    }

    x_desc = *_gd_bwd_value_desc(b, node->inputs[0]);
    w_desc = *_gd_bwd_value_desc(b, node->inputs[1]);
    if (value_needs_f32_grad(b, node->inputs[1])) {
        w_desc.dtype = GD_DTYPE_F32;
    }
    attrs.eps = node->attrs.eps;
    dn_inputs[0] = x;
    dn_inputs[1] = weight;
    dn_inputs[2] = dn;
    status = _gd_bwd_emit(b, _GD_OP_RMS_NORM_BWD, dn_inputs, 3, &attrs, &x_desc, &dx);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    }
    gd_tensor_release(dx);
    if (status != GD_OK) { goto done; }

    dw_inputs[0] = x;
    dw_inputs[1] = dn;
    status = _gd_bwd_emit(b, _GD_OP_RMS_NORM_WBWD, dw_inputs, 2, &attrs, &w_desc, &dw_norm);
    if (status == GD_OK) {
        status = _gd_bwd_accumulate(b, node->inputs[1], dw_norm);
    }
    gd_tensor_release(dw_norm);

done:
    gd_tensor_release(dn);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_rms_norm_qkv = {
    .op = _GD_OP_RMS_NORM_QKV,
    .fn = rms_norm_qkv_backward,
    .unsupported_reason = NULL,
};
