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

static gd_status embedding_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *ids = NULL;
    gd_tensor *grad_out = go;
    gd_tensor *casted_go = NULL;
    gd_tensor *dweight = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc out_desc;

    if (go == NULL) {
        return GD_OK;
    }
    status = _gd_bwd_fwd(b, node->inputs[1], &ids);
    if (status != GD_OK) {
        return status;
    }
    out_desc = *_gd_bwd_value_desc(b, node->inputs[0]);
    if (value_needs_f32_grad(b, node->inputs[0])) {
        out_desc.dtype = GD_DTYPE_F32;
        if (gd_tensor_dtype(go) != GD_DTYPE_F32) {
            status = gd_cast(_gd_bwd_context(b), go, GD_DTYPE_F32, &casted_go);
            if (status != GD_OK) {
                return status;
            }
            grad_out = casted_go;
        }
    }
    inputs[0] = grad_out;
    inputs[1] = ids;
    status = _gd_bwd_emit(b, _GD_OP_EMBEDDING_BWD, inputs, 2, &node->attrs,
                          &out_desc, &dweight);
    gd_tensor_release(casted_go);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dweight);
    gd_tensor_release(dweight);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_embedding = {
    .op = _GD_OP_EMBEDDING,
    .fn = embedding_backward,
    .unsupported_reason = NULL,
};
