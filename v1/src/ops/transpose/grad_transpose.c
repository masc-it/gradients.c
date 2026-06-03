#include "../grad_impl.h"

static gd_status transpose_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *dx = NULL;
    _gd_op_attrs attrs = {0};
    int i = 0;

    if (go == NULL) {
        return GD_OK;
    }
    attrs.perm_ndim = node->attrs.perm_ndim;
    for (i = 0; i < node->attrs.perm_ndim; ++i) {
        attrs.perm[node->attrs.perm[i]] = i;
    }
    status = _gd_bwd_emit(b, _GD_OP_TRANSPOSE, &go, 1, &attrs,
                          _gd_bwd_value_desc(b, node->inputs[0]), &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_transpose = {
    .op = _GD_OP_TRANSPOSE,
    .fn = transpose_backward,
    .unsupported_reason = NULL,
};
