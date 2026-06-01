#include "../grad_impl.h"
#include "../meta_common.h"

static gd_status cast_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    const gd_tensor_desc *in_desc = _gd_bwd_value_desc(b, node->inputs[0]);
    const gd_tensor_desc *out_desc = _gd_bwd_value_desc(b, node->outputs[0]);
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    gd_tensor *dx = NULL;
    gd_status status = GD_OK;

    if (go == NULL) {
        return GD_OK;
    }
    if (!_gd_dtype_is_float(in_desc->dtype) || !_gd_dtype_is_float(out_desc->dtype)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "cast backward supports floating casts only");
    }
    status = gd_cast(_gd_bwd_context(b), go, in_desc->dtype, &dx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, node->inputs[0], dx);
    gd_tensor_release(dx);
    return status;
}

const _gd_bwd_rule _gd_bwd_rule_cast = {
    .op = _GD_OP_CAST,
    .fn = cast_backward,
    .unsupported_reason = NULL,
};
