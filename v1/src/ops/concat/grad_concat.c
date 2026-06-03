#include "../grad_impl.h"

static gd_status concat_backward(_gd_bwd_ctx *b, const _gd_node *node)
{
    gd_status status = GD_OK;
    gd_tensor *go = _gd_bwd_grad(b, node->outputs[0]);
    int64_t start = 0;
    int i = 0;

    if (go == NULL) {
        return GD_OK;
    }
    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *in_desc = _gd_bwd_value_desc(b, node->inputs[i]);
        gd_tensor_desc dx_desc;
        gd_tensor *dx = NULL;
        gd_tensor *slice_inputs[1] = {go};
        _gd_op_attrs attrs = {0};

        if (in_desc == NULL) {
            return _gd_error(GD_ERR_INTERNAL, "concat backward input desc is NULL");
        }
        status = gd_tensor_desc_contiguous(in_desc->dtype, in_desc->device,
                                           in_desc->ndim, in_desc->sizes, &dx_desc);
        if (status != GD_OK) {
            return status;
        }
        attrs.dim = node->attrs.dim;
        attrs.slice_start = start;
        attrs.slice_len = in_desc->sizes[node->attrs.dim];
        status = _gd_bwd_emit(b, _GD_OP_SLICE, slice_inputs, 1, &attrs, &dx_desc, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_bwd_accumulate(b, node->inputs[i], dx);
        gd_tensor_release(dx);
        if (status != GD_OK) {
            return status;
        }
        start += attrs.slice_len;
    }
    return GD_OK;
}

const _gd_bwd_rule _gd_bwd_rule_concat = {
    .op = _GD_OP_CONCAT,
    .fn = concat_backward,
    .unsupported_reason = NULL,
};
