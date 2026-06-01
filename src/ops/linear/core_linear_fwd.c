#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status linear_meta(const gd_tensor_desc *const *inputs,
                             int n_inputs,
                             _gd_op_attrs *attrs,
                             gd_tensor_desc *outputs,
                             int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *bias = n_inputs == 3 ? inputs[2] : NULL;

    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    attrs->has_bias = n_inputs == 3;
    return _gd_meta_linear(inputs[0], inputs[1], bias, attrs->trans_b, &outputs[0]);
}

const _gd_op_def _gd_opdef_linear = {
    .kind = _GD_OP_LINEAR,
    .name = "linear",
    .min_inputs = 2,
    .max_inputs = 3,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = linear_meta,
};

gd_status gd_linear(gd_context *ctx,
                    gd_tensor *x,
                    gd_tensor *w,
                    gd_tensor *bias,
                    gd_tensor **out)
{
    return gd_linear_ex(ctx, NULL, x, w, bias, out);
}

gd_status gd_linear_ex(gd_context *ctx,
                       const gd_linear_desc *desc,
                       gd_tensor *x,
                       gd_tensor *w,
                       gd_tensor *bias,
                       gd_tensor **out)
{
    gd_tensor *inputs[3];
    _gd_op_attrs attrs = {0};
    int n_inputs = 2;

    if (x == NULL || w == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_linear_ex argument is NULL");
    }
    *out = NULL;
    if (desc != NULL) {
        attrs.trans_b = desc->trans_w;
        attrs.compute = desc->compute;
    } else {
        attrs.compute = gd_context_compute_policy(ctx);
    }
    inputs[0] = x;
    inputs[1] = w;
    if (bias != NULL) {
        inputs[2] = bias;
        n_inputs = 3;
    }
    return _gd_emit_checked(ctx, _GD_OP_LINEAR, inputs, n_inputs, &attrs, out, 1);
}
