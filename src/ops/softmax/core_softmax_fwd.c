#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status softmax_meta(const gd_tensor_desc *const *inputs,
                              int n_inputs,
                              _gd_op_attrs *attrs,
                              gd_tensor_desc *outputs,
                              int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_softmax(inputs[0], attrs->dim, &attrs->dim, &outputs[0]);
}

const _gd_op_def _gd_opdef_softmax = {
    .kind = _GD_OP_SOFTMAX,
    .name = "softmax",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = softmax_meta,
};

gd_status gd_softmax(gd_context *ctx, gd_tensor *x, int dim, gd_tensor **out)
{
    _gd_op_attrs attrs = {0};

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_softmax argument is NULL");
    }
    *out = NULL;
    attrs.dim = dim;
    return _gd_emit_checked(ctx, _GD_OP_SOFTMAX, &x, 1, &attrs, out, 1);
}
