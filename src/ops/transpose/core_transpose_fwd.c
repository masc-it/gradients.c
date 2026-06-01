#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status transpose_meta(const gd_tensor_desc *const *inputs,
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
    return _gd_meta_transpose(inputs[0], attrs->perm, attrs->perm_ndim, &outputs[0]);
}

const _gd_op_def _gd_opdef_transpose = {
    .kind = _GD_OP_TRANSPOSE,
    .name = "transpose",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = transpose_meta,
};

gd_status gd_transpose(gd_context *ctx,
                       gd_tensor *x,
                       const int *perm,
                       int ndim,
                       gd_tensor **out)
{
    _gd_op_attrs attrs = {0};
    int i = 0;

    if (x == NULL || perm == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_transpose argument is NULL");
    }
    *out = NULL;
    if (ndim < 0 || ndim > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_transpose ndim is out of range");
    }
    attrs.perm_ndim = ndim;
    for (i = 0; i < ndim; ++i) {
        attrs.perm[i] = perm[i];
    }
    return _gd_emit_checked(ctx, _GD_OP_TRANSPOSE, &x, 1, &attrs, out, 1);
}
