#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status rms_norm_meta(const gd_tensor_desc *const *inputs,
                               int n_inputs,
                               _gd_op_attrs *attrs,
                               gd_tensor_desc *outputs,
                               int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *x = inputs[0];
    const gd_tensor_desc *weight = inputs[1];

    (void)n_inputs;
    if (attrs->eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm eps must be positive");
    }
    if (x->dtype != weight->dtype) {
        return _gd_error(GD_ERR_DTYPE, "rms_norm input and weight must share dtype");
    }
    status = _gd_meta_require_same_device(x, weight, "rms_norm inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (x->ndim < 1) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm input must have rank >= 1");
    }
    if (weight->ndim != 1 || weight->sizes[0] != x->sizes[x->ndim - 1]) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm weight must be [last_dim]");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_unary_float(x, &outputs[0]);
}

const _gd_op_def _gd_opdef_rms_norm = {
    .kind = _GD_OP_RMS_NORM,
    .name = "rms_norm",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = rms_norm_meta,
};

gd_status gd_rms_norm(gd_context *ctx,
                      gd_tensor *x,
                      gd_tensor *weight,
                      float eps,
                      gd_tensor **out)
{
    gd_tensor *inputs[2] = {x, weight};
    _gd_op_attrs attrs = {0};

    if (x == NULL || weight == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_rms_norm argument is NULL");
    }
    *out = NULL;
    if (eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm eps must be positive");
    }
    attrs.eps = eps;
    return _gd_emit_checked(ctx, _GD_OP_RMS_NORM, inputs, 2, &attrs, out, 1);
}
