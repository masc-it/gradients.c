#include "../op_impl.h"
#include "../meta_common.h"

#include <math.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status powlu_meta(const gd_tensor_desc *const *inputs,
                            int n_inputs,
                            _gd_op_attrs *attrs,
                            gd_tensor_desc *outputs,
                            int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    if (attrs == NULL || !isfinite(attrs->powlu_m) || attrs->powlu_m <= 0.0F ||
        attrs->powlu_m >= 10.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "powlu m must satisfy 0 < m < 10");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_powlu(inputs[0], inputs[1], &outputs[0]);
}

const _gd_op_def _gd_opdef_powlu = {
    .kind = _GD_OP_POWLU,
    .name = "powlu",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = powlu_meta,
};

gd_status gd_powlu(gd_context *ctx,
                   gd_tensor *x1,
                   gd_tensor *x2,
                   float m,
                   gd_tensor **out)
{
    gd_tensor *inputs[2] = {x1, x2};
    _gd_op_attrs attrs = {0};

    if (x1 == NULL || x2 == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_powlu argument is NULL");
    }
    *out = NULL;
    if (!isfinite(m) || m <= 0.0F || m >= 10.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "powlu m must satisfy 0 < m < 10");
    }
    attrs.powlu_m = m;
    return _gd_emit_checked(ctx, _GD_OP_POWLU, inputs, 2, &attrs, out, 1);
}
