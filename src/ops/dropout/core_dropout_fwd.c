#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include <math.h>

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status dropout_meta(const gd_tensor_desc *const *inputs,
                              int n_inputs,
                              _gd_op_attrs *attrs,
                              gd_tensor_desc *outputs,
                              int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    if (attrs == NULL || !isfinite(attrs->scale) || attrs->scale < 0.0F ||
        attrs->scale >= 1.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dropout probability must satisfy 0 <= p < 1");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_unary_float(inputs[0], &outputs[0]);
}

const _gd_op_def _gd_opdef_dropout = {
    .kind = _GD_OP_DROPOUT,
    .name = "dropout",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = dropout_meta,
};

gd_status gd_dropout(gd_context *ctx,
                     gd_tensor *x,
                     float p,
                     uint64_t seed,
                     bool training,
                     gd_tensor **out)
{
    _gd_op_attrs attrs = {0};

    (void)ctx;
    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_dropout argument is NULL");
    }
    *out = NULL;
    if (!isfinite(p) || p < 0.0F || p >= 1.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dropout probability must satisfy 0 <= p < 1");
    }
    if (!training || p == 0.0F) {
        gd_status status = gd_tensor_retain(x);
        if (status != GD_OK) {
            return status;
        }
        *out = x;
        return GD_OK;
    }
    attrs.scale = p;
    attrs.dropout_seed = seed != 0U ? seed : UINT64_C(0x9e3779b97f4a7c15);
    return _gd_emit_checked(ctx, _GD_OP_DROPOUT, &x, 1, &attrs, out, 1);
}
