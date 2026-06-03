#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status cast_meta(const gd_tensor_desc *const *inputs,
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
    return _gd_meta_cast(inputs[0], attrs->cast_dtype, &outputs[0]);
}

const _gd_op_def _gd_opdef_cast = {
    .kind = _GD_OP_CAST,
    .name = "cast",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = cast_meta,
};

gd_status gd_cast(gd_context *ctx, gd_tensor *x, gd_dtype dtype, gd_tensor **out)
{
    _gd_op_attrs attrs = {0};

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_cast argument is NULL");
    }
    *out = NULL;
    attrs.cast_dtype = dtype;
    return _gd_emit_checked(ctx, _GD_OP_CAST, &x, 1, &attrs, out, 1);
}
