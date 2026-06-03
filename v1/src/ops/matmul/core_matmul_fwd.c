#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status matmul_meta(const gd_tensor_desc *const *inputs,
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
    status = _gd_meta_matmul(inputs[0], inputs[1], attrs->trans_a, attrs->trans_b,
                             &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    /* Internal AMP grad sentinel: F16 inputs, F32 output, no input upcast nodes. */
    if (inputs[0]->dtype == GD_DTYPE_F16 && inputs[1]->dtype == GD_DTYPE_F16 &&
        attrs->compute.compute_dtype == GD_DTYPE_F32 &&
        attrs->compute.accum_dtype == GD_DTYPE_INVALID) {
        outputs[0].dtype = GD_DTYPE_F32;
    }
    return GD_OK;
}

const _gd_op_def _gd_opdef_matmul = {
    .kind = _GD_OP_MATMUL,
    .name = "matmul",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF | GD_OPF_BROADCAST,
    .meta = matmul_meta,
};

gd_status gd_matmul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out)
{
    return gd_matmul_ex(ctx, NULL, a, b, out);
}

gd_status gd_matmul_ex(gd_context *ctx,
                       const gd_matmul_desc *desc,
                       gd_tensor *a,
                       gd_tensor *b,
                       gd_tensor **out)
{
    gd_tensor *inputs[2] = {a, b};
    _gd_op_attrs attrs = {0};

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_matmul_ex argument is NULL");
    }
    *out = NULL;
    if (desc != NULL) {
        attrs.trans_a = desc->trans_a;
        attrs.trans_b = desc->trans_b;
        attrs.compute = desc->compute;
    } else {
        attrs.compute = gd_context_compute_policy(ctx);
    }
    return _gd_emit_checked(ctx, _GD_OP_MATMUL, inputs, 2, &attrs, out, 1);
}
