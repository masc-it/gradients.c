#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/graph.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status assert_finite_meta(const gd_tensor_desc *const *inputs,
                                    int n_inputs,
                                    _gd_op_attrs *attrs,
                                    gd_tensor_desc *outputs,
                                    int *n_outputs)
{
    (void)n_inputs;
    (void)attrs;
    (void)outputs;
    if (inputs[0]->dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_DTYPE, "assert_finite requires F32 input");
    }
    return _gd_meta_set_output_count(0, n_outputs);
}

const _gd_op_def _gd_opdef_assert_finite = {
    .kind = _GD_OP_ASSERT_FINITE,
    .name = "assert_finite",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 0,
    .flags = GD_OPF_PUBLIC | GD_OPF_SIDE_EFFECT | GD_OPF_DEBUG,
    .meta = assert_finite_meta,
};

gd_status gd_assert_finite(gd_context *ctx, gd_tensor *tensor)
{
    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_assert_finite argument is NULL");
    }
    return _gd_emit_checked(ctx, _GD_OP_ASSERT_FINITE, &tensor, 1, NULL, NULL, 0);
}
