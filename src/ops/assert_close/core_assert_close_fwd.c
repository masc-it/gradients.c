#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/graph.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status assert_close_meta(const gd_tensor_desc *const *inputs,
                                   int n_inputs,
                                   _gd_op_attrs *attrs,
                                   gd_tensor_desc *outputs,
                                   int *n_outputs)
{
    int i = 0;

    (void)n_inputs;
    (void)outputs;
    if (inputs[0]->dtype != GD_DTYPE_F32 || inputs[1]->dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_DTYPE, "assert_close requires F32 inputs");
    }
    if (!gd_device_equal(inputs[0]->device, inputs[1]->device)) {
        return _gd_error(GD_ERR_DEVICE, "assert_close inputs must share a device");
    }
    if (inputs[0]->ndim != inputs[1]->ndim) {
        return _gd_error(GD_ERR_SHAPE, "assert_close inputs must have equal shape");
    }
    for (i = 0; i < inputs[0]->ndim; ++i) {
        if (inputs[0]->sizes[i] != inputs[1]->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "assert_close inputs must have equal shape");
        }
    }
    if (attrs->atol < 0.0F || attrs->rtol < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tolerances must be nonnegative");
    }
    return _gd_meta_set_output_count(0, n_outputs);
}

const _gd_op_def _gd_opdef_assert_close = {
    .kind = _GD_OP_ASSERT_CLOSE,
    .name = "assert_close",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 0,
    .flags = GD_OPF_PUBLIC | GD_OPF_SIDE_EFFECT | GD_OPF_DEBUG,
    .meta = assert_close_meta,
};

gd_status gd_assert_close(gd_context *ctx,
                          gd_tensor *a,
                          gd_tensor *b,
                          float atol,
                          float rtol)
{
    gd_tensor *inputs[2] = {a, b};
    _gd_op_attrs attrs = {0};

    if (ctx == NULL || a == NULL || b == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_assert_close argument is NULL");
    }
    if (atol < 0.0F || rtol < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tolerances must be nonnegative");
    }
    attrs.atol = atol;
    attrs.rtol = rtol;
    return _gd_emit_checked(ctx, _GD_OP_ASSERT_CLOSE, inputs, 2, &attrs, NULL, 0);
}
