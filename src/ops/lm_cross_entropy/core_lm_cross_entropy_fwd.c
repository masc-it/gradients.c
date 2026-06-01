#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status lm_cross_entropy_meta(const gd_tensor_desc *const *inputs,
                                       int n_inputs,
                                       _gd_op_attrs *attrs,
                                       gd_tensor_desc *outputs,
                                       int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    (void)attrs;
    status = _gd_meta_set_output_count(3, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_lm_cross_entropy(inputs[0], inputs[1], inputs[2], &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_desc_contiguous(GD_DTYPE_F32, inputs[0]->device,
                                       inputs[2]->ndim, inputs[2]->sizes, &outputs[1]);
    if (status != GD_OK) {
        return status;
    }
    outputs[2] = outputs[1];
    return GD_OK;
}

const _gd_op_def _gd_opdef_lm_cross_entropy = {
    .kind = _GD_OP_LM_CROSS_ENTROPY,
    .name = "lm_cross_entropy",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 3,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF | GD_OPF_AUX_OUTS,
    .meta = lm_cross_entropy_meta,
};

gd_status gd_lm_cross_entropy(gd_context *ctx,
                              gd_tensor *hidden,
                              gd_tensor *weight,
                              gd_tensor *targets,
                              gd_tensor **loss)
{
    gd_status status = GD_OK;
    gd_tensor *inputs[3] = {hidden, weight, targets};
    gd_tensor *outs[3] = {NULL, NULL, NULL};

    if (hidden == NULL || weight == NULL || targets == NULL || loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_lm_cross_entropy argument is NULL");
    }
    *loss = NULL;
    status = _gd_emit_checked(ctx, _GD_OP_LM_CROSS_ENTROPY, inputs, 3, NULL, outs, 3);
    if (status != GD_OK) {
        return status;
    }
    *loss = outs[0];
    gd_tensor_release(outs[1]);
    gd_tensor_release(outs[2]);
    return GD_OK;
}
