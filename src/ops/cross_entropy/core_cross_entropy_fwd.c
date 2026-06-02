#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status cross_entropy_meta(const gd_tensor_desc *const *inputs,
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
    return _gd_meta_cross_entropy(inputs[0], inputs[1], attrs->dim, &attrs->dim,
                                  &outputs[0]);
}

const _gd_op_def _gd_opdef_cross_entropy = {
    .kind = _GD_OP_CROSS_ENTROPY,
    .name = "cross_entropy",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = cross_entropy_meta,
};

gd_status gd_cross_entropy(gd_context *ctx,
                           gd_tensor *logits,
                           gd_tensor *targets,
                           int class_dim,
                           gd_tensor **loss)
{
    gd_cross_entropy_desc desc;

    desc.class_dim = class_dim;
    desc.has_ignore_index = false;
    desc.ignore_index = 0;
    return gd_cross_entropy_ex(ctx, &desc, logits, targets, loss);
}

gd_status gd_cross_entropy_ex(gd_context *ctx,
                              const gd_cross_entropy_desc *desc,
                              gd_tensor *logits,
                              gd_tensor *targets,
                              gd_tensor **loss)
{
    gd_tensor *inputs[2] = {logits, targets};
    _gd_op_attrs attrs = {0};

    if (desc == NULL || logits == NULL || targets == NULL || loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_cross_entropy_ex argument is NULL");
    }
    *loss = NULL;
    attrs.dim = desc->class_dim;
    attrs.has_ignore_index = desc->has_ignore_index;
    attrs.ignore_index = desc->ignore_index;
    return _gd_emit_checked(ctx, _GD_OP_CROSS_ENTROPY, inputs, 2, &attrs, loss, 1);
}
