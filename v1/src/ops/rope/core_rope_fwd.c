#include "../op_impl.h"
#include "../meta_common.h"

#include <limits.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status rope_meta(const gd_tensor_desc *const *inputs,
                           int n_inputs,
                           _gd_op_attrs *attrs,
                           gd_tensor_desc *outputs,
                           int *n_outputs)
{
    gd_status status = GD_OK;
    int64_t head_dim = 0;
    int n_dims = 0;

    (void)n_inputs;
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_rope(inputs[0], inputs[1], &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    head_dim = inputs[0]->sizes[inputs[0]->ndim - 1];
    if (head_dim > (int64_t)INT_MAX) {
        return _gd_error(GD_ERR_SHAPE, "rope head_dim exceeds int range");
    }
    n_dims = attrs->rope_n_dims > 0 ? attrs->rope_n_dims : (int)head_dim;
    if (n_dims % 2 != 0 || n_dims > (int)head_dim) {
        return _gd_error(GD_ERR_SHAPE, "rope n_dims must be even and <= head_dim");
    }
    attrs->rope_n_dims = n_dims;
    if (attrs->rope_theta <= 0.0F) {
        attrs->rope_theta = 10000.0F;
    }
    return GD_OK;
}

const _gd_op_def _gd_opdef_rope = {
    .kind = _GD_OP_ROPE,
    .name = "rope",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = rope_meta,
};

gd_status gd_rope(gd_context *ctx,
                  gd_tensor *x,
                  gd_tensor *pos_ids,
                  const gd_rope_config *config,
                  gd_tensor **out)
{
    gd_tensor *inputs[2] = {x, pos_ids};
    _gd_op_attrs attrs = {0};

    if (x == NULL || pos_ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_rope argument is NULL");
    }
    *out = NULL;
    attrs.rope_theta = (config != NULL && config->theta > 0.0F) ? config->theta : 10000.0F;
    attrs.rope_n_dims = (config != NULL) ? config->n_dims : 0;
    attrs.rope_interleaved = (config != NULL && config->interleaved) ? 1 : 0;
    return _gd_emit_checked(ctx, _GD_OP_ROPE, inputs, 2, &attrs, out, 1);
}
