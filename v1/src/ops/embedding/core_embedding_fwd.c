#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status embedding_meta(const gd_tensor_desc *const *inputs,
                                int n_inputs,
                                _gd_op_attrs *attrs,
                                gd_tensor_desc *outputs,
                                int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    (void)attrs;
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_embedding(inputs[0], inputs[1], &outputs[0]);
}

const _gd_op_def _gd_opdef_embedding = {
    .kind = _GD_OP_EMBEDDING,
    .name = "embedding",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = embedding_meta,
};

gd_status gd_embedding(gd_context *ctx,
                       gd_tensor *table,
                       gd_tensor *ids,
                       gd_tensor **out)
{
    gd_tensor *inputs[2] = {table, ids};

    if (table == NULL || ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_embedding argument is NULL");
    }
    *out = NULL;
    return _gd_emit_checked(ctx, _GD_OP_EMBEDDING, inputs, 2, NULL, out, 1);
}
