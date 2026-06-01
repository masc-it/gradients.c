#include "../op_impl.h"
#include "../meta_common.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status reduce_to_meta(const gd_tensor_desc *const *inputs,
                                int n_inputs,
                                _gd_op_attrs *attrs,
                                gd_tensor_desc *outputs,
                                int *n_outputs)
{
    gd_status status = GD_OK;

    (void)n_inputs;
    if (attrs == NULL || !attrs->has_reduce_to_desc) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "reduce_to target descriptor is missing");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return _gd_meta_reduce_to(inputs[0], &attrs->reduce_to_desc, &outputs[0]);
}

const _gd_op_def _gd_opdef_reduce_to = {
    .kind = _GD_OP_REDUCE_TO,
    .name = "reduce_to",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_INTERNAL,
    .meta = reduce_to_meta,
};
