#include "../op_impl.h"

const _gd_op_def _gd_opdef_relu_bwd = {
    .kind = _GD_OP_RELU_BWD,
    .name = "relu_bwd",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_INTERNAL,
    .meta = _gd_meta_not_implemented,
};
