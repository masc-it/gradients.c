#include "../op_impl.h"

const _gd_op_def _gd_opdef_cross_entropy_bwd = {
    .kind = _GD_OP_CROSS_ENTROPY_BWD,
    .name = "cross_entropy_bwd",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 1,
    .flags = GD_OPF_INTERNAL,
    .meta = _gd_meta_not_implemented,
};
