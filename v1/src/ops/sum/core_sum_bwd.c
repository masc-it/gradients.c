#include "../op_impl.h"

const _gd_op_def _gd_opdef_sum_bwd = {
    .kind = _GD_OP_SUM_BWD,
    .name = "sum_bwd",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_INTERNAL,
    .meta = _gd_meta_not_implemented,
};
