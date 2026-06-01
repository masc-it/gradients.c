#include "../op_impl.h"

const _gd_op_def _gd_opdef_transpose = {
    .kind = _GD_OP_TRANSPOSE,
    .name = "transpose",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = _gd_meta_not_implemented,
};
