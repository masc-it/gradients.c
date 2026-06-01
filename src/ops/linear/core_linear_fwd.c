#include "../op_impl.h"

const _gd_op_def _gd_opdef_linear = {
    .kind = _GD_OP_LINEAR,
    .name = "linear",
    .min_inputs = 2,
    .max_inputs = 3,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = _gd_meta_not_implemented,
};
