#include "../op_impl.h"

const _gd_op_def _gd_opdef_matmul = {
    .kind = _GD_OP_MATMUL,
    .name = "matmul",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF | GD_OPF_BROADCAST,
    .meta = _gd_meta_not_implemented,
};
