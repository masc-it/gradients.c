#include "../op_impl.h"

const _gd_op_def _gd_opdef_zero_grad = {
    .kind = _GD_OP_ZERO_GRAD,
    .name = "zero_grad",
    .min_inputs = 1,
    .max_inputs = 256,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES | GD_OPF_PSEUDO,
    .meta = _gd_meta_not_implemented,
};
