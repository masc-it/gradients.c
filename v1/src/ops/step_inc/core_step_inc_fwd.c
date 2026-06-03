#include "../op_impl.h"

const _gd_op_def _gd_opdef_step_inc = {
    .kind = _GD_OP_STEP_INC,
    .name = "step_inc",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
