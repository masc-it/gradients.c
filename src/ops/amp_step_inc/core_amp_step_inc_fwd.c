#include "../op_impl.h"

const _gd_op_def _gd_opdef_amp_step_inc = {
    .kind = _GD_OP_AMP_STEP_INC,
    .name = "amp_step_inc",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
