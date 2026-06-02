#include "../op_impl.h"

const _gd_op_def _gd_opdef_adamw_step_amp = {
    .kind = _GD_OP_ADAMW_STEP_AMP,
    .name = "adamw_step_amp",
    .min_inputs = 6,
    .max_inputs = 8,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
