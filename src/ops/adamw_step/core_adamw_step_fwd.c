#include "../op_impl.h"

const _gd_op_def _gd_opdef_adamw_step = {
    .kind = _GD_OP_ADAMW_STEP,
    .name = "adamw_step",
    .min_inputs = 5,
    .max_inputs = 6,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
