#include "../op_impl.h"

const _gd_op_def _gd_opdef_optimizer_step = {
    .kind = _GD_OP_OPTIMIZER_STEP,
    .name = "optimizer_step",
    .min_inputs = 1,
    .max_inputs = 256,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
