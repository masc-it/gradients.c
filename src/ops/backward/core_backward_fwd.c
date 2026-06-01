#include "../op_impl.h"

const _gd_op_def _gd_opdef_backward = {
    .kind = _GD_OP_BACKWARD,
    .name = "backward",
    .min_inputs = 1,
    .max_inputs = 1,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_SIDE_EFFECT,
    .meta = _gd_meta_not_implemented,
};
