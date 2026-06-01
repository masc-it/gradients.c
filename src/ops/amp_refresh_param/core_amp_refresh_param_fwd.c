#include "../op_impl.h"

const _gd_op_def _gd_opdef_amp_refresh_param = {
    .kind = _GD_OP_AMP_REFRESH_PARAM,
    .name = "amp_refresh_param",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
