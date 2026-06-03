#include "../op_impl.h"

const _gd_op_def _gd_opdef_amp_unscale_grad = {
    .kind = _GD_OP_AMP_UNSCALE_GRAD,
    .name = "amp_unscale_grad",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 0,
    .flags = GD_OPF_INTERNAL | GD_OPF_MUTATES,
    .meta = _gd_meta_not_implemented,
};
