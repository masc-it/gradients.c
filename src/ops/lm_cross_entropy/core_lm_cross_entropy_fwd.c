#include "../op_impl.h"

const _gd_op_def _gd_opdef_lm_cross_entropy = {
    .kind = _GD_OP_LM_CROSS_ENTROPY,
    .name = "lm_cross_entropy",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 3,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF | GD_OPF_AUX_OUTS,
    .meta = _gd_meta_not_implemented,
};
