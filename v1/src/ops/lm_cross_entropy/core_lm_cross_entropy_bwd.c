#include "../op_impl.h"

const _gd_op_def _gd_opdef_lm_cross_entropy_bwd = {
    .kind = _GD_OP_LM_CROSS_ENTROPY_BWD,
    .name = "lm_cross_entropy_bwd",
    .min_inputs = 6,
    .max_inputs = 6,
    .n_outputs = 2,
    .flags = GD_OPF_INTERNAL | GD_OPF_AUX_OUTS,
    .meta = _gd_meta_not_implemented,
};
