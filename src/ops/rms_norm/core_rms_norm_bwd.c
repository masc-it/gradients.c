#include "../op_impl.h"

const _gd_op_def _gd_opdef_rms_norm_bwd = {
    .kind = _GD_OP_RMS_NORM_BWD,
    .name = "rms_norm_bwd",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 1,
    .flags = GD_OPF_INTERNAL,
    .meta = _gd_meta_not_implemented,
};
