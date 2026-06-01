#include "../op_impl.h"

const _gd_op_def _gd_opdef_rms_norm = {
    .kind = _GD_OP_RMS_NORM,
    .name = "rms_norm",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = _gd_meta_not_implemented,
};
