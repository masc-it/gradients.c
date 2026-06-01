#include "../op_impl.h"

const _gd_op_def _gd_opdef_powlu_bwd = {
    .kind = _GD_OP_POWLU_BWD,
    .name = "powlu_bwd",
    .min_inputs = 3,
    .max_inputs = 3,
    .n_outputs = 2,
    .flags = GD_OPF_INTERNAL | GD_OPF_AUX_OUTS,
    .meta = _gd_meta_not_implemented,
};
