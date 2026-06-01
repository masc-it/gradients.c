#include "../op_impl.h"

const _gd_op_def _gd_opdef_powlu = {
    .kind = _GD_OP_POWLU,
    .name = "powlu",
    .min_inputs = 2,
    .max_inputs = 2,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = _gd_meta_not_implemented,
};
